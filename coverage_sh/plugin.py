#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove

from __future__ import annotations

import contextlib
import inspect
import sys
import os
import re
import io
import logging
import itertools
import functools
import fnmatch
import string
import stat
import subprocess
import selectors
import threading
from types import FrameType
from typing import TYPE_CHECKING, TypeVar, Any, IO, Literal, cast
from collections.abc import Callable, Collection, Iterable, Iterator
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from socket import gethostname
from time import sleep

import coverage
import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from tree_sitter_languages import get_parser

from tree_sitter import Parser, Node
from coverage.types import TConfigurable, TArc, TLineNo

if sys.version_info < (3, 9): # pragma: no-cover-if-python-gte-39
    from typing import Dict, List, Set

    LineData = Dict[str, Set[TLineNo]]
    ArcData = Dict[str, List[TArc]]
    PreviousLineData = Dict[str, TLineNo]
else: # pragma: no-cover-if-python-lt-39
    LineData = dict[str, set[TLineNo]]
    ArcData = dict[str, list[TArc]]
    PreviousLineData = dict[str, TLineNo]

if sys.version_info < (3, 10): # pragma: no-cover-if-python-gte-310
    T = TypeVar("T")
    def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
else: # pragma: no-cover-if-python-lt-310
    from itertools import pairwise

TMP_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))  # noqa: S108
TRACEFILE_PREFIX = "shelltrace"
EXECUTABLE_NODE_TYPES = {
    "subshell",
    "redirected_statement",
    "variable_assignment",
    "variable_assignments",
    "command",
    "declaration_command",
    "unset_command",
    "test_command",
    "negated_command",
    "for_statement",
    "c_style_for_statement",
    "while_statement",
    "if_statement",
    "case_statement",
    "pipeline",
    "list",
}
SUPPORTED_MIME_TYPES = {"text/x-shellscript"}

logger = logging.getLogger(__name__)

class ShellFileParser:
    filename: str
    source: str
    exclude: str | None

    _parser: Parser = get_parser("bash")

    @functools.cached_property
    def logger(self) -> logging.Logger:
        return logger.getChild("parser")

    def __init__(self, filename: str, source: str, exclude: str | None = None):
        self.filename = filename
        self.source = source
        self.exclude = exclude
        self._tree = self._parser.parse(source.encode("utf-8"))
        self.arc_translations: dict[TArc, set[TArc]] = {}

    @functools.cached_property
    def lines(self) -> set[TLineNo]:
        executable_lines: set[TLineNo] = set()

        def parse_ast(node: Node) -> None:
            if node.is_named and node.type in EXECUTABLE_NODE_TYPES:
                executable_lines.add(node.start_point[0] + 1)

            for child in node.children:
                parse_ast(child)

        parse_ast(self._tree.root_node)
        self.logger.debug(f"[{self.filename}] lines: {executable_lines}")
        return executable_lines

    @functools.cached_property
    def excluded_lines(self) -> set[TLineNo]:
        excluded_lines: set[TLineNo] = set()

        for lineno, line in enumerate(self.source.split("\n"), 1):
            if self.exclude is not None and re.search(self.exclude, line):
                excluded_lines.add(lineno)

        return excluded_lines

    @property
    def bash_supports_case_item_trace(self) -> bool:
        # NOTE: bash's trace does not support showing case_items as of 5.2
        return False

    @property
    def bash_supports_case_item_termination_trace(self) -> bool:
        # NOTE: bash's trace does not support showing case_items as of 5.2
        return False

    def nearest_next_line(self, target: int) -> int:
        return next((lineno for lineno in self.lines if lineno >= target), -1)

    @functools.cached_property
    def arcs(self) -> list[TArc]:
        arcs: list[TArc] = list()

        def of(*types: str, match: Literal["all", "any"] = "any") -> Callable[[Node], bool]:
            return lambda node: {"all": all, "any": any}[match](fnmatch.fnmatch(node.type, type) for type in types)

        is_statement = of(*EXECUTABLE_NODE_TYPES)

        def not_of(*types: str, match: Literal["all", "any"] = "any") -> Callable[[Node], bool]:
            negations: dict[Literal["all", "any"], Literal["all", "any"]] = {"all": "any", "any": "all"}
            return lambda node: not of(*types, match=negations[match])(node)

        def parse_ast(node: Node) -> None:
            if node.type == "if_statement":
                self.logger.debug(f"if_statement: {node!r}")
                conditions = list(filter(of("command", "*_command", "compound_statement"), node.children_by_field_name("condition")))
                assert conditions, f"conditions are required as per the grammar: {node.children_by_field_name('condition')!r}"
                self.logger.debug(f"conditions: {conditions}")

                elif_clauses = list(filter(of("elif_clause"), node.children))
                self.logger.debug(f"elif_clauses: {elif_clauses}")
                else_clauses = list(filter(of("else_clause"), node.children))
                assert len(else_clauses) <= 1
                self.logger.debug(f"else_clauses: {else_clauses}")
                for clause, statements in { node: list(filter(not_of("*_clause"), node.children)), **{elif_clause: elif_clause.children for elif_clause in elif_clauses }}.items():
                    self.logger.debug(f"statements: {statements}")
                    arcs.append((clause.start_point[0] + 1, statements[0].start_point[0] + 1))
                    arcs.append((statements[-1].end_point[1] + 1, node.end_point[1] + 1))
                for source, destination in pairwise([node] + elif_clauses + else_clauses):
                    arcs.append((source.start_point[0] + 1, self.nearest_next_line(destination.start_point[0] + 1)))

            elif node.type == "c_style_for_statement":
                self.logger.debug(f"c_style_for_statement: {node!r}")
                body = node.child_by_field_name("body")
                self.logger.debug(f"body: {body}")
                conditions = node.children_by_field_name("condition")
                self.logger.debug(f"conditions: {conditions}")
                assert conditions, "conditions are required as per the grammar"
                # TODO

            elif node.type == "for_statement":
                self.logger.debug(f"for_statement: {node!r}")
                body = node.child_by_field_name("body")
                self.logger.debug(f"body: {body}")
                variable = node.child_by_field_name("variable")
                self.logger.debug(f"variable: {variable}")
                values = node.children_by_field_name("value")
                self.logger.debug(f"values: {values}")
                # TODO

            elif node.type == "while_statement":
                self.logger.debug(f"while_statement: {node!r}")
                body = node.child_by_field_name("body")
                self.logger.debug(f"body: {body}")
                conditions = node.children_by_field_name("condition")
                assert conditions, "conditions are required as per the grammar"
                self.logger.debug(f"conditions: {conditions}")
                # TODO

            elif node.type == "case_statement":
                self.logger.debug(f"case_statement: {node!r}")
                value = node.child_by_field_name("value")
                self.logger.debug(f"value: {value}")
                case_items = list(filter(of("case_item"), node.children))
                self.logger.debug(f"case_items: {case_items}")

                # TODO: find a way to be more precise on the branches for each condition in  each case_item

                # only bridge from case_statement to the first case_item
                arcs.append((node.start_point[0] + 1, case_items[0].start_point[0] + 1))

                for case_item in case_items:
                    if statements := list(filter(is_statement, case_item.children)):
                        # matched
                        arcs.append((case_item.start_point[0] + 1, statements[0].start_point[0] + 1))

                        if not self.bash_supports_case_item_trace: # pragma: no branch
                            # matched
                            self.arc_translations[(node.start_point[0] + 1, statements[0].start_point[0] + 1)] = {
                                # case_statement.start_point -> case_item
                                (node.start_point[0] + 1, case_item.start_point[0] + 1),
                                # case_item -> statements[0]
                                (case_item.start_point[0] + 1, statements[0].start_point[0] + 1),
                            }

                        # termination
                        if termination := case_item.child_by_field_name("termination"):
                            arcs.append((termination.end_point[0] + 1, self.nearest_next_line(node.end_point[0] + 1)))
                            if not self.bash_supports_case_item_termination_trace: # pragma: no branch
                                self.arc_translations[(statements[-1].end_point[0] + 1, self.nearest_next_line(node.end_point[0] + 1))] = {
                                    # statements[-1] -> case_statement.end_point
                                    (statements[-1].start_point[0] + 1, node.end_point[0] + 1),
                                    (node.end_point[0] + 1, self.nearest_next_line(node.end_point[0] + 1)),
                                }

                for source, destination in pairwise(case_items):
                    self.logger.debug(f"source: {source}, source.children: {source.children}")
                    # TODO: consider dropping unilt ) and taking while not
                    if statements := list(filter(is_statement, source.children)):
                        self.logger.debug(f"statements: {statements}")
                        # mismatch
                        arcs.append((source.start_point[0] + 1, destination.start_point[0] + 1))

                    # fallthrough
                    if fallthrough := source.child_by_field_name("fallthrough"):
                        arcs.append((fallthrough.end_point[0] + 1, destination.start_point[0] + 1))


                    assert fallthrough or source.child_by_field_name("termination"), f"case_item: {source!r} should have either fallthrough or termination"
            elif node.type == "negated_command":
                self.logger.debug(f"negated_command: {node!r}")
                _, child = node.children
                self.logger.debug(f"child: {child}")
                arcs.append((node.start_point[0] + 1, child.start_point[0] + 1))

            elif node.type == "ternary_expression":
                self.logger.debug(f"ternary_expression: {node!r}")
                condition = node.child_by_field_name("condition")
                assert condition, "condition is required as per the grammar"
                self.logger.debug(f"condition: {condition}")

                consequence = node.child_by_field_name("consequence")
                assert consequence, "consequence is required as per the grammar"
                arcs.append((node.start_point[0] + 1, consequence.start_point[0] + 1))
                self.logger.debug(f"consequence: {consequence}")
                alternative = node.child_by_field_name("alternative")
                assert alternative, "alternative is required as per the grammar"
                self.logger.debug(f"alternative: {alternative}")
                arcs.append((node.start_point[0] + 1, alternative.start_point[0] + 1))

            elif node.type == "test_command":
                self.logger.debug(f"test_command: {node!r}")
                # TODO

            elif node.type in EXECUTABLE_NODE_TYPES:
                self.logger.debug(f"{node.type}: {node!r}")
                arcs.append((node.start_point[0] + 1, self.nearest_next_line(node.end_point[0] + 1)))

            for child in node.children:
                parse_ast(child)

        parse_ast(self._tree.root_node)
        extra_lines = set((lineno, (source, destination)) for source, destination in arcs for lineno in (source, destination) if lineno not in self.lines)
        self.logger.debug(f"extra_lines: {extra_lines}")
        # assert not extra_lines, f"extra lines in arcs: {extra_lines!r}"
        # self.logger.debug(arcs)
        return arcs

    @functools.cached_property
    def exit_counts(self) -> dict[TLineNo, int]:
        lines: dict[TLineNo, list[int]] = defaultdict(list)

        for source, destination in self.arcs:
            lines[source].append(destination)

        return { source: len(destinations) for source, destinations in lines.items() }

    # def source_token_lines(self):
    #     pass

class ShellFileReporter(FileReporter):
    exclude_regex: str
    _content: str | None = None
    _parser: ShellFileParser | None = None

    def __init__(self, filename: str, exclude_regex: str = ""):
        super().__init__(filename)

        self.path = Path(filename)
        self.exclude_regex = exclude_regex

    @property
    def parser(self) -> ShellFileParser:
        if self._parser is None:
            self._parser = ShellFileParser(
                self.filename,
                self.source(),
                exclude=self.exclude_regex,
            )
        return self._parser

    def source(self) -> str:
        if self._content is None:
            if not self.path.is_file():
                return ""
            try:
                self._content = self.path.read_text()
            except UnicodeDecodeError:
                return ""

        return self._content

    def lines(self) -> set[TLineNo]:
        return self.parser.lines

    def translate_lines(self, lines: Iterable[TLineNo]) -> set[TLineNo]:
        translated_lines = set(lines)
        logger.debug(f"[{self.filename}] translated_lines: {translated_lines}")
        return translated_lines

    def excluded_lines(self) -> set[TLineNo]:
        return self.parser.excluded_lines

    def arcs(self) -> set[tuple[int, int]]:
        return set((source, destination) for source, destination in self.parser.arcs if source != destination)

    def translate_arcs(self, arcs: Iterable[TArc]) -> set[TArc]:
        translated_arcs = set().union(*(self.parser.arc_translations.get(arc, {arc}) for arc in arcs))
        logger.debug(f"[{self.filename}] translated_arcs: {translated_arcs}")
        unrecognized_arcs = set(self.parser.arcs) - translated_arcs 
        logger.warning(f"[{self.filename}] unrecognized_arcs: {unrecognized_arcs}")
        negatives = set(arc for arc in unrecognized_arcs if -1 in arc)
        final_arcs = translated_arcs - (unrecognized_arcs - negatives)
        logger.debug(f"[{self.filename} final_arcs: {final_arcs}")
        return final_arcs

    def exit_counts(self) -> dict[TLineNo, int]:
        return self.parser.exit_counts

    # def source_token_lines(self):
    #     return self.parser.source_token_lines()

def filename_suffix() -> str:
    die = Random(os.urandom(8))
    letters = string.ascii_uppercase + string.ascii_lowercase
    rolls = "".join(die.choice(letters) for _ in range(6))
    return f"{gethostname()}.{os.getpid()}.X{rolls}x"

@dataclass
class CovLineParser:
    line_data: LineData = field(default_factory=lambda: defaultdict(set))
    arc_data: ArcData = field(default_factory=lambda: defaultdict(list))
    _stack: list[tuple[str, TLineNo]] = field(default_factory=list)
    _previous_lines: list[tuple[str, str, TLineNo]] = field(default_factory=list)
    _last_line_fragment: str = ""

    def parse(self, buf: bytes) -> None:
        self._report_lines(list(self._buf_to_lines(buf)))

    def _buf_to_lines(self, buf: bytes) -> Iterator[str]:
        raw = self._last_line_fragment + buf.decode()
        self._last_line_fragment = ""

        for line in raw.splitlines(keepends=True):
            if line == "\n":
                pass
            elif line.endswith("\n"):
                yield line[:-1]
            else:
                self._last_line_fragment = line

    def _report_lines(self, lines: list[str]) -> None:
        if not lines:
            return

        for line in lines:
            if "COV:::" not in line:
                continue

            try:
                _, path_, funcname, lineno_, _ = line.split(":::", maxsplit=4)
                lineno = int(lineno_)
                path = Path(path_).absolute()
            except ValueError as e:
                raise ValueError(f"could not parse line {line}") from e

            self.line_data[str(path)].add(lineno)
            previous_lines = [lineno for filename, funcname_, lineno in self._previous_lines if (filename, funcname_) == (str(path), funcname)]
            previous_line = previous_lines[-1] if previous_lines else -1
            self.arc_data[str(path)].append((previous_line, lineno))
            self._previous_lines.append((str(path), funcname, lineno))

    def flush(self) -> None:
        self.parse(b"\n")

class CoverageWriter:
    plugin: ShellPlugin
    _coverage_data_path: Path

    def __init__(self, plugin: ShellPlugin, coverage_data_path: Path):
        self.plugin = plugin

        # pytest-cov uses the COV_CORE_DATAFILE env var to configure the datafile base path
        coverage_data_env_var = os.environ.get("COV_CORE_DATAFILE")
        if coverage_data_env_var is not None:
            coverage_data_path = Path(coverage_data_env_var).absolute()

        self._coverage_data_path = coverage_data_path

    def write(self, line_data: LineData, arc_data: dict[str, Collection[TArc]]) -> None:
        suffix_ = "sh." + filename_suffix()
        coverage_data = coverage.CoverageData(
            basename=self._coverage_data_path,
            suffix=suffix_,
            # TODO: set warn, debug and no_disk
        )

        coverage_data.add_file_tracers(
            {f: "coverage_sh.ShellPlugin" for f in set.union(set(arc_data.keys()), set(line_data.keys()))}
        )

        if self.plugin.branch:
            coverage_data.add_arcs(arc_data)
        else: # pragma: false negative
            coverage_data.add_lines(line_data)

        coverage_data.write()


class CoverageParserThread(threading.Thread):
    def __init__(
        self,
        coverage_writer: CoverageWriter,
        name: str | None = None,
        parser: CovLineParser | None = None,
    ) -> None:
        super().__init__(name=name)
        self._keep_running = True
        self._listening = False
        self._parser = parser or CovLineParser()
        self._coverage_writer = coverage_writer

        self.fifo_path: Path = TMP_PATH / f"coverage-sh.{filename_suffix()}.pipe"
        with contextlib.suppress(FileNotFoundError):
            self.fifo_path.unlink()
        os.mkfifo(self.fifo_path, mode=stat.S_IRUSR | stat.S_IWUSR)

    def start(self) -> None:
        super().start()
        while not self._listening:
            sleep(0.0001)

    def stop(self) -> None:
        self._keep_running = False

    def run(self) -> None:
        sel = selectors.DefaultSelector()
        while self._keep_running:
            # we need to keep reopening the fifo as long as the subprocess is running because multiple bash processes
            # might write EOFs to it
            fifo = os.open(self.fifo_path, flags=os.O_RDONLY | os.O_NONBLOCK)
            sel.register(fifo, selectors.EVENT_READ)
            self._listening = True

            eof = False
            data_incoming = True
            while not eof and data_incoming:
                events = sel.select(timeout=1)
                data_incoming = len(events) > 0
                for key, _ in events:
                    buf = os.read(key.fd, 2**10)
                    if not buf:
                        eof = True
                        break
                    self._parser.parse(buf)

            self._parser.flush()

            sel.unregister(fifo)
            os.close(fifo)

        self._coverage_writer.write(self._parser.line_data, { filename: set((source, destination) for source, destination in arcs if source != destination) for filename, arcs in self._parser.arc_data.items() })
        with contextlib.suppress(FileNotFoundError):
            self.fifo_path.unlink()


OriginalPopen = subprocess.Popen


def init_helper(fifo_path: Path) -> Path:
    helper_path = Path(TMP_PATH, f"coverage-sh.{filename_suffix()}.sh")
    helper_path.write_text(
        rf"""#!/usr/bin/env bash
PS4="COV:::\${{BASH_SOURCE}}:::\${{FUNCNAME:-<global>}}:::\${{LINENO}}:::"
exec {{BASH_XTRACEFD}}>>"{fifo_path!s}"
export BASH_XTRACEFD
set -x
"""
    )
    helper_path.chmod(mode=stat.S_IRUSR | stat.S_IWUSR)
    return helper_path


# the proper way to do this would be using OriginalPopen[Any] but that is not supported by python 3.8, so we jusrt
# ignore this for the time being
class PatchedPopen(OriginalPopen):  # type: ignore[type-arg]
    stdout: IO[str]
    stderr: IO[str]
    plugin: ShellPlugin | None = None

    def __init__(self, *args, plugin: ShellPlugin | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if plugin is None:
            # we are not recording coverage, so just act like the original Popen
            self._parser_thread = None
            super().__init__(*args, **kwargs)
            return

        self.plugin = plugin

        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        self._parser_thread = CoverageParserThread(
            coverage_writer=CoverageWriter(self.plugin, coverage_data_path=self.plugin.coverage_data_path),
            name="CoverageShCoverageParserThread(None)",
        )
        self._parser_thread.start()

        self._helper_path = init_helper(self._parser_thread.fifo_path)

        env = kwargs.get("env", os.environ.copy())
        env["BASH_ENV"] = str(self._helper_path)
        env["ENV"] = str(self._helper_path)
        kwargs["env"] = env

        super().__init__(**kwargs)

    def wait(self, timeout: float | None = None) -> int:
        retval = super().wait(timeout)
        if self._parser_thread is None:
            # no coverage recording was active during __init__
            return retval

        self._parser_thread.stop()
        self._parser_thread.join()
        with contextlib.suppress(FileNotFoundError):
            self._helper_path.unlink()
        return retval


class MonitorThread(threading.Thread):
    def __init__(
        self,
        parser_thread: CoverageParserThread,
        main_thread: threading.Thread | None = None,
        name: str | None = None,
        join_main_thread: bool = True,
    ) -> None:
        super().__init__(name=name)
        self._main_thread = main_thread or threading.main_thread()
        self.parser_thread = parser_thread
        self.join_main_thread = join_main_thread

    def run(self) -> None:
        if self.join_main_thread: # pragma: no cover
            self._main_thread.join()
        self.parser_thread.stop()
        self.parser_thread.join()


def _iterdir(path: Path) -> Iterator[Path]:
    """Recursively iterate over path. Race-condition safe(r) alternative to Path.rglob("*")"""
    for p in path.iterdir():
        yield p
        if p.is_dir():
            yield from _iterdir(p)


class ShellPlugin(CoveragePlugin):
    config: TConfigurable
    _helper_path: Path | None = None

    def __init__(self, options: dict[str, Any]):
        self.options = options

    def configure(self, config: TConfigurable) -> None:
        self.config = config

        self.coverage_data_path = Path(self.data_file).absolute()

        if self.options.get("cover_always", False):
            parser_thread = CoverageParserThread(
                coverage_writer=CoverageWriter(self, self.coverage_data_path),
                name=f"CoverageShCoverageParserThread({self.coverage_data_path!s})",
            )
            parser_thread.start()

            monitor_thread = MonitorThread(
                parser_thread=parser_thread,
                name="CoverageShMonitorThread",
                join_main_thread=self.options.get("join_main_thread", True),
            )
            monitor_thread.start()

            self._helper_path = init_helper(parser_thread.fifo_path)
            os.environ["BASH_ENV"] = str(self._helper_path)
            os.environ["ENV"] = str(self._helper_path)
            # do not remove - prevent having stale plugin when re-run
            subprocess.Popen = OriginalPopen # type: ignore[misc]
        else:
            # https://github.com/python/mypy/issues/1152
            subprocess.Popen = functools.partial(PatchedPopen, plugin=self)  # type: ignore[misc,assignment]

    @property
    def data_file(self) -> str:
        return cast(str, self.config.get_option("run:data_file"))

    @property
    def branch(self) -> bool:
        assert self.config is not None
        return cast(bool, self.config.get_option("run:branch"))

    def __del__(self) -> None:
        if self._helper_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._helper_path.unlink()

    @staticmethod
    def _is_relevant(path: Path) -> bool:
        return magic.from_file(path.resolve(), mime=True) in SUPPORTED_MIME_TYPES

    def file_tracer(self, filename: str) -> FileTracer | None:  # noqa: ARG002
        return None

    @property
    def exclude_regex(self) -> str:
        assert self.config is not None
        default_exclude = coverage.config.DEFAULT_EXCLUDE
        exclude_lines = cast(list[str], self.config.get_option("report:exclude_lines"))
        exclude_also = cast(list[str], self.config.get_option("report:exclude_also"))
        return coverage.misc.join_regex(default_exclude + exclude_lines + exclude_also)

    def file_reporter(
        self,
        filename: str,
    ) -> ShellFileReporter | str:
        return ShellFileReporter(filename, exclude_regex=self.exclude_regex)

    def find_executable_files(
        self,
        src_dir: str,
    ) -> Iterable[str]:
        for f in _iterdir(Path(src_dir)):
            # TODO: Use coverage's logic for figuring out if a file should be excluded
            if not (f.is_file() or (f.is_symlink() and f.resolve().is_file())) or any(
                p.startswith(".") for p in f.parts
            ):
                continue

            if self._is_relevant(f):
                yield str(f)
