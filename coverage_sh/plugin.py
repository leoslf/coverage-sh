#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove

from __future__ import annotations

import contextlib
import inspect
import sys
import os
import ctypes
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

import magic
import coverage
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

from coverage_sh.models import ShellFile
from coverage_sh.parsers import parse # ShellFileParser

TMP_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))  # noqa: S108
TRACEFILE_PREFIX = "shelltrace"
SUPPORTED_MIME_TYPES = {"text/x-shellscript"}

logger = logging.getLogger(__name__)

class ShellFileReporter(coverage.FileReporter):
    shell_file: ShellFile

    def __init__(self, filename: str, exclude_regex: str = ""):
        super().__init__(filename)
        self.shell_file = parse(filename, exclude_regex=exclude_regex)

    def lines(self) -> set[TLineNo]:
        return self.shell_file.lines

    def translate_lines(self, lines: Iterable[TLineNo]) -> set[TLineNo]:
        translated_lines = set(lines)
        logger.debug(f"[{os.path.basename(self.filename)}] translated_lines: {translated_lines}")
        return translated_lines

    def excluded_lines(self) -> set[TLineNo]:
        return self.shell_file.excluded_lines

    def arcs(self) -> set[tuple[int, int]]:
        return self.shell_file.arcs

    def translate_arcs(self, arcs: Iterable[TArc]) -> set[TArc]:
        logger.debug(f"[{os.path.basename(self.filename)}] arcs: {arcs}")
        translated_arcs = set().union(*(self.shell_file.arc_translations.get(arc, {arc}) for arc in arcs))
        logger.debug(f"[{os.path.basename(self.filename)}] translated_arcs: {translated_arcs}")
        unrecognized_arcs = translated_arcs - set(self.shell_file.arcs) # - self_loops
        negatives = set(arc for arc in unrecognized_arcs if -1 in arc)
        unrecognized_arcs = unrecognized_arcs - negatives
        logger.warning(f"[{os.path.basename(self.filename)}] unrecognized_arcs: {unrecognized_arcs}")
        final_arcs = translated_arcs - unrecognized_arcs
        logger.debug(f"[{os.path.basename(self.filename)} final_arcs: {final_arcs}")
        return final_arcs

    def exit_counts(self) -> dict[TLineNo, int]:
        return self.shell_file.exit_counts

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
            if "+:::COV:::" not in line:
                continue

            try:
                replicating_characters, _, caller_filename, caller_lineno, path_, funcname, lineno_, _ = line.split(":::", maxsplit=7)
                indirections = len(replicating_characters) - 1
                lineno = int(lineno_)
                path = Path(path_).absolute()
            except ValueError as e:
                raise ValueError(f"could not parse line {line}") from e

            self.line_data[str(path)].add(lineno)
            previous_lines = [lineno for filename, funcname_, lineno in self._previous_lines if (filename, funcname_) == (str(path), funcname)]
            previous_line = previous_lines[-1] if previous_lines else -1
            self.arc_data[str(path)].append((previous_line, lineno))
            self._previous_lines.append((str(path), funcname, lineno))
            logger.info(f"filename: {path.name}, line: {line}, indirections: {indirections}, arc: {self.arc_data[str(path)][-1]}")

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
            logger.info(f"arc_data: {arc_data}")
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
PS4="+:::COV:::\${{BASH_SOURCE[1]:-<global>}}:::\${{BASH_LINENO[0]:--1}}:::\${{BASH_SOURCE}}:::\${{FUNCNAME:-<global>}}:::\${{COVERAGE_SH_OVERRIDE_LINENO:-\${{LINENO}}}}:::"
exec {{BASH_XTRACEFD}}>>"{fifo_path!s}"
export BASH_XTRACEFD
set -x
# trap 'COVERAGE_SH_OVERRIDE_LINENO=-1 >&{{BASH_XTRACEFD}}' EXIT
# trap 'COVERAGE_SH_OVERRIDE_LINENO=-1 >&{{BASH_XTRACEFD}}' ERR
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


class ShellPlugin(coverage.CoveragePlugin):
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

    def file_tracer(self, filename: str) -> coverage.FileTracer | None:  # noqa: ARG002
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
