#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove

from __future__ import annotations

import contextlib
import inspect
import os
import selectors
import stat
import string
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from random import Random
from socket import gethostname
from time import sleep
from typing import TYPE_CHECKING, Iterable, Iterator

import coverage
import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from tree_sitter_languages import get_parser

if TYPE_CHECKING:
    from coverage.types import TLineNo
    from tree_sitter import Node

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

parser = get_parser("bash")


class ShellFileReporter(FileReporter):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)

        self.path = Path(filename)
        self._content = None
        self._executable_lines = set()

    def source(self) -> str:
        if self._content is None:
            self._content = self.path.read_text()

        return self._content

    def _parse_ast(self, node: Node) -> None:
        if node.is_named and node.type in EXECUTABLE_NODE_TYPES:
            self._executable_lines.add(node.start_point[0] + 1)

        for child in node.children:
            self._parse_ast(child)

    def lines(self) -> set[TLineNo]:
        tree = parser.parse(self.source().encode("utf-8"))
        self._parse_ast(tree.root_node)

        return self._executable_lines


def filename_suffix(*, add_random: bool = True) -> str:
    die = Random(os.urandom(8))
    letters = string.ascii_uppercase + string.ascii_lowercase
    rolls = "".join(die.choice(letters) for _ in range(6))
    if add_random:
        return f"{gethostname()}.{os.getpid()}.X{rolls}x"
    return f"{gethostname()}.{os.getpid()}"


class CoverageParserThread(threading.Thread):
    def __init__(self, *args, coverage_data_path: Path | None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._keep_running = True
        self._last_line_fragment = ""
        self._coverage_data_path = coverage_data_path
        self._line_data = defaultdict(set)
        self._listening = False

        self.fifo_path = TMP_PATH / f"coverage-sh.{filename_suffix()}.pipe"
        with contextlib.suppress(FileNotFoundError):
            self.fifo_path.unlink()
        os.mkfifo(self.fifo_path, mode=stat.S_IRUSR | stat.S_IWUSR)

    def start(self) -> None:
        super().start()
        while not self._listening:
            sleep(0.0001)

    def stop(self) -> None:
        self._keep_running = False

    def _buf_to_lines(self, buf: bytes) -> Iterator[str]:
        raw = self._last_line_fragment + buf.decode()
        self._last_line_fragment = ""

        for line in raw.splitlines(keepends=True):
            if line.endswith("\n"):
                yield line[:-1]
            else:
                self._last_line_fragment = line

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
            while not eof and (data_incoming or self._keep_running):
                events = sel.select(timeout=1)
                data_incoming = len(events) > 0
                for key, _ in events:
                    buf = os.read(key.fd, 2**10)
                    if not buf:
                        eof = True
                        break
                    self._report_lines(list(self._buf_to_lines(buf)))

            # flush our internal buffer
            self._report_lines([l for l in self._buf_to_lines(b"\n") if l != ""])  # noqa: E741

            sel.unregister(fifo)
            os.close(fifo)

        self._write_trace()
        with contextlib.suppress(FileNotFoundError):
            self.fifo_path.unlink()

    def _report_lines(self, lines: list[str]) -> None:
        if not lines:
            return

        for line in lines:
            if "COV:::" not in line:
                continue

            try:
                _, path_, lineno_, _ = line.split(":::")
                lineno = int(lineno_)
                path = Path(path_).absolute()
            except ValueError as e:
                raise ValueError(f"could not parse line {line}") from e

            self._line_data[str(path)].add(lineno)

    def _write_trace(self) -> None:
        suffix_ = "sh." + filename_suffix()
        coverage_data = coverage.CoverageData(
            # TODO: This probably wont work with pytest-cov
            basename=self._coverage_data_path,
            suffix=suffix_,
            # TODO: set warn, debug and no_disk
        )

        coverage_data.add_file_tracers(
            {f: "coverage_sh.ShellPlugin" for f in self._line_data}
        )
        coverage_data.add_lines(self._line_data)
        coverage_data.write()


OriginalPopen = subprocess.Popen


def init_helper(fifo_path: Path) -> Path:
    helper_path = Path(TMP_PATH, f"coverage-sh.{filename_suffix()}.sh")
    helper_path.write_text(
        rf"""#!/bin/sh
PS4="COV:::\${{BASH_SOURCE}}:::\${{LINENO}}:::"
exec {{BASH_XTRACEFD}}>>"{fifo_path!s}"
export BASH_XTRACEFD
set -x
"""
    )
    helper_path.chmod(mode=stat.S_IRUSR | stat.S_IWUSR)
    return helper_path


class PatchedPopen(OriginalPopen):
    data_file_path: Path = Path.cwd()

    def __init__(self, *args, **kwargs):
        if coverage.Coverage.current() is None:
            # we are not recording coverage, so just act like the original Popen
            self._parser_thread = None
            super().__init__(*args, **kwargs)
            return

        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        self._parser_thread = CoverageParserThread(
            coverage_data_path=self.data_file_path, name="CoverageParserThread(None)"
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
            return retval

        self._parser_thread.stop()
        self._parser_thread.join()
        with contextlib.suppress(FileNotFoundError):
            self._helper_path.unlink()
        return retval


class MonitorThread(threading.Thread):
    def __init__(self, *args, parser_thread: CoverageParserThread, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parser_thread = parser_thread

    def run(self) -> None:
        main_thread = threading.main_thread()
        main_thread.join()
        self.parser_thread.stop()
        self.parser_thread.join()


class ShellPlugin(CoveragePlugin):
    def __init__(self, options: dict[str, str]):
        self.options = options
        self._helper_path = None

        coverage_data_path = Path(coverage.Coverage().config.data_file).absolute()
        if self.options.get("cover_always", False):
            parser_thread = CoverageParserThread(
                coverage_data_path=coverage_data_path,
                name=f"CoverageParserThread({coverage_data_path!s})",
            )
            parser_thread.start()

            monitor_thread = MonitorThread(
                parser_thread=parser_thread, name="MonitorThread"
            )
            monitor_thread.start()

            self._helper_path = init_helper(parser_thread.fifo_path)
            os.environ["BASH_ENV"] = str(self._helper_path)
            os.environ["ENV"] = str(self._helper_path)
        else:
            PatchedPopen.data_file_path: Path = coverage_data_path
            subprocess.Popen = PatchedPopen

    def __del__(self):
        if self._helper_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._helper_path.unlink()

    @staticmethod
    def _is_relevant(path: Path) -> bool:
        return magic.from_file(path, mime=True) in SUPPORTED_MIME_TYPES

    def file_tracer(self, filename: str) -> FileTracer | None:  # noqa: ARG002
        return None

    def file_reporter(
        self,
        filename: str,
    ) -> ShellFileReporter | str:
        return ShellFileReporter(filename)

    def find_executable_files(
        self,
        src_dir: str,
    ) -> Iterable[str]:
        for f in Path(src_dir).rglob("*"):
            # TODO: Use coverage's logic for figuring out if a file should be excluded
            if not f.is_file() or any(p.startswith(".") for p in f.parts):
                continue

            if self._is_relevant(f):
                yield str(f)
