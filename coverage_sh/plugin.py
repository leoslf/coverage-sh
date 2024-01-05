#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove

from __future__ import annotations

import contextlib
import inspect
import os
import string
import subprocess
from collections import defaultdict
from pathlib import Path
from random import Random
from socket import gethostname
from threading import Thread
from typing import TYPE_CHECKING, Iterable, Iterator

import coverage
import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from tree_sitter_languages import get_parser

if TYPE_CHECKING:
    from coverage.types import TLineNo
    from tree_sitter import Node

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


class CoverageParser(Thread):
    def __init__(self, fifo_path: Path) -> None:
        super().__init__()
        self._keep_running = True
        self.last_line_fragment = ""

        self._fifo_path = fifo_path
        os.mkfifo(self._fifo_path)

    def stop(self) -> None:
        self._keep_running = False

    def buf_to_lines(self, buf: bytes) -> Iterator[str]:
        raw = self.last_line_fragment + buf.decode()
        self.last_line_fragment = ""

        for line in raw.splitlines(keepends=True):
            if line.endswith("\n"):
                yield line[:-1]
            else:
                self.last_line_fragment = line

    def run(self) -> None:
        while self._keep_running:
            fifo = os.open(self._fifo_path, flags=os.O_RDONLY | os.O_NONBLOCK)
            while True:
                try:
                    buf = os.read(fifo, 2**10)
                except BlockingIOError:
                    if not self._keep_running:
                        break
                    continue
                if not buf:
                    if not self._keep_running:
                        break
                    continue

                lines = list(self.buf_to_lines(buf))
                if lines:
                    self.report_lines(lines)

            lines = [l for l in self.buf_to_lines(b"\n") if l != ""]  # noqa: E741
            if lines:
                self.report_lines(lines)

        with contextlib.suppress(FileNotFoundError):
            self._fifo_path.unlink()

    def report_lines(self, lines: list[str]) -> None:
        cov = coverage.Coverage.current()
        if cov is None:
            raise RuntimeError(f"no coverage object, discarding lines {lines}")

        line_data = defaultdict(set)

        for line in lines:
            if "COV:::" not in line:
                continue

            try:
                _, path_, lineno_, _ = line.split(":::")
                lineno = int(lineno_)
                path = Path(path_).absolute()
            except ValueError as e:
                raise ValueError(f"could not parse line {line}") from e

            line_data[str(path)].add(lineno)

        cov.get_data().add_lines(line_data)


OriginalPopen = subprocess.Popen


class PatchedPopen(OriginalPopen):
    tracefiles_dir_path: Path = Path.cwd()

    def __init__(self, *args, **kwargs):
        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        self._fifo_path = (
            Path(os.environ["XDG_RUNTIME_DIR"])
            / f"coverage-sh.{filename_suffix()}.pipe"
        )
        with contextlib.suppress(FileNotFoundError):
            self._fifo_path.unlink()

        self._parser_thread = CoverageParser(self._fifo_path)
        self._parser_thread.start()

        self._helper_path = (
            Path(os.environ["XDG_RUNTIME_DIR"]) / f"coverage-sh.{filename_suffix()}.sh"
        )
        self._helper_path.write_text(
            rf"""#!/bin/sh
PS4="COV:::\${{BASH_SOURCE}}:::\${{LINENO}}:::"
exec {{BASH_XTRACEFD}}>>"{self._fifo_path!s}"
set -x
"""
        )

        env = kwargs.get("env", os.environ.copy())
        env["BASH_ENV"] = str(self._helper_path)
        env["ENV"] = str(self._helper_path)
        kwargs["env"] = env

        super().__init__(**kwargs)

    def wait(self, timeout: float | None = None) -> int:
        retval = super().wait(timeout)
        self._parser_thread.stop()
        self._parser_thread.join()
        with contextlib.suppress(FileNotFoundError):
            self._helper_path.unlink()
        return retval


class ShellPlugin(CoveragePlugin):
    def __init__(self, options: dict[str, str]):
        self.options = options

        subprocess.Popen = PatchedPopen

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
