#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove

from __future__ import annotations

import atexit
import contextlib
import inspect
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from socket import gethostname
from typing import TYPE_CHECKING, Iterable

import coverage
import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from coverage.sqldata import filename_suffix
from tree_sitter_languages import get_parser

if TYPE_CHECKING:
    from coverage.types import TLineNo
    from tree_sitter import Node

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


OriginalPopen = subprocess.Popen


class PatchedPopen(OriginalPopen):
    tracefiles_dir_path: Path = Path.cwd()

    def __init__(self, *args, **kwargs):
        self._envfile_path = None

        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        self._setup_envfile()

        env = kwargs.get("env", os.environ.copy())
        env["BASH_ENV"] = str(self._envfile_path)
        env["ENV"] = str(self._envfile_path)
        kwargs["env"] = env

        super().__init__(**kwargs)

    def _setup_envfile(self) -> None:
        self.tracefiles_dir_path.mkdir(parents=True, exist_ok=True)
        self._envfile_path = (
            self.tracefiles_dir_path / f"env-helper.{filename_suffix(suffix=True)}.sh"
        )
        tracefile_path = (
            self.tracefiles_dir_path / f"shelltrace.{filename_suffix(suffix=True)}"
        )
        self._envfile_path.write_text(
            rf"""\
#!/bin/sh
PS4="COV:::\${{BASH_SOURCE}}:::\${{LINENO}}:::"
exec {{BASH_XTRACEFD}}>>"{tracefile_path!s}"
set -x
"""
        )

    def __del__(self):
        if self._envfile_path is not None and self._envfile_path.is_file():
            self._envfile_path.unlink()
        super().__del__()


class ShellPlugin(CoveragePlugin):
    def __init__(self, options: dict[str, str]):
        self.options = options
        self._data = None
        self.tracefiles_dir_path = Path.cwd() / ".coverage-sh"
        self.tracefiles_dir_path.mkdir(parents=True, exist_ok=True)

        atexit.register(self._convert_traces)
        # TODO: Does this work with multithreading? Isnt there an easier way of finding the base path?
        PatchedPopen.tracefiles_dir_path = self.tracefiles_dir_path
        subprocess.Popen = PatchedPopen

    def _init_data(self) -> None:
        if self._data is None:
            self._data = coverage.CoverageData(
                # TODO: make basename configurable
                basename=self.tracefiles_dir_path.parent / ".coverage",
                suffix="sh." + filename_suffix(suffix=True),
                # TODO: set warn, debug and no_disk
            )

    def _convert_traces(self) -> None:
        self._init_data()

        # TODO: This is fragile, move the name generation into a function
        for tracefile_path in self.tracefiles_dir_path.glob(
            f"shelltrace.{gethostname()}.{os.getpid()}.*"
        ):
            line_data = self._parse_tracefile(tracefile_path)
            self._write_trace(line_data)

            tracefile_path.unlink()

        with contextlib.suppress(FileNotFoundError, OSError):
            self.tracefiles_dir_path.rmdir()

    @staticmethod
    def _parse_tracefile(tracefile_path: Path) -> dict[str, set[int]]:
        if not tracefile_path.exists():
            return {}

        line_data = defaultdict(set)
        with tracefile_path.open("r") as fd:
            for line in fd:
                if "COV:::" not in line:
                    continue

                try:
                    _, path, lineno, _ = line.split(":::")
                    lineno = int(lineno)
                    path = Path(path).absolute()
                except ValueError as e:
                    raise ValueError(
                        f"could not parse line {line} in {tracefile_path}"
                    ) from e

                line_data[str(path)].add(lineno)

        return line_data

    def _write_trace(self, line_data: dict[str, set[int]]) -> None:
        self._data.add_file_tracers({f: "coverage_sh.ShellPlugin" for f in line_data})
        self._data.add_lines(line_data)
        self._data.write()

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
