from __future__ import annotations

import atexit
import inspect
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from shutil import which
from stat import S_IXUSR, S_IRUSR, S_IWUSR

import coverage

SH_ALIASES = {"sh", "/bin/sh", "/usr/bin/sh", which("sh")}

BASH_ALIASES = {"bash", "/bin/bash", "/usr/bin/bash", which("bash")}
from typing import Iterable, Optional, Set, Union

import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from coverage.types import TLineNo
import subprocess

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
    "compound_statement",
}

SUPPORTED_MIME_TYPES = ("text/x-shellscript",)

from tree_sitter_languages import get_parser

parser = get_parser("bash")


class ShellFileReporter(FileReporter):
    def __init__(self, filename: str):
        super().__init__(filename)

        self.path = Path(filename)
        self._content = None
        self._executable_lines = set()

    def source(self) -> str:
        if self._content is None:
            self._content = self.path.read_text()

        return self._content

    def _parse_ast(self, node):
        if node.is_named and node.type in EXECUTABLE_NODE_TYPES:
            self._executable_lines.add(node.start_point[0] + 1)

        for child in node.children:
            self._parse_ast(child)

    def lines(self) -> Set[TLineNo]:
        tree = parser.parse(self.source().encode("utf-8"))
        self._parse_ast(tree.root_node)

        return self._executable_lines


def get_tracefile_path():
    return Path.cwd() / ".bash_tracefile"


OriginalPopen = subprocess.Popen


class PatchedPopen(OriginalPopen):
    def get_wrapper(self, executable, tracefile_path):
        return f"""\
#!/bin/sh
echo "inside patch wrapper"
export BASH_XTRACEFD
export PS4='COV:::$BASH_SOURCE:::$LINENO:::'
exec {executable} -x "$@" {{BASH_XTRACEFD}}>>{tracefile_path}
"""

    def __init__(self, *args, **kwargs):

        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        executable = kwargs.get("executable")
        args : list[str] = kwargs.get("args")

        patch_executable = None
        if (args[0] in (SH_ALIASES)) or executable in (
            SH_ALIASES
        ):
            patch_executable = which("sh")
        elif ( args[0] in (BASH_ALIASES)) or executable in (
            BASH_ALIASES
        ):
            patch_executable = which("bash")

        if patch_executable is not None:
            wrapper = tempfile.NamedTemporaryFile(
                "w", dir=Path.home() / ".cache", delete=False, suffix=".sh"
            )
            try:
                wrapper.write(
                    self.get_wrapper(patch_executable, str(get_tracefile_path()))
                )
                wrapper.close()
                os.chmod(wrapper.name, S_IRUSR | S_IWUSR | S_IXUSR)
                kwargs["executable"] = wrapper.name

                super().__init__(*args, **kwargs)

            finally:
                os.unlink(wrapper.name)

            return

        super().__init__(*args, **kwargs)

    @staticmethod
    def _parse_tracefile():
        tracefile = get_tracefile_path()
        if not tracefile.exists():
            return

        line_data = defaultdict(set)
        with tracefile.open("r") as fd:
            for line in fd:
                _, path, lineno, _ = line.split(":::")
                path = Path(path).absolute()
                line_data[str(path)].add(int(lineno))

        cov = coverage.Coverage.current()
        if cov is None:
            raise ValueError("no Coverage object")
        cov_data = cov.get_data()
        cov_data.add_lines(line_data)
        cov_data.write()

    def __del__(self):
        self._parse_tracefile()
        del self


class ShellPlugin(CoveragePlugin):
    def __init__(self, options: dict[str, str]):
        self.options = options

        self.tracefile_path = get_tracefile_path()
        self.tracefile_path.unlink(missing_ok=True)

        subprocess.Popen = PatchedPopen

    @staticmethod
    def _is_relevant(path):
        return magic.from_file(path, mime=True) in SUPPORTED_MIME_TYPES

    def file_tracer(self, filename: str) -> Optional[FileTracer]:
        return None

    def file_reporter(
        self,
        filename: str,
    ) -> Union[ShellFileReporter, str]:
        return ShellFileReporter(filename)

    def find_executable_files(
        self,
        src_dir: str,
    ) -> Iterable[str]:
        for f in Path(src_dir).rglob("*"):
            if not f.is_file() or any(p.startswith(".") for p in f.parts):
                continue

            if self._is_relevant(f):
                yield str(f)
