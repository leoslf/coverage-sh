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


    def __init__(self, *args, **kwargs):

        self._tracefile_path = Path.cwd().joinpath(".coverage-sh", f"{os.getpid()}.trace")
        self._tracefile_path.parent.mkdir(parents=True, exist_ok=True)

        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        executable = kwargs.get("executable")
        args : list[str] = kwargs.get("args")

        patch_executable = None
        if (args[0] in SH_ALIASES) or executable in (
            SH_ALIASES
        ):
            patch_executable = which("sh")
        elif (args[0] in BASH_ALIASES) or executable in (
            BASH_ALIASES
        ):
            patch_executable = which("bash")

        if patch_executable is not None:
            self._tracefile_path.touch()
            self._tracefile_fd = os.open(self._tracefile_path, flags=os.O_RDWR | os.O_CREAT)

            env = kwargs.get("env", os.environ.copy())
            env["BASH_XTRACEFD"] = str(self._tracefile_fd)
            env["PS4"] = 'COV:::$BASH_SOURCE:::$LINENO:::'
            kwargs["env"] = env

            args.insert(1, "-x")
            kwargs["args"] = args

            pass_fds = list(kwargs.get("pass_fds", ()))
            pass_fds.append(self._tracefile_fd)
            kwargs["pass_fds"]= pass_fds

            atexit.register(self.__del__)
            super().__init__(**kwargs)

            return

        super().__init__( **kwargs)



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
        os.close(self._tracefile_fd)
        self._tracefile_path.unlink()
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
