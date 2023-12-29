from __future__ import annotations

import atexit
import inspect
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Any, Iterable

import coverage
import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from coverage.sqldata import filename_suffix
from tree_sitter_languages import get_parser

if TYPE_CHECKING:
    from coverage.types import TLineNo
    from tree_sitter import Node

SH_ALIASES = {"sh", "/bin/sh", "/usr/bin/sh", which("sh")}
BASH_ALIASES = {"bash", "/bin/bash", "/usr/bin/bash", which("bash")}
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
    def __init__(self, *args, **kwargs):
        self._tracefile_path = Path.cwd().joinpath(
            ".coverage-sh", f"{os.getpid()}.trace"
        )
        self._tracefile_path.parent.mkdir(parents=True, exist_ok=True)
        self._tracefile_fd = None
        self._data = None

        # convert args into kwargs
        sig = inspect.signature(subprocess.Popen)
        kwargs.update(dict(zip(sig.parameters.keys(), args)))

        executable = kwargs.get("executable")
        args: list[str] = kwargs.get("args")

        patch_executable = None
        if (args[0] in SH_ALIASES) or executable in SH_ALIASES:
            patch_executable = which("sh")
        elif (args[0] in BASH_ALIASES) or executable in BASH_ALIASES:
            patch_executable = which("bash")

        cov = coverage.Coverage.current()
        if cov is not None and patch_executable is not None:
            self._init_trace(kwargs)
            return

        super().__init__(**kwargs)

    def _init_trace(self, kwargs: dict[str, Any]) -> None:
        self._init_data()

        self._tracefile_path.touch()
        self._tracefile_fd = os.open(self._tracefile_path, flags=os.O_RDWR | os.O_CREAT)

        env = kwargs.get("env", os.environ.copy())
        env["BASH_XTRACEFD"] = str(self._tracefile_fd)
        env["PS4"] = "COV:::$BASH_SOURCE:::$LINENO:::"
        kwargs["env"] = env

        args = list(kwargs.get("args", ()))
        args.insert(1, "-x")
        kwargs["args"] = args

        pass_fds = list(kwargs.get("pass_fds", ()))
        pass_fds.append(self._tracefile_fd)
        kwargs["pass_fds"] = pass_fds

        atexit.register(self._finish_trace)

        super().__init__(**kwargs)

    @staticmethod
    def _filename_suffix() -> str:
        return "sh." + filename_suffix(suffix=True)

    def _init_data(self) -> None:
        if self._data is None:
            config = coverage.Coverage.current().config
            Path(config.data_file).parent.mkdir(parents=True, exist_ok=True)
            self._data = coverage.CoverageData(
                basename=config.data_file,
                suffix=self._filename_suffix(),
                # TODO set these via the plugin config
                warn=coverage.Coverage.current()._warn,  # noqa: SLF001
                debug=coverage.Coverage.current()._debug,  # noqa: SLF001
                no_disk=coverage.Coverage.current()._no_disk,  # noqa: SLF001
            )

    def _parse_tracefile(self) -> dict[str, set[int]]:
        if not self._tracefile_path.exists():
            return {}

        line_data = defaultdict(set)
        with self._tracefile_path.open("r") as fd:
            for line in fd:
                _, path, lineno, _ = line.split(":::")
                path = Path(path).absolute()
                line_data[str(path)].add(int(lineno))

        return line_data

    def _write_trace(self, line_data: dict[str, set[int]]) -> None:
        self._data.add_file_tracers({f: "coverage_sh.ShellPlugin" for f in line_data})
        self._data.add_lines(line_data)
        self._data.write()

    def _finish_trace(self) -> None:
        line_data = self._parse_tracefile()
        self._write_trace(line_data)

        if self._tracefile_fd is not None:
            os.close(self._tracefile_fd)

        self._tracefile_path.unlink(missing_ok=True)
        parent_dir = self._tracefile_path.parent
        if len(list(parent_dir.rglob("*"))) == 0:
            parent_dir.rmdir()


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
            if not f.is_file() or any(p.startswith(".") for p in f.parts):
                continue

            if self._is_relevant(f):
                yield str(f)
