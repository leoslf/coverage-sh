from pathlib import Path
from typing import Iterable, Optional, Set, Union

import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from coverage.types import TLineNo

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


class ShellPlugin(CoveragePlugin):
    def __init__(self, options: dict[str, str]):
        self.options = options

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

            if magic.from_file(f, mime=True) in SUPPORTED_MIME_TYPES:
                yield str(f)
