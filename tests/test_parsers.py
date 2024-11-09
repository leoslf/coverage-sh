# mypy: disable-error-code="no-untyped-def,no-untyped-call"
import pytest

import os
from pathlib import Path

from coverage_sh.parsers import ShellFileParser

class TestShellFileParser:
    @classmethod
    def create_parser(cls, path: Path) -> ShellFileParser:
        return ShellFileParser(
            filename=path,
            source=path.read_text(),
        )

    @pytest.mark.parametrize("filename", [
        "multiple_elifs_example.sh",
    ])
    def test_if_statement(self, dummy_project_dir: Path, filename: str):
        parser = self.create_parser(dummy_project_dir / filename)

