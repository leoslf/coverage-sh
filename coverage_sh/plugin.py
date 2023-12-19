from pathlib import Path
from typing import Iterable, Optional, Set, Union

import magic
from coverage import CoveragePlugin, FileReporter, FileTracer
from coverage.types import TLineNo

SUPPORTED_MIME_TYPES = ("text/x-shellscript",)




class ShellFileReporter(FileReporter):
    def __init__(self, filename: str):
        super().__init__(filename)

        self.path = Path(filename)

        self._executable_lines = set()

    def source(self) -> str:
        return self.path.read_text()

    def lines(self) -> Set[TLineNo]:
        return set(range(len(self.path.read_text().splitlines())))


class ShellPlugin(CoveragePlugin):
    def __init__(self, options: dict[str, str]):
        self.options = options

    def file_tracer(self, filename: str) -> Optional[FileTracer]:
        return None

    def file_reporter(
        self,
        filename: str,
    ) -> Union[ShellFileReporter, str]:
        sfr = ShellFileReporter(filename)
        print(sfr)
        return sfr

    def find_executable_files(
        self,
        src_dir: str,
    ) -> Iterable[str]:
        for f in Path(src_dir).rglob("*"):
            if not f.is_file() or any(p.startswith(".") for p in f.parts):
                continue



            if magic.from_file(f, mime=True) in SUPPORTED_MIME_TYPES:
                yield str(f)
