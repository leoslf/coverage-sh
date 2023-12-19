from pathlib import Path
from typing import Optional

import coverage


class ShFileTracer(coverage.FileTracer):

    def __int__(self, path: Path):
        self._path = path


class ShPlugin(coverage.CoveragePlugin):

    def file_tracer(self, filename: str) -> Optional[ShFileTracer]:
        return ShFileTracer(Path(filename))
