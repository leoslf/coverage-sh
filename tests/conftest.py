import shutil
from pathlib import Path

import pytest as pytest


@pytest.fixture()
def resources_dir():
    return Path(__file__).parent / "resources"

@pytest.fixture
def dummy_project_dir(resources_dir, tmp_path):
    source = resources_dir / "testproject"
    dest = tmp_path / "testproject"
    shutil.copytree(source, dest)

    return dest
