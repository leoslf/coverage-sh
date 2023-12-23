from pathlib import Path

import pytest as pytest


@pytest.fixture()
def resources_dir():
    return Path(__file__).parent / "resources"
