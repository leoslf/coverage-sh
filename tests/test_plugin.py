from pathlib import Path

import pytest

from coverage_sh import ShellPlugin


@pytest.fixture()
def examples_dir(resources_dir):
    return resources_dir / "examples"


def test_ShellPlugin_file_tracer():
    assert False


def test_ShellPlugin_file_reporter():
    assert False


def test_ShellPlugin_find_executable_files(examples_dir):
    plugin = ShellPlugin({})

    executable_files = plugin.find_executable_files(str(examples_dir))

    assert [Path(f) for f in sorted(executable_files)] == [
        examples_dir / "atom.sh",
        examples_dir / "shell-file.weird.suffix",
    ]
