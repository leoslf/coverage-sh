#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove
import json
import subprocess
import sys
from pathlib import Path

import coverage
import pytest

from coverage_sh import ShellPlugin
from coverage_sh.plugin import PatchedPopen

SYNTAX_EXAMPLE_STDOUT = (
    "Hello, World!\n"
    "Variable is set to 'Hello, World!'\n"
    "Iteration 1\n"
    "Iteration 2\n"
    "Iteration 3\n"
    "Iteration 4\n"
    "Iteration 5\n"
    "Hello from a function!\n"
    "Current OS is: Linux\n"
    "5 + 3 = 8\n"
    "This is a sample file.\n"
    "You selected a banana.\n"
)


@pytest.fixture()
def examples_dir(resources_dir):
    return resources_dir / "examples"


def test_ShellPlugin_file_tracer():
    pytest.skip("Not yet implemented")


def test_ShellPlugin_file_reporter():
    pytest.skip("Not yet implemented")


def test_ShellPlugin_find_executable_files(examples_dir):
    plugin = ShellPlugin({})

    executable_files = plugin.find_executable_files(str(examples_dir))

    assert [Path(f) for f in sorted(executable_files)] == [
        examples_dir / "shell-file.weird.suffix",
    ]


def test_PatchedPopen(
    resources_dir,
    dummy_project_dir,
    monkeypatch,
):
    monkeypatch.chdir(dummy_project_dir)

    cov = coverage.Coverage()
    cov.start()

    test_sh_path = resources_dir / "testproject" / "test.sh"
    proc = PatchedPopen(
        ["/bin/bash", test_sh_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8",
    )
    proc.wait()

    cov.stop()

    assert proc.stderr.read() == ""
    assert proc.stdout.read() == SYNTAX_EXAMPLE_STDOUT


@pytest.mark.parametrize("cover_always", [(True), (False)])
def test_end2end(dummy_project_dir, monkeypatch, cover_always: bool):
    monkeypatch.chdir(dummy_project_dir)

    coverage_file_path = dummy_project_dir.joinpath(".coverage")
    assert not coverage_file_path.is_file()

    if cover_always:
        pyproject_file = dummy_project_dir.joinpath("pyproject.toml")
        with pyproject_file.open("a") as fd:
            fd.write("\n[tool.coverage.coverage_sh]\ncover_always = true")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "coverage", "run", "main.py"],
            cwd=dummy_project_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except subprocess.TimeoutExpired as e:
        assert e.stdout == "failed stdout"  # noqa: PT017
        assert e.stderr == "failed stderr"  # noqa: PT017
        assert False
    assert proc.stderr == ""
    assert proc.stdout == SYNTAX_EXAMPLE_STDOUT
    assert proc.returncode == 0

    assert len(list(dummy_project_dir.glob(".coverage*"))) == 2
    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=dummy_project_dir,
        check=False,
    )
    print("recombined")
    assert proc.returncode == 0

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "html"], cwd=dummy_project_dir, check=False
    )
    assert proc.returncode == 0

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "json"], cwd=dummy_project_dir, check=False
    )
    assert proc.returncode == 0

    coverage_json = json.loads(dummy_project_dir.joinpath("coverage.json").read_text())
    assert coverage_json["files"]["test.sh"]["executed_lines"] == [8, 9]
    assert coverage_json["files"]["syntax_example.sh"]["excluded_lines"] == []
    assert coverage_json["files"]["syntax_example.sh"]["executed_lines"] == [
        12,
        15,
        18,
        19,
        25,
        26,
        31,
        34,
        37,
        38,
        41,
        42,
        45,
        46,
        47,
        48,
        51,
        52,
        57,
    ]
    assert coverage_json["files"]["syntax_example.sh"]["missing_lines"] == [
        21,
        54,
        60,
        63,
    ]
