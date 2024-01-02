#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove

import json
import subprocess
import sys


def test_run_and_report(dummy_project_dir, monkeypatch):
    monkeypatch.chdir(dummy_project_dir)

    coverage_file_path = dummy_project_dir.joinpath(".coverage")
    assert not coverage_file_path.is_file()

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "main.py"],
        cwd=dummy_project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.stderr == ""
    assert proc.stdout == (
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
    assert proc.returncode == 0

    assert len(list(dummy_project_dir.glob(".coverage*"))) == 2

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=dummy_project_dir,
        check=False,
    )
    assert proc.returncode == 0

    assert len(list(dummy_project_dir.glob(".coverage*"))) == 1

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "html"], cwd=dummy_project_dir, check=False
    )
    assert proc.returncode == 0

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "json"], cwd=dummy_project_dir, check=False
    )
    assert proc.returncode == 0

    coverage_json = json.loads(dummy_project_dir.joinpath("coverage.json").read_text())
    assert coverage_json["files"]["test.sh"]["excluded_lines"] == []
    assert coverage_json["files"]["test.sh"]["executed_lines"] == [
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
    assert coverage_json["files"]["test.sh"]["missing_lines"] == [21, 54, 60, 63]
