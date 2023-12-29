import json
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def dummy_project_dir(resources_dir, tmp_path):
    source = resources_dir / "testproject"
    dest = tmp_path / "testproject"
    shutil.copytree(source, dest)

    # source.joinpath(".coverage").unlink(missing_ok=True)
    # source.joinpath("coverage.json").unlink(missing_ok=True)

    return dest


def test_run_and_report(dummy_project_dir):
    coverage_file_path = dummy_project_dir.joinpath(".coverage")
    assert not coverage_file_path.is_file()

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "run", str(dummy_project_dir / "main.py")],
        cwd=dummy_project_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stderr == ""
    assert proc.stdout == "hello from shell\n"

    assert len(list(dummy_project_dir.glob(".coverage*"))) == 2

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"], cwd=dummy_project_dir
    )
    assert proc.returncode == 0

    assert len(list(dummy_project_dir.glob(".coverage*"))) == 1

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "json"], cwd=dummy_project_dir
    )
    assert proc.returncode == 0

    coverage_json = json.loads(dummy_project_dir.joinpath("coverage.json").read_text())
    assert coverage_json["files"] == {'main.py': {'excluded_lines': [],
                                                   'executed_lines': [1, 4, 5, 8, 9],
                                                   'missing_lines': [],
                                                   'summary': {'covered_lines': 5,
                                                               'excluded_lines': 0,
                                                               'missing_lines': 0,
                                                               'num_statements': 5,
                                                               'percent_covered': 100.0,
                                                               'percent_covered_display': '100'}},
                                       'test.sh': {'excluded_lines': [],
                                                   'executed_lines': [3],
                                                   'missing_lines': [],
                                                   'summary': {'covered_lines': 1,
                                                               'excluded_lines': 0,
                                                               'missing_lines': 0,
                                                               'num_statements': 1,
                                                               'percent_covered': 100.0,
                                                               'percent_covered_display': '100'}}}
    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "html"], cwd=dummy_project_dir
    )
    assert proc.returncode == 0
