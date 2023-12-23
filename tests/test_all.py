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
    assert "hello from shell" in proc.stdout
    assert coverage_file_path.is_file()

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "json"], cwd=dummy_project_dir
    )

    coverage_json = json.loads(dummy_project_dir.joinpath("coverage.json").read_text())
    assert coverage_json == {}

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "html"], cwd=dummy_project_dir
    )
