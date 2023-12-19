import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.resources.testproject.main import main


def test_debug(resources_dir):
    proc = subprocess.run([sys.executable, "-m", "coverage", "run", "main.py"], cwd=testproject_dir)



@pytest.fixture
def testproject_dir(resources_dir, tmp_path):
    tp =  resources_dir / "testproject"
    tp.joinpath(".coverage").unlink(missing_ok=True)
    tp.joinpath("coverage.json").unlink(missing_ok=True)

    return tp

def test_run_and_report(testproject_dir):
    coverage_file_path = testproject_dir.joinpath(".coverage")

    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = str(testproject_dir / "pyproject.toml")


    assert not coverage_file_path.is_file()

    proc = subprocess.run([sys.executable,"/home/kilian/code/coverage-sh/tests/resources/testproject/main.py"])

    assert  proc.returncode == 0
    assert  coverage_file_path.is_file()

    proc = subprocess.run(["coverage", "json"], cwd=testproject_dir)

    coverage_json = json.loads(testproject_dir.joinpath("coverage.json").read_text())
    # assert coverage_json == {}


def test_hacky():
    import coverage

    cov = coverage.Coverage()
    cov.start()

    main()

    cov.stop()



