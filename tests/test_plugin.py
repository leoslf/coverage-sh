import atexit
import subprocess
from pathlib import Path

import coverage
import pytest

import coverage_sh
from coverage_sh import ShellPlugin
from coverage_sh.plugin import PatchedPopen


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


def test_patched_popen(resources_dir, monkeypatch, tmp_path):

    monkeypatch.chdir(tmp_path)

    atexit_callables = []
    def atexit_register(callable):
        atexit_callables.append(callable)

    cov = coverage.Coverage()
    cov.start()

    monkeypatch.setattr(coverage_sh.plugin.atexit, "register", atexit_register)



    test_sh_path = resources_dir / "testproject" / "test.sh"
    proc = PatchedPopen(["/bin/bash", test_sh_path], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, encoding="utf8")

    proc.wait()

    cov.stop()

    assert proc.stderr.read() == ""
    assert proc.stdout.read() == "hello from shell\n"

    assert len(atexit_callables) == 1

    with pytest.raises(ValueError) as e:
        for c in atexit_callables:
            c()

    assert e.value.args[0] == 'no Coverage object'

    # line_data = proc._parse_tracefile()
    # assert {str(test_sh_path): {3}} == dict(line_data)





