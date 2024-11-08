#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023-2024 Kilian Lackhove
# mypy: disable-error-code="no-untyped-def,no-untyped-call"
from __future__ import annotations

import sys
import os
import io
import re
import json
import importlib
import subprocess
import platform
import threading
import logging

from typing import cast
from collections import defaultdict
from pathlib import Path
from socket import gethostname
from time import sleep

import coverage
import pytest

from unittest import mock

from coverage_sh import ShellPlugin
from coverage_sh.plugin import (
    CoverageParserThread,
    CoverageWriter,
    CovLineParser,
    ArcData,
    LineData,
    MonitorThread,
    PatchedPopen,
    ShellFileReporter,
    filename_suffix,
)

logger = logging.getLogger(__name__)

SYNTAX_EXAMPLE_EXECUTABLE_LINES = {
    12,
    15,
    18,
    19,
    21,
    25,
    26,
    30,
    31,
    35,
    36,
    37,
    38,
    43,
    46,
    49,
    52,
    55,
    56,
    59,
    60,
    63,
    64,
    67,
    70,
    73,
    74,
    75,
    76,
    79,
    80,
    82,
    85,
    88,
    95,
}

SYNTAX_EXAMPLE_STDOUT = (
    "Hello, World!\n"
    "Variable is set to 'Hello, World!'\n"
    "Iteration 0\n"
    "Iteration 1\n"
    "Iteration 2\n"
    "Iteration 3\n"
    "Iteration 4\n"
    "Iteration 5\n"
    "Iteration 6\n"
    "Iteration 7\n"
    "Iteration 8\n"
    "Iteration 9\n"
    "Iteration 1\n"
    "Iteration 2\n"
    "Iteration 3\n"
    "Iteration 4\n"
    "Iteration 5\n"
    "Iteration 0\n"
    "Iteration 1\n"
    "Iteration 2\n"
    "Iteration 3\n"
    "Iteration 4\n"
    "Iteration 5\n"
    "Iteration 6\n"
    "Iteration 7\n"
    "Iteration 8\n"
    "Iteration 9\n"
    "Hello from a function!\n"
    "Bye from a function!\n"
    "Current OS is: Darwin\n"
    "5 + 3 = 8\n"
    "result == 8 ? 1 : 0 = 1\n"
    "testing\n"
    "This is a sample file.\n"
    "You selected a banana.\n"
)
SYNTAX_EXAMPLE_EXECUTED_BRANCHES = [
    [84, 85],
]
SYNTAX_EXAMPLE_EXCLUDED_BRANCHES: list[tuple[int, int]] = []
SYNTAX_EXAMPLE_MISSING_BRANCHES = [
    [81, 82],
    [81, 84],
    [84, 87],
    [87, 88],
    [87, 90],
]
SYNTAX_EXAMPLE_COVERED_LINES = [
    12,
    15,
    18,
    19,
    25,
    26,
    30,
    31,
    35,
    36,
    37,
    38,
    43,
    46,
    49,
    52,
    55,
    56,
    59,
    60,
    63,
    64,
    67,
    70,
    73,
    74,
    75,
    76,
    79,
    80,
    85,
]
SYNTAX_EXAMPLE_EXCLUDED_LINES = [21]
SYNTAX_EXAMPLE_MISSING_LINES = [
    21,
    82,
    88,
    95,
]
COVERAGE_LINE_CHUNKS = (
    b"""\
CCOV:::/home/dummy_user/dummy_dir_a:::<global>:::1:::a normal line
COV:::/home/dummy_user/dummy_dir_b:::<global>:::10:::a line
with a line fragment

COV:::/home/dummy_user/dummy_dir_a:::<global>:::2:::a  line with ::: triple columns
COV:::/home/dummy_user/dummy_dir_a:::<global>:::3:::a  line """,
    b"that spans multiple chunks\n",
    b"C",
    b"O",
    b"V",
    b":",
    b":",
    b":",
    b"/",
    b"ho",
    b"m",
    b"e",
    b"/dummy_user/dummy_dir_a:::<global>:::4:::a chunked line",
)
COVERAGE_LINES = [
    "CCOV:::/home/dummy_user/dummy_dir_a:::<global>:::1:::a normal line",
    "COV:::/home/dummy_user/dummy_dir_b:::<global>:::10:::a line",
    "with a line fragment",
    "COV:::/home/dummy_user/dummy_dir_a:::<global>:::2:::a  line with ::: triple columns",
    "COV:::/home/dummy_user/dummy_dir_a:::<global>:::3:::a  line that spans multiple chunks",
    "COV:::/home/dummy_user/dummy_dir_a:::<global>:::4:::a chunked line",
]
COVERAGE_LINE_COVERAGE = {
    "/home/dummy_user/dummy_dir_a": {1, 2, 3, 4},
    "/home/dummy_user/dummy_dir_b": {10},
}
COVERAGE_ARC_COVERAGE = {
    "/home/dummy_user/dummy_dir_a": {(-1, 1), (1, 2), (2, 3), (3, 4)},
    "/home/dummy_user/dummy_dir_b": {(-1, 10)},
}

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec
    module = sys.modules[module_name] = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module

@pytest.fixture()
def examples_dir(resources_dir):
    return resources_dir / "examples"


@pytest.fixture()
def syntax_example_path(resources_dir, tmp_path):
    original_path = resources_dir / "syntax_example.sh"
    working_copy_path = tmp_path / "syntax_example.sh"
    working_copy_path.write_bytes(original_path.read_bytes())
    return working_copy_path

@pytest.mark.parametrize("cover_always", [(True), (False)])
@pytest.mark.parametrize("branch", [(True), (False)])
def test_end2end(dummy_project_dir, monkeypatch, branch: bool, cover_always: bool):
    monkeypatch.chdir(dummy_project_dir)

    coverage_file_path = dummy_project_dir.joinpath(".coverage")
    assert not coverage_file_path.is_file()

    pyproject_file = dummy_project_dir.joinpath("pyproject.toml")
    with pyproject_file.open("a") as fd:
        fd.write(
            f"""
            branch = {json.dumps(branch)}

            [tool.coverage.coverage_sh]
            cover_always = {json.dumps(cover_always)}
            """
        )

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "coverage", "run", "main.py"],
            cwd=dummy_project_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
            env={**os.environ, "PYTHONWARNINGS": "ignore::FutureWarning"},
        )
    except subprocess.TimeoutExpired as e:  # pragma: no cover
        assert e.stdout == b"failed stdout"  # noqa: PT017
        assert e.stderr == b"failed stderr"  # noqa: PT017
        assert False
    assert proc.stderr == ""
    assert proc.stdout == SYNTAX_EXAMPLE_STDOUT
    assert proc.returncode == 0

    assert dummy_project_dir.joinpath(".coverage").is_file()
    assert len(list(dummy_project_dir.glob(f".coverage.sh.{gethostname()}.*"))) == 1

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=dummy_project_dir,
        check=False,
        text=True,
    )
    logger.info("recombined")
    assert proc.returncode == 0

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "html"],
        cwd=dummy_project_dir,
        check=False,
        text=True,
    )
    assert proc.returncode == 0
    logger.info(f"html_report: {os.path.join(dummy_project_dir, 'htmlcov', 'index.html')}")

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "json"],
        cwd=dummy_project_dir,
        check=False,
        text=True,
    )
    assert proc.returncode == 0

    coverage_json = json.loads(dummy_project_dir.joinpath("coverage.json").read_text())
    logger.debug(coverage_json)

    assert coverage_json["files"]["test.sh"]["summary"]["percent_covered"] == 100.0
    assert coverage_json["files"]["test.sh"]["executed_lines"] == [8, 9]
    assert coverage_json["files"]["test.sh"]["excluded_lines"] == []
    assert coverage_json["files"]["test.sh"]["missing_lines"] == []
    if branch:
        assert coverage_json["files"]["test.sh"]["executed_branches"] == []
        assert coverage_json["files"]["test.sh"]["missing_branches"] == []

    assert coverage_json["files"]["syntax_example.sh"]["summary"]["percent_covered"] > 0
    assert coverage_json["files"]["syntax_example.sh"]["executed_lines"] == SYNTAX_EXAMPLE_COVERED_LINES
    assert coverage_json["files"]["syntax_example.sh"]["excluded_lines"] == SYNTAX_EXAMPLE_EXCLUDED_LINES
    assert coverage_json["files"]["syntax_example.sh"]["missing_lines"] == SYNTAX_EXAMPLE_MISSING_LINES
    if branch:
        assert coverage_json["files"]["syntax_example.sh"]["executed_branches"] == SYNTAX_EXAMPLE_EXECUTED_BRANCHES
        assert coverage_json["files"]["syntax_example.sh"]["missing_branches"] == SYNTAX_EXAMPLE_MISSING_BRANCHES

@pytest.mark.parametrize("cover_always", [(True), (False)])
@pytest.mark.parametrize("branch", [(True), (False)])
def test_programmatic_end2end(dummy_project_dir, monkeypatch, branch: bool, cover_always: bool):
    monkeypatch.chdir(dummy_project_dir)

    coverage_file_path = dummy_project_dir.joinpath(".coverage")
    assert not coverage_file_path.is_file()

    pyproject_file = dummy_project_dir.joinpath("pyproject.toml")
    with pyproject_file.open("a") as fd:
        fd.write(
            f"""
            branch = {json.dumps(branch)}

            [tool.coverage.coverage_sh]
            cover_always = {json.dumps(cover_always)}
            join_main_thread = {json.dumps(False)}
            """
        )

    main = import_from_path("main", os.path.join(dummy_project_dir, "main.py"))

    cov = coverage.Coverage(data_file=dummy_project_dir.joinpath("coverage-data.db"), branch=branch)
    # force init
    cov.sys_info()

    shell_plugin = cast(ShellPlugin, cov._plugins.get("coverage_sh.ShellPlugin"))
    assert shell_plugin.branch == branch

    with cov.collect():
        main.main()

    if parser_thread := next((cast(CoverageParserThread, thread) for thread in threading.enumerate() if thread.name.startswith("CoverageShCoverageParserThread")), None):
        parser_thread.stop()
        parser_thread.join()

    if monitor_thread := next((thread for thread in threading.enumerate() if thread.name.startswith("CoverageShCoverageMonitorThread")), None):
        monitor_thread.join()

    cov.combine()
    html_report = cov.html_report()
    logger.info(f"html_report: {os.path.join(dummy_project_dir, 'htmlcov', 'index.html')}")
    cov.json_report()

    coverage_json = json.loads(dummy_project_dir.joinpath("coverage.json").read_text())
    logger.debug(coverage_json)

    assert coverage_json["files"]["test.sh"]["summary"]["percent_covered"] == 100.0
    assert coverage_json["files"]["test.sh"]["executed_lines"] == [8, 9]
    assert coverage_json["files"]["test.sh"]["excluded_lines"] == []
    assert coverage_json["files"]["test.sh"]["missing_lines"] == []
    if branch:
        assert coverage_json["files"]["test.sh"]["executed_branches"] == []
        assert coverage_json["files"]["test.sh"]["missing_branches"] == []

    assert coverage_json["files"]["syntax_example.sh"]["summary"]["percent_covered"] > 0
    assert coverage_json["files"]["syntax_example.sh"]["executed_lines"] == SYNTAX_EXAMPLE_COVERED_LINES
    assert coverage_json["files"]["syntax_example.sh"]["excluded_lines"] == SYNTAX_EXAMPLE_EXCLUDED_LINES
    assert coverage_json["files"]["syntax_example.sh"]["missing_lines"] == SYNTAX_EXAMPLE_MISSING_LINES
    if branch:
        assert coverage_json["files"]["syntax_example.sh"]["executed_branches"] == SYNTAX_EXAMPLE_EXECUTED_BRANCHES
        assert coverage_json["files"]["syntax_example.sh"]["missing_branches"] == SYNTAX_EXAMPLE_MISSING_BRANCHES


class TestShellFileReporter:
    @pytest.fixture()
    def reporter(self, syntax_example_path):
        return ShellFileReporter(str(syntax_example_path))

    def test_source_should_be_cached(self, syntax_example_path, reporter):
        reference = Path(reporter.filename).read_text()

        assert reporter.source() == reference
        syntax_example_path.unlink()
        assert reporter.source() == reference

    def test_lines_should_match_reference(self, reporter):
        assert reporter.lines() == SYNTAX_EXAMPLE_EXECUTABLE_LINES

    def test_invalid_syntax_should_be_treated_as_executable(self, resources_dir):
        reporter = ShellFileReporter(str(resources_dir / "invalid_syntax.sh"))
        assert reporter.lines() == {9, 12, 15, 18}

    def test_handle_missing_file(self, tmp_path):
        reporter = ShellFileReporter(str(tmp_path / "missing_file.sh"))
        assert reporter.lines() == set()

    def test_handle_binary_file(self, tmp_path):
        file_path = tmp_path / "missing_file.sh"
        file_path.write_bytes(bytes.fromhex("348765F32190"))
        reporter = ShellFileReporter(str(file_path))
        assert reporter.lines() == set()

    def test_handle_non_file(self, tmp_path):
        file_path = tmp_path / "missing_file.sh"
        file_path.mkdir()
        reporter = ShellFileReporter(str(file_path))
        assert reporter.lines() == set()


def test_filename_suffix_should_match_pattern():
    suffix = filename_suffix()
    assert re.match(r".+?\.\d+\.[a-zA-Z]+", suffix)


class TestCovLineParser:
    def test_parse_result_matches_reference(self):
        parser = CovLineParser()
        for chunk in COVERAGE_LINE_CHUNKS:
            parser.parse(chunk)
        parser.flush()

        assert parser.line_data == COVERAGE_LINE_COVERAGE

    def test_parse_should_raise_for_incomplete_line(self):
        parser = CovLineParser()
        with pytest.raises(ValueError, match="could not parse line"):
            parser.parse(
                b"COV:::/home/dummy_user/dummy_dir_b:::a line with missing line number\n"
            )


class TestCoverageParserThread:
    class WriterThread(threading.Thread):
        def __init__(self, fifo_path: Path):
            super().__init__()
            self._fifo_path = fifo_path

        def run(self):
            logger.debug("writer start")
            with self._fifo_path.open("wb") as fd:
                for c in COVERAGE_LINE_CHUNKS[0:2]:
                    fd.write(c)
                    sleep(0.1)

            sleep(0.1)
            with self._fifo_path.open("wb") as fd:
                for c in COVERAGE_LINE_CHUNKS[2:]:
                    fd.write(c)
                    sleep(0.1)

            logger.debug("writer done")

    class CovLineParserSpy(CovLineParser):
        def __init__(self):
            super().__init__()
            self.recorded_lines = []

        def _report_lines(self, lines: list[str]) -> None:
            self.recorded_lines.extend(lines)
            super()._report_lines(lines)

    class CoverageWriterFake:
        def __init__(self):
            self.line_data: LineData = defaultdict(set)
            self.arc_data: ArcData = defaultdict(list)

        def write(self, line_data: LineData, arc_data: ArcData) -> None:
            self.line_data.update(line_data)
            self.arc_data.update(arc_data)

    def test_lines_should_match_reference(self):
        parser = self.CovLineParserSpy()
        writer = self.CoverageWriterFake()
        parser_thread = CoverageParserThread(
            coverage_writer=writer,
            name="CoverageParserThread",
            parser=parser,
        )
        parser_thread.start()

        writer_thread = self.WriterThread(fifo_path=parser_thread.fifo_path)
        writer_thread.start()
        writer_thread.join()

        parser_thread.stop()
        parser_thread.join()

        assert parser.recorded_lines == COVERAGE_LINES

        for filename, arcs in COVERAGE_ARC_COVERAGE.items():
            assert set(writer.arc_data[filename]) == arcs
        for filename, lines in COVERAGE_LINE_COVERAGE.items():
            assert writer.line_data[filename] == lines



class TestCoverageWriter:
    def test_write_should_produce_readable_file(self, dummy_project_dir):
        data_file_path = dummy_project_dir.joinpath("coverage-data.db")
        writer = CoverageWriter(mock.Mock(spec=ShellPlugin), data_file_path)
        writer.write(COVERAGE_LINE_COVERAGE, COVERAGE_ARC_COVERAGE)

        concrete_data_file_path = next(
            data_file_path.parent.glob(data_file_path.stem + "*")
        )
        cov_db = coverage.CoverageData(
            basename=str(concrete_data_file_path), suffix=False
        )
        cov_db.read()

        assert cov_db.measured_files() == set(COVERAGE_LINE_COVERAGE.keys())

        if cov_db.has_arcs():
            for filename, arcs in COVERAGE_ARC_COVERAGE.items():
                assert cov_db.arcs(filename) == sorted(arcs)
        else:
            for filename, lines in COVERAGE_LINE_COVERAGE.items():
                assert cov_db.lines(filename) == sorted(lines)

    def test_writer_should_prefer_pytest_cov_env_vars(
        self, dummy_project_dir, tmp_path, monkeypatch
    ):
        pytest_cov_path = tmp_path / "coverage"
        monkeypatch.setenv("COV_CORE_DATAFILE", str(pytest_cov_path))

        data_file_path = dummy_project_dir.joinpath("coverage-data.db")
        writer = CoverageWriter(mock.Mock(spec=ShellPlugin), data_file_path)
        writer.write(COVERAGE_LINE_COVERAGE, COVERAGE_ARC_COVERAGE)

        assert list(data_file_path.parent.glob(data_file_path.stem + "*")) == []

        concrete_pytest_cov_path = next(
            pytest_cov_path.parent.glob(pytest_cov_path.stem + "*")
        )

        assert concrete_pytest_cov_path.is_file()


class TestPatchedPopen:
    @pytest.mark.parametrize("is_recording", [(True), (False)])
    def test_call_should_execute_example(
        self,
        is_recording,
        resources_dir,
        dummy_project_dir,
        monkeypatch,
    ):
        monkeypatch.chdir(dummy_project_dir)

        cov = None
        if is_recording:
            cov = coverage.Coverage.current()
            if cov is None:  # pragma: no cover
                # start coverage in case pytest was not executed with the coverage module. Otherwise, we just recod to
                # the parent coverage
                cov = coverage.Coverage()
            cov.start()
        else:
            monkeypatch.setattr(coverage.Coverage, "current", lambda: None)

        test_sh_path = resources_dir / "testproject" / "test.sh"

        proc = PatchedPopen(
            ["bash", test_sh_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
            cwd=dummy_project_dir,
            env={**os.environ, "PYTHONWARNINGS": "ignore::CoverageWarning"},
        )
        proc.wait()

        if cov is not None:  # pragma: no cover
            cov.stop()

        assert proc.stderr.read() == ""
        assert proc.stdout.read() == SYNTAX_EXAMPLE_STDOUT


class TestMonitorThread:
    class MainThreadStub:
        def join(self):
            return

    def test_run_should_wait_for_main_thread_join(self, dummy_project_dir):
        data_file_path = dummy_project_dir.joinpath("coverage-data.db")

        parser_thread = CoverageParserThread(
            coverage_writer=CoverageWriter(mock.Mock(spec=ShellPlugin), data_file_path),
        )
        parser_thread.start()

        monitor_thread = MonitorThread(
            parser_thread=parser_thread, main_thread=self.MainThreadStub()
        )
        monitor_thread.start()


class TestShellPlugin:
    def test_init_cover_always(self):
        plugin = ShellPlugin({"cover_always": True})
        del plugin

    def test_file_tracer_should_return_None(self):
        plugin = ShellPlugin({})
        assert plugin.file_tracer("foobar") is None

    def test_file_reporter_should_return_instance(self):
        plugin = ShellPlugin({})
        plugin.configure(coverage.Coverage().config)
        reporter = plugin.file_reporter("foobar")
        assert isinstance(reporter, ShellFileReporter)
        assert reporter.filename == "foobar"

    def test_find_executable_files_should_find_shell_files(self, examples_dir):
        plugin = ShellPlugin({})
        plugin.configure(coverage.Coverage().config)

        executable_files = plugin.find_executable_files(str(examples_dir))

        assert [Path(f) for f in sorted(executable_files)] == [
            examples_dir / "shell-file.weird.suffix",
        ]

    def test_find_executable_files_should_find_symlinks(self, tmp_path):
        plugin = ShellPlugin({})

        foo_file_path = tmp_path.joinpath("foo.sh")
        foo_file_path.write_text("#!/usr/bin/env bash\necho foo")
        foo_file_link = foo_file_path.with_suffix(".link.sh")
        foo_file_link.symlink_to(foo_file_path)

        bar_dir_path = tmp_path.joinpath("bar")
        bar_dir_path.mkdir()
        bar_link_path = bar_dir_path.with_suffix(".link")
        bar_link_path.symlink_to(bar_dir_path)

        executable_files = plugin.find_executable_files(str(tmp_path))

        assert set(executable_files) == {str(f) for f in (foo_file_path, foo_file_link)}
