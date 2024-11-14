# mypy: disable-error-code="no-untyped-def,no-untyped-call"
import pytest
import pprint

from typing import Any, Union, Optional
from collections.abc import Callable, Set

import os
import logging
from pathlib import Path

from sortedcontainers import SortedSet

from tree_sitter import Node

from coverage.types import TArc, TLineNo
from coverage_sh.models import (
    Branch,
    Point,
    format_tree,
    format_branch,
    format_point,
    Symbols,
    entry,
    exit,
    to_arc,
)
from coverage_sh.parsers import (
    # ShellFileParser,
    parse,
)

logger = logging.getLogger(__name__)

syntax_examples = Path("tests/resources/syntax_examples")

@pytest.mark.parametrize("predicate", [
    lambda _: True,
    lambda _: False,
])
@pytest.mark.parametrize("node", [
    None,
])
def test_format_tree(node: Optional[Node], predicate: Callable[[Node], bool]) -> None:
    format_tree(node)

def diff_arcs(path: Path, actual: Set[TArc], expected: Set[TArc]) -> str:
    left_only = actual - expected
    right_only = expected - actual

    source_code = path.read_text().split("\n")
    logger.debug("\n".join(f"[{i}] {line}" for i, line in enumerate(source_code, 1)))

    def line(lineno: int) -> str:
        try:
            if lineno <= 0:
                return "<entry/exit>"
            return source_code[lineno - 1]
        except IndexError:
            return f"<out-of-range: {lineno} / {len(source_code)}>"

    left_onlys = [f"{(source, destination)}: {line(source)!r} -> {line(destination)!r}" for source, destination in sorted(left_only)]
    right_onlys = [f"{(source, destination)}: {line(source)!r} -> {line(destination)!r}" for source, destination in sorted(right_only)]
    return "\n".join(["left_only:"] + left_onlys + ["right_only:"] + right_onlys)

def diff_branches(path: Path, actual: Set[Branch], expected: list[Branch]) -> str:
    logger.debug(f"expected: {pprint.pformat(list(map(format_branch, expected)))}")

    left_only = set(actual) - set(expected)
    right_only = set(expected) - set(actual)

    source_code = path.read_text().split("\n")
    logger.debug("\n" + "\n".join(f"[{i}] {line}" for i, line in enumerate(source_code, 1)))

    def underline(s: str) -> str:
        return "\033[4m" + s + "\033[0m"

    def line(point: Point) -> str:
        try:
            row, column = point
            if row < 0:
                return "<entry/exit>"
            line = source_code[row] + " "
            line = line[:column] + underline(line[column]) + line[column + 1:]
            return f"{line} {format_point(point)}"
        except IndexError:
            return f"<out-of-range: {row + 1} / {len(source_code)}>"

    def key(branch: Branch) -> Any:
        return (branch[0][0], branch[1][0]), (branch[0][1], branch[1][1])

    left_onlys = [f"{(format_point(source), format_point(destination))}:\t{line(source)} -> {line(destination)}" for source, destination in sorted(left_only, key=key)]
    right_onlys = [f"{(format_point(source), format_point(destination))}:\t{line(source)} -> {line(destination)}" for source, destination in sorted(right_only, key=key)]
    return "\n".join(["left_only:"] + left_onlys + ["right_only:"] + right_onlys)

@pytest.mark.parametrize("path,branches", [
    (
        syntax_examples / "if_statement.sh",
        lambda symbols: {
            # program[0] -> program.statements[0]
            (entry.start_point, symbols["program"][0]["variable_assignment"][0].node.end_point),
            # variable_assignment -> if
            (symbols["variable_assignment"][0].node.end_point, symbols["if_statement"][0].node.start_point),
            # if -> if.conditions[0]
            (symbols["if_statement"][0]["if"][0].node.end_point, symbols["if_statement"][0]["if.condition"][0].node.start_point),
            # if.conditions[-1] -> if.then
            (symbols["if_statement"][0]["if.condition"][-1].node.end_point, symbols["if_statement"][0]["then"][0].node.start_point),
            # if matched: if.then -> if.statements[0]
            (symbols["if_statement"][0]["then"][0].node.end_point, symbols["if_statement"][0]["statement"][0].node.start_point),
            # if[1].statements[0] -> if[1].statements[1]
            (symbols["if_statement"][0]["statement"][0].node.end_point, symbols["if_statement"][0]["statement"][1].node.start_point),
            # if[1].statements[1] -> if[1].statements[2]
            (symbols["if_statement"][0]["statement"][1].node.end_point, symbols["if_statement"][0]["statement"][2].node.start_point),
            # if.statements[-1] -> fi
            (symbols["if_statement"][0]["statement"][-1].node.end_point, symbols["if_statement"][0].children["fi"][0].start_point),
            # if mismatch: if.conditions[-1] -> fi
            (symbols["if_statement"][0]["if.condition"][-1].node.end_point, symbols["if_statement"][0].children["fi"][0].start_point),
            # fi -> exit
            (symbols["if_statement"][0].children["fi"][0].end_point, exit.start_point),
        },
    ),
    (
        syntax_examples / "multiple_elifs_example.sh",
        lambda symbols: {
            # program[0] -> program.statements[0]
            (entry.start_point, symbols["program"][0]["variable_assignment"][0].node.end_point),
            # variable_assignment -> if
            (symbols["variable_assignment"][0].node.end_point, symbols["if_statement"][0].node.start_point),
            # if -> if.conditions[0]
            (symbols["if_statement"][0]["if"][0].node.end_point, symbols["if_statement"][0]["if.condition"][0].node.start_point),
            # if.conditions[-1] -> if.then
            (symbols["if_statement"][0]["if.condition"][-1].node.end_point, symbols["if_statement"][0]["then"][0].node.start_point),
            # if matched: if.then -> if.statements[0]
            (symbols["if_statement"][0]["then"][0].node.end_point, symbols["if_statement"][0]["statement"][0].node.start_point),
            # if.statements[-1] -> fi
            (symbols["if_statement"][0]["statement"][-1].node.end_point, symbols["if_statement"][0].nodes_by_type["fi"][0].start_point),
            # if mismatch: if.conditions[-1] -> elif[0]
            (symbols["if_statement"][0]["if.condition"][-1].node.end_point, symbols["if_statement"][0]["elif_clause"][0].node.start_point),
            # elif[0] -> elif[0].conditions[0]
            (symbols["if_statement"][0]["elif_clause"][0].children["elif"][0].end_point, symbols["if_statement"][0]["elif_clause"][0]["elif.condition"][0].node.start_point),
            # elif[0].conditions[-1] -> elif[0].then
            (symbols["if_statement"][0]["elif_clause"][0]["elif.condition"][-1].node.end_point, symbols["if_statement"][0]["elif_clause"][0]["then"][0].node.start_point),
            # elif[0] matched:  elif[0].then -> elif[0].statements[0]
            (symbols["if_statement"][0]["elif_clause"][0].children["then"][0].end_point, symbols["if_statement"][0]["elif_clause"][0]["statement"][0].node.start_point),
            # elif[0].statements[-1] -> fi
            (symbols["if_statement"][0]["elif_clause"][0]["statement"][-1].node.end_point, symbols["if_statement"][0].nodes_by_type["fi"][0].start_point),
            # elif[0] mismatched: elif[0].condition[-1] -> elif[1]
            (symbols["if_statement"][0]["elif_clause"][0]["elif.condition"][-1].node.end_point, symbols["if_statement"][0]["elif_clause"][1].node.start_point),
            # elif[1] -> elif[0].conditions[0]
            (symbols["if_statement"][0]["elif_clause"][1].children["elif"][0].end_point, symbols["if_statement"][0]["elif_clause"][1]["elif.condition"][0].node.start_point),
            # elif[1] matched: elif[1].conditions[-1] -> elif[0].then
            (symbols["if_statement"][0]["elif_clause"][1]["elif.condition"][-1].node.end_point, symbols["if_statement"][0]["elif_clause"][1]["then"][0].node.start_point),
            # elif[1] matched: elif[1].then -> elif[1].statements[0]
            (symbols["if_statement"][0]["elif_clause"][1].children["then"][0].end_point, symbols["if_statement"][0]["elif_clause"][1]["statement"][0].node.start_point),
            # elif[1].statements[0] -> elif[1].statements[1]
            (symbols["if_statement"][0]["elif_clause"][1]["statement"][0].node.end_point, symbols["if_statement"][0]["elif_clause"][1]["statement"][1].node.start_point),
            # elif[1].statements[-1] -> fi
            (symbols["if_statement"][0]["elif_clause"][1]["statement"][-1].node.end_point, symbols["if_statement"][0].nodes_by_type["fi"][0].start_point),
            # elif[1] mismatched: elif[1].conditions[-1] -> else
            (symbols["if_statement"][0]["elif_clause"][1]["elif.condition"][-1].node.end_point, symbols["if_statement"][0]["else_clause"][0].node.start_point),
            # else matched: else -> else.statmeents[0]
            (symbols["if_statement"][0]["else_clause"][0].children["else"][0].end_point, symbols["if_statement"][0]["else_clause"][0]["statement"][0].node.start_point),
            # else.statements[0] -> else.statements[1]
            (symbols["if_statement"][0]["else_clause"][0]["statement"][0].node.end_point, symbols["if_statement"][0]["else_clause"][0]["statement"][1].node.start_point),
            # else.statements[-1] -> fi
            (symbols["if_statement"][0]["else_clause"][0]["statement"][-1].node.end_point, symbols["if_statement"][0].nodes_by_type["fi"][0].start_point),
            # fi -> true
            (symbols["if_statement"][0].nodes_by_type["fi"][0].end_point, symbols["program"][0]["true"][0].node.start_point),
            # true -> exit
            (symbols["program"][0]["true"][0].node.end_point, exit.start_point),
        },
    ),
    (
        syntax_examples / "function.sh",
        lambda symbols: {
            # program[0] -> program.statements[0] function_definition 
            (entry.start_point, symbols["program"][0].children["function_definition"][0].start_point),
            # function_definition -> helloworld
            (symbols["program"][0].children["function_definition"][0].end_point, symbols["program"][0].children["command"][0].start_point),
            # program.statements[-1] -> exit
            (symbols["program"][0].children["command"][0].end_point, exit.start_point),
            # function_definition[0] -> function_definition[0].compound_statement.statements[0]
            (entry.start_point, symbols["function_definition"][0]["compound_statement"][0]["statement"][0].node.end_point),
            # function_definition[0].compound_statement.statements[0] -> function_definition[0].compound_statement.statements[1]
            (symbols["function_definition"][0]["compound_statement"][0]["statement"][0].node.end_point, symbols["function_definition"][0]["compound_statement"][0]["statement"][1].node.start_point),
            # function_definition[0].compound_statement.statements[1] -> function_definition[0].compound_statement.statements[2]
            (symbols["function_definition"][0]["compound_statement"][0]["statement"][1].node.end_point, symbols["function_definition"][0]["compound_statement"][0]["statement"][2].node.start_point),
            # function_definition[0].compound_statement.statements[-1] -> exit
            (symbols["function_definition"][0]["compound_statement"][0]["statement"][2].node.end_point, exit.start_point),
        },
    ),
    (
        syntax_examples / "for_loop.sh",
        lambda symbols: [
            # program[0] -> program.statements[0] for_statement
            (entry.start_point, symbols["for_statement"][0].node.start_point),
            # for_statement.do -> for_statement.statements[0]
            (symbols["for_statement"][0]["do_group"][0]["do"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["command"][0].node.start_point),
            # for_statement.do_group.statements[-1] -> for_statement
            (symbols["for_statement"][0]["do_group"][0]["command"][0].node.end_point, symbols["for_statement"][0].node.start_point),
            # for_statement -> for_statement.done
            (symbols["for_statement"][0].node.start_point, symbols["for_statement"][0]["do_group"][0]["done"][0].node.start_point),
            # program.statements[-1] -> exit
            (symbols["for_statement"][0].node.end_point, exit.start_point),
        ],
    ),
    (
        syntax_examples / "for_loop_break.sh",
        lambda symbols: [
            # program[0] -> program.statements[0] for_statement
            (entry.start_point, symbols["for_statement"][0].node.start_point),
            # for_statement.do -> for_statement.statements[0]
            (symbols["for_statement"][0]["do_group"][0]["do"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0].node.start_point),
            # for_statement.do_group.if_statement[0].if -> for_statement.do_group.if_statement[0].condition[0]
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["if"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["condition"][0].node.start_point),
            # if matched: for_statement.do_group.if_statement[0].condition[-1] -> for_statement.do_group.if_statement[0].then
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["condition"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["then"][0].node.start_point),
            # for_statement.do_group.if_statement[0].then -> for_statement.do_group.if_statement[0].statements[0]
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["then"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["statement"][0].node.start_point),
            # for_statement.do_group.if_statement.break -> for_statement.done
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["break"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["done"][0].node.start_point),
            # if mismatch: for_statement.do_group.if_statement[0].condition[-1] -> for_statement.do_group.if_statement[0].fi
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["condition"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["fi"][0].node.start_point),
            # for_statement.do_group.statements[-1] -> for_statement
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["fi"][0].node.end_point, symbols["for_statement"][0].node.start_point),
            # for_statement -> for_statement.done
            (symbols["for_statement"][0].node.start_point, symbols["for_statement"][0]["do_group"][0]["done"][0].node.start_point),
            # program.statements[-1] -> exit
            (symbols["for_statement"][0].node.end_point, exit.start_point),
        ],
    ),
    (
        syntax_examples / "for_loop_continue.sh",
        lambda symbols: [
            # program[0] -> program.statements[0] for_statement
            (entry.start_point, symbols["for_statement"][0].node.start_point),
            # for_statement.do -> for_statement.statements[0]
            (symbols["for_statement"][0]["do_group"][0]["do"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0].node.start_point),
            # for_statement.do_group.if_statement[0].if -> for_statement.do_group.if_statement[0].condition[0]
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["if"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["condition"][0].node.start_point),
            # if matched: for_statement.do_group.if_statement[0].condition[-1] -> for_statement.do_group.if_statement[0].then
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["condition"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["then"][0].node.start_point),
            # for_statement.do_group.if_statement[0].then -> for_statement.do_group.if_statement[0].statements[0]
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["then"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["statement"][0].node.start_point),
            # for_statement.do_group.if_statement.continue -> for_statement
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["continue"][0].node.end_point, symbols["for_statement"][0].node.start_point),
            # if mismatch: for_statement.do_group.if_statement[0].condition[-1] -> for_statement.do_group.if_statement[0].fi
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["condition"][0].node.end_point, symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["fi"][0].node.start_point),
            # for_statement.do_group.statements[-1] -> for_statement
            (symbols["for_statement"][0]["do_group"][0]["if_statement"][0]["fi"][0].node.end_point, symbols["for_statement"][0].node.start_point),
            # for_statement -> for_statement.done
            (symbols["for_statement"][0].node.start_point, symbols["for_statement"][0]["do_group"][0]["done"][0].node.start_point),
            # program.statements[-1] -> exit
            (symbols["for_statement"][0].node.end_point, exit.start_point),
        ],
    ),
    (
        syntax_examples / "while_loop.sh",
        lambda symbols: [
            # program[0] -> program.statements[0] i=0
            (entry.start_point, symbols["program"][0]["statement"][0].node.end_point),
            # program.statement[0] -> program.statement[1] while_statement
            (symbols["program"][0]["statement"][0].node.end_point, symbols["program"][0]["statement"][1].node.start_point),
            # while_statement.do -> while_statement.statements[0]
            (symbols["while_statement"][0]["do_group"][0]["do"][0].node.end_point, symbols["while_statement"][0]["do_group"][0]["statement"][0].node.start_point),
            # while_statement.do_group.statements[0] -> while_statement.do_group.statements[1]
            (symbols["while_statement"][0]["do_group"][0]["statement"][0].node.end_point, symbols["while_statement"][0]["do_group"][0]["statement"][1].node.start_point),
            # while_statement.do_group.statements[-1] -> while_statement
            (symbols["while_statement"][0]["do_group"][0]["statement"][-1].node.end_point, symbols["while_statement"][0]["condition"][0].node.start_point),
            # while_statement -> while_statement.done
            (symbols["while_statement"][0].node.start_point, symbols["while_statement"][0]["do_group"][0]["done"][0].node.start_point),
            # program.statements[-1] -> exit
            (symbols["while_statement"][0].node.end_point, exit.start_point),
        ],
    ),
])
def test_branches(path: Path, branches: Callable[[Symbols], list[Branch]]):
    shell_file = parse(path)
    # expected = set(map(lambda branch: (branch[0][0], branch[1][0]), branches_))
    expected: set[TArc] = set(map(to_arc, branches(shell_file.symbols)))
    # assert shell_file.arcs == expected, diff_arcs(path, shell_file.arcs, expected)

    actual_branches = shell_file.branches
    expected_branches = branches(shell_file.symbols)
    assert set(map(format_branch, actual_branches)) == set(map(format_branch, expected_branches)), diff_branches(path, actual_branches, expected_branches)
