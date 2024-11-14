from __future__ import annotations

import builtins
import importlib.resources
import sys
import os
import inspect
import ctypes
import re
import fnmatch
import itertools
import functools
import bisect
import logging
import pprint

from typing import (
    TYPE_CHECKING,
    ClassVar,
    TypeVar,
    Any,
    IO,
    Union,
    Optional,
    Literal,
    cast,
    Protocol,
    runtime_checkable,
    overload,
)
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Generator, Collection, Sequence, Set, Mapping
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from more_itertools import pairwise
from sortedcontainers import SortedSet

from tree_sitter import Language, Parser, Tree, Node
import tree_sitter_bash

from coverage.types import TConfigurable, TArc, TLineNo
from coverage_sh.utils import dropuntil, instance_of, find_index
from coverage_sh.searches import dijkstra
from coverage_sh.models import (
    Point,
    Range,
    Branch,
    of, not_of,
    is_statement,
    is_break,
    is_continue,
    is_return,
    to_arc,
    to_line,
    node_to_line,
    format_tree,
    format_branch,
    entry,
    exit,
    Shebang,
    ShellFile,
    EXECUTABLE_NODE_TYPES,
)

if sys.version_info < (3, 10): # pragma: no-cover-if-python-gte-310
    from typing_extensions import TypeGuard, Self
else: # pragma: no-cover-if-python-lt-310
    from typing import TypeGuard, Self

if TYPE_CHECKING:
    _LoggerAdapter = logging.LoggerAdapter[logging.Logger]
else:
    _LoggerAdapter = logging.LoggerAdapter

T = TypeVar("T")

languages = {
    name: Language(module.language())
    for name, module in {
        "bash": tree_sitter_bash,
    }.items()
}
parsers = {
    name: Parser(language)
    for name, language in languages.items()
}

logger = logging.getLogger(__name__)

def parse(
    filename: str | os.PathLike[str],
    language: Optional[str] = None,
    default_language: str = "bash",
    exclude_regex: Optional[str] = None,
) -> ShellFile:
    path = Path(filename)
    source = path.read_text()
    shebang = Shebang.parse(source)
    language = language or Shebang.infer_language(shebang) or default_language
    tree = parsers[language].parse(source.encode("utf-8"))

    lines: set[TLineNo] = set()
    excluded_lines: set[TLineNo] = set()

    branch: Branch
    branches: set[Branch] = set()

    class LevelAdapter(_LoggerAdapter):
        def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
            indent = cast(str, self.extra["indentation"]) * cast(int, self.extra["level"])
            return f"{indent}{msg}", kwargs

    def parse_ast(node: Node, indentation: str = "\t", level: int = 0) -> None:
        level_logger = LevelAdapter(logger, {"indentation": indentation, "level": level})
        level_logger.debug(f"[{node.type}]: {node!r} {node.text!r}")

        for child in node.children:
            parse_ast(child, indentation=indentation, level=level + 1)

    parse_ast(tree.root_node)

    query = languages[language].query(
        importlib.resources.files("coverage_sh").joinpath("queries/query.scm").read_text()
    )

    logger.error("\n".join(format_tree(tree.root_node)))

    captures: dict[str, list[Node]] = query.captures(tree.root_node)
    for capture_name, nodes in captures.items():
        for i, node in enumerate(nodes):
            name = f"{capture_name}[{i}]"
            # logger.error(f"{name}: {node!r} {node.text!r}")

    # logger.error(pprint.pformat(captures))

    comments = captures.get("comment", [])
    for comment in comments:
        assert comment.text
        if exclude_regex is not None and re.search(exclude_regex, comment.text.decode()):
            excluded_lines.add(to_line(comment.start_point))

    if_statements = captures.get("if_statement", [])
    logger.warning(f"if_statements: {pprint.pformat(if_statements)}")

    elif_clauses: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for elif_clause in captures.get("elif_clause", []):
        elif_clauses[cast(Node, elif_clause.parent)].add(elif_clause)
    logger.warning(f"elif_clauses: {pprint.pformat(elif_clauses)}")

    else_clauses: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for else_clause in captures.get("else_clause", []):
        else_clauses[cast(Node, else_clause.parent)].add(else_clause)
    logger.warning(f"else_clauses: {pprint.pformat(else_clauses)}")

    if_statement_fis: dict[Node, Node] = {
        cast(Node, fi.parent): fi
        for fi in captures.get("if_statement.fi", [])
    }

    conditions: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for condition in captures.get("condition", []):
        conditions[cast(Node, condition.parent)].add(condition)

    if_conditions: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for if_condition in captures.get("if.condition", []):
        if_conditions[cast(Node, if_condition.parent)].add(if_condition)

    elif_conditions: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for elif_condition in captures.get("elif.condition", []):
        elif_conditions[cast(Node, elif_condition.parent)].add(elif_condition)

    # def conditions(node: Node) -> SortedSet[Node]:
    #     if node.type == "if_statement":
    #         return if_conditions[node]
    #     if node.type == "elif_clause":
    #         return elif_conditions[node]
    #     if node.type == "while_statement":
    #         return 
    #     return SortedSet(key=node_to_line)

    # if_body_statements: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    # for key in ["if.statement", "elif.statement", "else.statement"]:
    #     for statement in captures.get(key, []):
    #         if_body_statements[cast(Node, statement.parent)].add(statement)

    statements: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for statement in captures.get("statement", []):
        # # NOTE: skipping conditions
        # if statement.parent in if_statements and statement in if_conditions[statement.parent]:
        #     continue
        statements[cast(Node, statement.parent)].add(statement)
    logger.error(f"statements: {pprint.pformat(statements)}")

    thens: dict[Node, Node] = {
        cast(Node, then.parent): then
        for then in captures.get("then", [])
    }

    breaks = set(captures.get("break", []))
    continues = set(captures.get("continue", []))
    returns = set(captures.get("return", []))
    exits = set(captures.get("exit", []))

    special_commands = set.union(breaks, continues, returns, exits)

    for scope, scoped_statements in statements.items():
        for source_statement, destination_statement in pairwise(scoped_statements - conditions[scope]):
            if source_statement in special_commands:
                continue

            branch = (source_statement.end_point, destination_statement.start_point)
            logger.info(f"(source_statement.end_point: {source_statement!r} text: {source_statement.text!r}, destination_statement.start_point: {destination_statement!r} text: {destination_statement.text!r}): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

    for multiline_statement in captures.get("multiline_statement", []):
        for lineno in range(to_line(multiline_statement.start_point), to_line(multiline_statement.end_point)):
            lines.add(lineno)

    def destination_point(statement: Node) -> Point:
        if statement.type in ("function_definition", "if_statement", "while_statement", "for_statement", "c_style_for_statement", "case_statement"):
            return statement.start_point
        return statement.end_point

    for scope in statements:
        if scope.type in ("program",):
            branch = (entry.end_point, destination_point(statements[scope][0]))
            logger.info(f"(entry.start_point, destination_point(statements[scope][0]: {statements[scope][0]!r})): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

            branch = (statements[scope][-1].end_point, exit.end_point)
            logger.info(f"(statements[scope][-1].end_point, exit.end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)
        elif scope.type == "compound_statement" and scope.parent and scope.parent.type == "function_definition":
            branch = (entry.end_point, destination_point(statements[scope][0]))
            logger.info(f"(entry.start_point, destination_point(statements[scope][0]: {statements[scope][0]!r})): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

            branch = (statements[scope][-1].end_point, exit.end_point)
            logger.info(f"(statements[scope][-1].end_point, exit.end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)
        elif scope.type == "if_statement":
            assert scope in thens
            branch = (thens[scope].end_point, statements[scope][0].start_point)
            logger.info(f"(thens[scope].end_point, statements[scope][0].end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

            assert scope in if_statement_fis
            # skip special commands
            if statements[scope][-1] not in special_commands:
                branch = (statements[scope][-1].end_point, if_statement_fis[scope].start_point)
                logger.info(f"(statements[scope][-1].end_point, if_statement_fis[scope].end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)
        elif scope.type == "elif_clause":
            assert scope.parent and scope.parent.type == "if_statement"
            assert scope in thens
            branch = (thens[scope].end_point, statements[scope][0].start_point)
            logger.info(f"(thens[scope].end_point, statements[scope][0].start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

            assert scope.parent in if_statement_fis
            if statements[scope][-1] not in special_commands:
                branch = (statements[scope][-1].end_point, if_statement_fis[scope.parent].start_point)
                logger.info(f"(statements[scope][-1].end_point, destination_point(if_statement_fis[scope.parent])): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)
        elif scope.type == "else_clause":
            assert scope.parent and scope.parent.type == "if_statement"
            assert scope.children[0].type == "else"
            branch = (scope.children[0].end_point, statements[scope][0].start_point)
            logger.info(f"(scope.children[0].end_point, statements[scope][0].end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

            assert scope.parent in if_statement_fis
            if statements[scope][-1] not in special_commands:
                branch = (statements[scope][-1].end_point, if_statement_fis[scope.parent].start_point)
                logger.info(f"(statements[scope][-1].end_point, if_statement_fis[scope.parent].end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)
        elif scope.type == "do_group":
            assert scope.parent

            do = scope.children[0]
            assert do.type == "do"
            # do -> statements[0]
            branch = (do.end_point, statements[scope][0].start_point)
            logger.info(f"(do.end_point, statements[scope][0].start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

            if statements[scope][-1] not in special_commands:
                if scope.parent.type == "for_statement":
                    branch = (statements[scope][-1].end_point, scope.parent.start_point)
                    logger.info(f"(statements[scope][-1].end_point, if_statement_fis[scope.parent].end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
                    branches.add(branch)
                elif scope.parent.type == "while_statement":
                    assert scope.parent in conditions
                    branch = (statements[scope][-1].end_point, conditions[scope.parent][0].start_point)
                    logger.info(f"(statements[scope][-1].end_point, conditions[scope.parent][0].start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
                    branches.add(branch)
                else:
                    raise NotImplementedError

            done = scope.children[-1]
            assert done.type == "done"
            # TODO: decide where to start -> done
            branch = (scope.parent.start_point, done.start_point)
            logger.info(f"(scope.parent.start_point, scope.children[-1].start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

    for if_statement in if_statements:
        assert if_statement in if_conditions
        for source_node, destination_node in pairwise(itertools.chain([if_statement.children[0]], if_conditions[if_statement], [thens[if_statement]])):
            branch = (source_node.end_point, destination_node.start_point)
            logger.info(f"(source_node.end_point: {source_node}, destination_node.start_point: {destination_node}): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

        for elif_clause in elif_clauses[if_statement]:
            assert elif_clause in elif_conditions
            logger.warning(f"elif_clause: {elif_clause}")

            for source_node, destination_node in pairwise(itertools.chain([elif_clause.children[0]], elif_conditions[elif_clause], [thens[elif_clause]])):
                branch = (source_node.end_point, destination_node.start_point)
                logger.info(f"(source_node.end_point: {source_node}, destination_node.start_point: {destination_node}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

    def clauses(if_statement: Node, else_clause_default: Optional[Node] = None) -> Iterable[Node]:
        return itertools.chain(
            [if_statement],
            elif_clauses.get(if_statement, []),
            else_clauses.get(if_statement, [else_clause_default] if else_clause_default else []),
        )

    for if_statement in if_statements:
        for source_clause, destination_clause in pairwise(clauses(if_statement, else_clause_default=if_statement_fis[if_statement])):
            branch = (conditions[source_clause][-1].end_point, destination_clause.start_point)
            logger.info(f"(source_clause.start_point, destination_clause.start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)

    def nearest_loop(node: Optional[Node]) -> Optional[Node]:
        while node:
            if node.type in ("while_statement", "for_statement", "c_style_for_statement"):
                return node

            node = node.parent
        return None

    for break_statement in captures.get("break", []):
        loop = nearest_loop(break_statement)
        assert loop, "break can only be used in a loop"
        do_group_index = find_index(of("do_group"), loop.children)
        assert do_group_index
        do_group = loop.children[do_group_index]
        done = do_group.children[-1]
        assert done.type == "done"
        branch = (break_statement.end_point, done.start_point)
        logger.info(f"(break_statement.end_point, done.start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
        branches.add(branch)

    for continue_statement in captures.get("continue", []):
        loop = nearest_loop(continue_statement)
        assert loop, "continue can only be used in a loop"
        if loop.type == "for_statement":
            branch = (continue_statement.end_point, loop.start_point)
            logger.info(f"(continue_statement.end_point, loop.start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
            branches.add(branch)
        else:
            raise NotImplementedError

    def interpolate(branch: Branch) -> list[Branch]:
        source: Point
        destination: Point
        source, destination = branch
        edges: dict[Point, set[Point]] = defaultdict(set)
        source_point: Point
        destination_point: Point
        for source_point, destination_point in branches:
            edges[source_point].add(destination_point)

        def predicate(path: list[Point]) -> bool:
            return path[-1] == destination

        def successors(path: list[Point]) -> Sequence[list[Point]]:
            return [path + [point] for point in edges[path[-1]]]

        def representation(path: list[Point]) -> tuple[Point, ...]:
            return tuple(path)

        def priority(path: list[Point]) -> int:
            return len(path)

        initial: list[Point] = [source]

        path = dijkstra(
            predicate=predicate,
            successors=successors,
            initials=[initial],
            representation=representation,
            priority=priority,
        ) or []
        return list(pairwise(path)) or [branch]

    traced_statements: dict[Node, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
    for traced_statement in captures.get("traced_statement", []):
        assert traced_statement.parent
        traced_statements[traced_statement.parent].add(traced_statement)

    logger.debug(f"traced_statements: {pprint.pformat(traced_statements)}")

    def scoped_stmts(scope: Node, statements: Sequence[Node]) -> Generator[Range]:
        if scope.type in ("program", "function_definition"):
            yield entry
        # elif scope.type in ("if_statement", "elif_clause"):
        #     yield scope.type.children[cast(int, find_index(of("then")))]
        # elif scope.type in ("else_clause",):
        #     yield scope.type.children[cast(int, find_index(of("else")))]
        # elif scope.type in ("while_statement", "for_statement", "c_style_for_statement"):
        #     yield scope.type.children[cast(int, find_index(of("do")))]
        # else:
        #     raise NotImplementedError

        yield from statements

        if scope.type in ("program", "function_definition"):
            yield exit

    branch_translations: dict[Branch, list[Branch]] = defaultdict(list)
    for parent, stmts in traced_statements.items():
        ranges = scoped_stmts(parent, stmts)
        for source_range, destination_range in pairwise(ranges):
            branch_translations[(source_range.end_point, destination_range.start_point)] = interpolate((source_range.end_point, destination_range.start_point))
    logger.debug(f"branch_translations: {pprint.pformat(branch_translations)}")

    return ShellFile(
        path,
        language,
        shebang,
        source,
        tree,
        captures,
        lines,
        excluded_lines,
        branches,
        branch_translations,
    )

