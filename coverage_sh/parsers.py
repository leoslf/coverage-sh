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
    Scope,
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
    scope_types,
    entry,
    exit,
    LoopScope,
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

# class ShellFileParser:
#     language: str
#     filename: str | os.PathLike[str]
#     source: str
#     exclude: str | None
# 
#     def __init__(
#         self,
#         language: str,
#         filename: str | os.PathLike[str],
#         source: str,
#         exclude: str | None = None,
#     ):
#         self.language = language
#         self.filename = filename
#         self.source = source
#         self.exclude = exclude
#         self._tree = parsers[language].parse(self.source.encode("utf-8"))
#         self.lines, self.branches, self.branch_translations = self.analyse(self._tree)
# 
#     @classmethod
#     def parse(cls, path: str | bytes | os.PathLike, exclude: Optional[str] = None) -> Self:
#         return self
# 
#     @functools.cached_property
#     def language(self) -> str:
#         ...
# 
#     @functools.cached_property
#     def logger(self) -> logging.Logger:
#         return logger.getChild(type(self).__name__).getChild(os.path.basename(self.filename))
# 
#     @functools.cached_property
#     def trace_logger(self) -> logging.Logger:
#         return logger.getChild(type(self).__name__).getChild("trace").getChild(os.path.basename(self.filename))
# 
#     @functools.cached_property
#     def excluded_lines(self) -> set[TLineNo]:
#         excluded_lines: set[TLineNo] = set()
# 
#         for lineno, line in enumerate(self.source.split("\n"), 1):
#             if self.exclude is not None and re.search(self.exclude, line):
#                 excluded_lines.add(lineno)
# 
#         return excluded_lines
# 
#     @property
#     def bash_supports_case_item_trace(self) -> bool:
#         # NOTE: bash's trace does not support showing case_items as of 5.2
#         return False
# 
#     @property
#     def bash_supports_case_item_termination_trace(self) -> bool:
#         # NOTE: bash's trace does not support showing case_items as of 5.2
#         return False
# 
#     @property
#     def bash_supports_else_clause_trace(self) -> bool:
#         # NOTE: bash's trace does not support showing else_clause as of 5.2
#         return False
# 
#     def analyse(self, tree: Tree) -> tuple[set[Branch], dict[Branch, set[Branch]]]:
#         lines: set[TLineNo] = set()
#         branches: set[Branch] = set()
#         translations: dict[Branch, set[Branch]] = defaultdict(set)
#         block_exits: dict[Node, set[Node]] = defaultdict(set)
# 
#         def parse_ast(node: Node, scope: Scope | None) -> None:
#             if node.type == "function_definition":
#                 self.trace_logger.debug(f"{node.type}: {node!r}")
#                 branch = (entry.start_point, node.start_point)
#                 self.trace_logger.debug(f"(entry.start_point: {entry}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 branch = (node.end_point, exit.start_point)
#                 self.trace_logger.debug(f"(node.end_point, exit.start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#             elif node.type == "if_statement":
#                 self.trace_logger.debug(f"if_statement: {node!r}")
#                 conditions = list(filter(of("command", "*_command", "compound_statement"), node.children_by_field_name("condition")))
#                 assert conditions, f"conditions are required as per the grammar: {node.children_by_field_name('condition')!r}"
#                 self.trace_logger.debug(f"conditions: {conditions}")
# 
#                 elif_clauses = list(filter(of("elif_clause"), node.children))
#                 self.trace_logger.debug(f"elif_clauses: {elif_clauses}")
#                 else_clauses = list(filter(of("else_clause"), node.children))
#                 assert len(else_clauses) <= 1
#                 self.trace_logger.debug(f"else_clauses: {else_clauses}")
#                 names = {
#                     node: "if",
#                     **{elif_clause: f"elif[{i}]" for i, elif_clause in enumerate(elif_clauses)},
#                     **{else_clause: "else" for else_clause in else_clauses},
#                 }
#                 clauses = {
#                     node: list(filter(is_statement, dropuntil(of("then"), node.children))),
#                     **{elif_clause: list(filter(is_statement, dropuntil(of("then"), elif_clause.children))) for elif_clause in elif_clauses},
#                     **{else_clause: list(filter(is_statement, dropuntil(of("else"), else_clause.children))) for else_clause in else_clauses},
#                 }
#                 for clause, statements in clauses.items():
#                     # logger.debug(f"clause: {clause}, statements: {statements}")
#                     branch = (clause.start_point, statements[0].start_point)
#                     branches.add(branch)
#                     self.trace_logger.debug(f"(clause.start_point: {clause}, self.nearest_next_line(statements[0]: {statements[0]}.start_point)): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     # branches.add((clause.start_point[0] + 1, statements[0].start_point[0] + 1))
# 
#                     # if not is_break(statements[-1]) and not is_continue(statements[-1]):
#                     #     branch = (statements[-1].end_point, node.end_point)
#                     #     branches.add(branch)
#                     #     self.trace_logger.debug(f"(statements[-1]: {statements[-1]}, node.end_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
# 
#                 for source, destination in pairwise(clauses.keys()):
#                     # mismatch
#                     branch = (source.start_point, destination.start_point)
#                     # arc = (source.start_point[0] + 1, destination.start_point[0] + 1)
#                     branches.add(branch)
#                     self.trace_logger.debug(f"[{names[source]} -> {names[destination]}] (source.start_point: {source}, destination.start_point: {destination}): {format_branch(branch)}, arc: {to_arc(branch)}")
# 
#                 if else_clauses:
#                     else_clause, = else_clauses
#                     statements = clauses[else_clause]
#                     if not self.bash_supports_else_clause_trace: # pragma: no branch
#                         previous_clause = list(clauses.keys())[-2]
# 
#                         branch = (previous_clause.start_point, statements[0].start_point)
#                         translations[branch] = {
#                             (previous_clause.start_point, else_clause.start_point),
#                             (else_clause.start_point, statements[0].start_point),
#                         }
#                         self.trace_logger.debug(f"[else] transitions[{branch}]: {set(map(format_branch, translations[branch]))}, arcs: {set(map(to_arc, translations[branch]))}")
# 
#             elif node.type == "c_style_for_statement":
#                 self.trace_logger.debug(f"c_style_for_statement: {node!r}")
#                 body = node.child_by_field_name("body")
#                 assert body, "body is required as per the grammar"
#                 self.trace_logger.debug(f"body: {body}")
# 
#                 initializers = node.children_by_field_name("initializer")
#                 self.trace_logger.debug(f"initializers: {initializers}")
# 
#                 conditions = node.children_by_field_name("condition")
#                 self.trace_logger.debug(f"conditions: {conditions}")
# 
#                 updates = node.children_by_field_name("update")
#                 self.trace_logger.debug(f"updates: {updates}")
# 
#                 statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
#                 assert not list(filter(of("comment"), statements))
# 
#                 branch = (node.start_point, statements[0].start_point)
#                 branches.add(branch)
#                 self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
# 
#                 if updates:
#                     branch = (statements[-1].end_point, updates[0].start_point)
#                     branches.add(branch)
#                     self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, updates[0].start_point: {updates[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     # loop_context.append((node, updates[0], statements))
#                 else:
#                     branch = (statements[-1].end_point, node.start_point)
#                     branches.add(branch)
#                     self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     # loop_context.append((node, node, statements))
# 
#             elif node.type == "for_statement":
#                 self.trace_logger.debug(f"for_statement: {node!r}")
# 
#                 variable = node.child_by_field_name("variable")
#                 self.trace_logger.debug(f"variable: {variable}")
#                 values = node.children_by_field_name("value")
#                 self.trace_logger.debug(f"values: {values}")
# 
#                 body = node.child_by_field_name("body")
#                 assert body, "body is required as per the grammar"
#                 self.trace_logger.debug(f"body: {body}")
# 
#                 statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
#                 assert statements, "at least one statement is required in for_statement as per the grammar"
#                 assert list(filter(not_of("comment"), statements)), "comments should not appear in statements"
# 
#                 branch = (node.start_point, statements[0].start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 branch = (statements[-1].end_point, node.end_point)
#                 self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, node.end_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
# 
#                 # loop_context.append((node, node, statements))
# 
#             elif node.type == "while_statement":
#                 self.trace_logger.debug(f"while_statement: {node!r}")
# 
#                 conditions = node.children_by_field_name("condition")
#                 assert conditions, "at least one condition is required for while_statement as per the grammar"
#                 self.trace_logger.debug(f"conditions: {conditions}")
# 
#                 body = node.child_by_field_name("body")
#                 assert body, "body is required as per the grammar"
#                 self.trace_logger.debug(f"body: {body}")
# 
#                 statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
#                 assert statements, "at least one statement is required in while_statement as per the grammar"
# 
#                 branch = (node.start_point, statements[0].start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 branch = (statements[-1].end_point, conditions[0].start_point)
#                 self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, conditions[0].start_point: {conditions[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 # loop_context.append((node, conditions[0], statements))
# 
#             elif node.type == "case_statement":
#                 self.trace_logger.debug(f"case_statement: {node!r}")
#                 value = node.child_by_field_name("value")
#                 self.trace_logger.debug(f"value: {value}")
#                 case_items = list(filter(of("case_item"), node.children))
#                 self.trace_logger.debug(f"case_items: {case_items}")
# 
#                 # TODO: find a way to be more precise on the branches for each condition in  each case_item
# 
#                 # only bridge from case_statement to the first case_item
#                 branch = (node.start_point, case_items[0].start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, case_items[0].start_point: {case_items[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 for i, case_item in enumerate(case_items):
#                     if statements := list(filter(is_statement, case_item.children)):
#                         # matched
#                         branch = (case_item.start_point, statements[0].start_point)
#                         self.trace_logger.debug(f"(case_item.start_point: {case_item}, statements[0].start_point: {statements}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                         branches.add(branch)
# 
#                         if not self.bash_supports_case_item_trace: # pragma: no branch
#                             # matched
#                             branch = (node.start_point, statements[0].start_point)
#                             self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                             translations[branch] = {
#                                 # case_statement.start_point -> case_item[0] -> ... -> case_item
#                                 *((source.start_point, destination.start_point) for source, destination in pairwise([node] + case_items[:i])),
#                                 # *((source.start_point[0] + 1, destination.start_point[0] + 1) for source, destination in pairwise([node] + case_items[:i])),
#                                 # case_item -> statements[0]
#                                 (case_item.start_point, statements[0].start_point),
#                                 # (case_item.start_point[0] + 1, statements[0].start_point[0] + 1),
#                             }
#                             self.trace_logger.debug(f"[case_items[{i}]: {case_item}] transitions[{branch}]: {set(map(format_branch, translations[branch]))}, arcs: {set(map(to_arc, translations[branch]))}")
# 
#                         # termination
#                         if termination := case_item.child_by_field_name("termination"):
#                             branch = (termination.end_point, node.end_point)
#                             self.trace_logger.debug(f"(termination.end_point: {termination}, node.end_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                             branches.add(branch)
# 
#                             if not self.bash_supports_case_item_termination_trace: # pragma: no branch
#                                 # TODO: consider making fi, done, esac, etc. executable?
#                                 branch = (statements[-1].end_point, node.end_point)
#                                 self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, node.end_point: {node}, round=True): {format_branch(branch)}, arc: {to_arc(branch)}")
# 
#                                 translations[branch] = {
#                                     # statements[-1] -> case_statement.end_point
#                                     (statements[-1].start_point, node.end_point),
#                                     # (statements[-1].start_point[0] + 1, node.end_point[0] + 1),
#                                     (node.end_point, node.end_point),
#                                     # (node.end_point[0] + 1, self.nearest_next_line(node.end_point[0] + 1)),
#                                 }
#                                 self.trace_logger.debug(f"[case_items[{i}]: termination] transitions[{branch}]: {set(map(format_branch, translations[branch]))}, arcs: {set(map(to_arc, translations[branch]))}")
# 
#                 for source, destination in pairwise(case_items):
#                     self.trace_logger.debug(f"source: {source}, source.children: {source.children}")
#                     # TODO: consider dropping unilt ) and taking while not
#                     if statements := list(filter(is_statement, source.children)):
#                         self.trace_logger.debug(f"statements: {statements}")
#                         # mismatch
#                         branch = (source.start_point, destination.start_point)
#                         self.trace_logger.debug(f"(source.start_point: {source}, destination.start_point: {destination}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                         branches.add(branch)
# 
#                     # fallthrough
#                     if fallthrough := source.child_by_field_name("fallthrough"):
#                         branch = (fallthrough.end_point, destination.start_point)
#                         self.trace_logger.debug(f"(fallthrough.end_point: {fallthrough}, destination.start_point: {destination}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                         branches.add(branch)
# 
# 
#                     assert fallthrough or source.child_by_field_name("termination"), f"case_item: {source!r} should have either fallthrough or termination"
#             elif node.type == "negated_command":
#                 self.trace_logger.debug(f"negated_command: {node!r}")
#                 _, child = node.children
#                 self.trace_logger.debug(f"child: {child}")
#                 branch = (node.start_point, child.start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, child.start_point: {child}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#             elif node.type == "ternary_expression":
#                 self.trace_logger.debug(f"ternary_expression: {node!r}")
#                 condition = node.child_by_field_name("condition")
#                 assert condition, "condition is required as per the grammar"
#                 self.trace_logger.debug(f"condition: {condition}")
# 
#                 consequence = node.child_by_field_name("consequence")
#                 assert consequence, "consequence is required as per the grammar"
#                 self.trace_logger.debug(f"consequence: {consequence}")
# 
#                 branch = (node.start_point, consequence.start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, consequence.start_point: {consequence}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 alternative = node.child_by_field_name("alternative")
#                 assert alternative, "alternative is required as per the grammar"
#                 self.trace_logger.debug(f"alternative: {alternative}")
# 
#                 branch = (node.start_point, alternative.start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, alternative.start_point: {alternative}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#             elif node.type == "binary_expression":
#                 self.trace_logger.debug(f"binary_expression: {node!r} {node.text!r}")
# 
#                 left = node.child_by_field_name("left")
#                 assert left, "left is required for binary_expression as per the grammar"
# 
#                 operator = node.child_by_field_name("operator")
#                 assert operator, "operator is required for binary_expression as per the grammar"
# 
#                 right = node.child_by_field_name("right")
#                 assert right, "right is required for binary_expression as per the grammar"
# 
#                 if operator.type in ("&&", "||"):
#                     branch = (left.end_point, right.start_point)
#                     self.trace_logger.debug(f"(left.end_point: {left} {left.text!r}, right.start_point: {right} {right.text!r}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     branches.add(branch)
# 
#             elif node.type in "variable_assignment":
#                 self.trace_logger.debug(f"variable_assignment: {node!r}")
# 
#                 name = node.child_by_field_name("name")
#                 assert name, "name is required for variable_assignment as per the grammar"
# 
#                 value = node.child_by_field_name("value")
#                 assert value, "value is required for variable_assignment as per the grammar"
# 
#                 self.trace_logger.debug(f"variable_assignment: name: {name} {name.text!r}, value: {value} {value.text!r}")
# 
#                 assert scope
#                 branch = (node.start_point, scope.next(node).start_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, scope.next(node)).start_point: {scope.next(node).start_point}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 branch = (value.end_point, node.start_point)
#                 self.trace_logger.debug(f"(value.end_point: {value}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#                 # TODO: do we need to handle previous -> node.end_point?
# 
#                 branch = (value.end_point, scope.next(node).start_point)
#                 self.trace_logger.debug(f"(value.end_point: {value}, scope.next(node).start_point: {scope.next(node)}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 translations[branch] = {
#                     (value.end_point, node.start_point),
#                     (node.start_point, scope.next(node).start_point),
#                 }
# 
#                 self.trace_logger.debug(f"[variable_assignment] transitions[{branch}]: {set(map(format_branch, translations[branch]))}, arcs: {set(map(to_arc, translations[branch]))}")
# 
#                 # self.trace_logger.debug(f"(node.end_point: {node}, node.parent.start_point: {node.parent}): {format_branch(branch)}, arc: {to_arc(branch)}")
# 
#             # elif node.type == "command_substitution":
#             #     # TODO: should be the same in subshell?
#             #     self.trace_logger.debug(f"command_substitution: {node!r}")
# 
#             #     # FIXME: this is pretty wrong
#             #     branch = (node.end_point, cast(Node, node.parent).start_point)
#             #     self.trace_logger.debug(f"(node.end_point: {node}, node.parent.start_point: {node.parent}): {format_branch(branch)}, arc: {to_arc(branch)}")
#             #     branches.add(branch)
# 
#             elif node.type == "test_command":
#                 self.trace_logger.debug(f"test_command: {node!r}")
#                 # TODO
# 
#             elif is_break(node):
#                 assert scope
#                 loop_scope = cast(LoopScope, scope.nearest(lambda ancestor: isinstance(ancestor, LoopScope)))
#                 assert loop_scope, f"break can only be used inside a loop scope, scopes: {list(scope.ancestors())}"
# 
#                 branch = (node.start_point, loop_scope.exit.end_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, loop_scope.exit.end_point: {loop_scope}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#             elif is_continue(node):
#                 assert scope
#                 loop_scope = cast(LoopScope, scope.nearest(lambda ancestor: isinstance(ancestor, LoopScope)))
#                 assert loop_scope, f"continue can only be used inside a loop scope, scopes: {list(scope.ancestors())}"
# 
#                 branch = (node.start_point, loop_scope.advance.end_point)
#                 self.trace_logger.debug(f"(node.start_point: {node}, loop_scope.advance.end_point: {loop_scope}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                 branches.add(branch)
# 
#             elif node.type in set(EXECUTABLE_NODE_TYPES) - {"elif_clause", "else_clause"}:
#                 assert scope
#                 self.trace_logger.debug(f"{node.type}: {node!r}, scope: {scope}")
#                 effective_scope = scope.nearest(lambda scope: node in scope)
#                 assert effective_scope
#                 self.trace_logger.debug(f"{node.type}: {node!r}, scope: {scope}, effective_scope: {effective_scope}")
# 
#                 if effective_scope.is_first_statement(node):
#                     branch = (effective_scope.entry.start_point, node.start_point)
#                     self.trace_logger.debug(f"(effective_scope.entry.start_point: {effective_scope.entry}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     branches.add(branch)
#                 elif node.end_point != effective_scope.next(node).end_point:
#                     branch = (node.end_point, effective_scope.next(node).end_point)
#                     self.trace_logger.debug(f"(node.end_point: {node} (text: {node.text!r}), effective_scope.next(node).end_point: {effective_scope.next(node)}): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     branches.add(branch)
#                     # if scope.is_last_statement(node):
#                     #     branch = (node.end_point, scope.end.end_point)
#                     # else:
#                     #     sibling = next_statement(node)
#                     #     self.trace_logger.debug(f"node: {node}, text: {node.text!r}, next_statement(node): {sibling}") 
#                     #     branch = (node.end_point, sibling.end_point)
#                     #     self.trace_logger.debug(f"(node.end_point, sibling.end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
#                     #     branches.add(branch)
#             else:
#                 # self.trace_logger.debug(f"{node.type}: {node!r}")
#                 pass
# 
#             if branch := next((branch for branch in branches if to_arc(branch) == (15, 24)), None): # type: ignore[arg-type]
#                 raise ValueError(f"branch: {format_branch(branch)!r}, arc: {to_arc(branch)}, added (15, 24) at {node!r}, text: {node.text!r}")
# 
# 
#             # if is_block(node):
#             #     branch = (node.end_point, scope.next(node).start_point)
#             #     self.trace_logger.debug("(node.end_point: {node}, scope.next(node).start_point: {next_statement(node)}: {to_arc(branch)}")
#             #     branches.add(branch)
# 
#             if scope_type := scope_types.get(node.type):
#                 scope = scope_type(node, scope)
# 
#             for child in node.children:
#                 parse_ast(child, scope)
# 
#         predicate = lambda _: True # matches(of("program"), is_statement, match="any")
#         self.trace_logger.debug("Tree:\n" + "\n".join(format_tree(tree.root_node, predicate=predicate, format=lambda node: repr(node) if node.type == "program" else f"{node!r} {node.text!s}")))
#         parse_ast(tree.root_node, None)
# 
#         # extra_lines = set((lineno, (source, destination)) for source, destination in branches for lineno in (source, destination) if lineno not in self.lines)
#         # self.trace_logger.debug(f"extra_lines: {extra_lines}")
#         # assert not extra_lines, f"extra lines in branches: {extra_lines!r}"
#         # self.trace_logger.debug(branches)
# 
#         branches.union(*translations.values())
#         assert all(branch not in branches for branch in translations), set(branches) & set(translations) 
#         self.logger.debug(f"executable_branches: {pprint.pformat(set(map(format_branch, branches)))}")
# 
#         return branches, translations
# 
#     # def source_token_lines(self):
#     #     pass

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

