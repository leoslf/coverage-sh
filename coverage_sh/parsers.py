from __future__ import annotations

import builtins
import sys
import os
import inspect
import ctypes
import re
import fnmatch
import itertools
import functools
import logging
import pprint

from typing import TYPE_CHECKING, ClassVar, TypeVar, Any, IO, Union, Literal, cast, Protocol, runtime_checkable, overload
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Collection, Sequence, Generator, Iterable, Iterator
from collections import defaultdict
from dataclasses import dataclass, field

from tree_sitter import Parser, Tree, Node, Range
from tree_sitter_languages import get_parser

from coverage.types import TConfigurable, TArc, TLineNo

T = TypeVar("T")
R = TypeVar("R")

if sys.version_info < (3, 10): # pragma: no-cover-if-python-gte-310
    from typing_extensions import TypeGuard
    def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
else: # pragma: no-cover-if-python-lt-310
    from typing import TypeGuard
    from itertools import pairwise

EXECUTABLE_NODE_TYPES = {
    "subshell",
    "redirected_statement",
    "variable_assignment",
    "variable_assignments",
    "command",
    "declaration_command",
    "unset_command",
    "test_command",
    "negated_command",
    "for_statement",
    "c_style_for_statement",
    "while_statement",
    "if_statement",
    "elif_clause",
    "else_clause",
    "case_statement",
    "pipeline",
    "list",
    "fi",
    "done",
    "esac"
}

entry: Range = Range((-1, 0), (-1, 0), 0, 0)
exit: Range = Range((-1, 0), (-1, 0), 0, 0)

# tree_sitter < 0.22
Point = tuple[int, int]
Branch = tuple[Point, Point]

Predicate = Callable[[Node], bool]
MatchMode = Literal["all", "any"]

logger = logging.getLogger(__name__)

def format_tree(
    node: Node | None,
    indentation: str = "\t",
    level: int = 0,
    index: int = 0,
    predicate: Callable[[Node], bool] = lambda _: True,
    format: Callable[[Node], str] = repr,
) -> Generator[str]:
    if node and predicate(node):
        yield f"{indentation * level}[{index}] {format(node)}"

        for child_index, child in enumerate(node.children):
            yield from format_tree(
                child,
                indentation=indentation,
                level=level + 1,
                index=child_index,
                predicate=predicate,
                format=format,
            )

def instance_of(type: type[R]) -> Callable[[T], TypeGuard[R]]:
    def inner(value: T) -> TypeGuard[R]:
        return isinstance(value, type)
    return inner

def to_line(point: Point, offset: int = 0) -> TLineNo:
    if (row := ctypes.c_int32(point[0] & ctypes.c_uint32(-1).value).value) < 0:
        return row
    return int(row) + 1 + offset

def to_arc(branch: Branch) -> TArc:
    source, destination = branch
    return (to_line(source), to_line(destination))

def format_point(point: Point) -> tuple[TLineNo, int]:
    _, column = point
    return (to_line(point), column)

def format_branch(branch: Branch, offset: int = 0) -> tuple[tuple[TLineNo, int], tuple[TLineNo, int]]:
    source, destination = branch
    return (format_point(source), format_point(destination))

def of(*types: str, match: MatchMode = "any") -> Callable[[Node], bool]:
    return lambda node: {"all": all, "any": any}[match](fnmatch.fnmatch(node.type, type) for type in types)

def matches(*predicates: Predicate, match: MatchMode) -> Predicate:
    return lambda node: {"all": all, "any": any}[match](predicate(node) for predicate in predicates)

is_statement = of(*EXECUTABLE_NODE_TYPES)
is_block = of("if_statement", "*for_statement", "while_statement", "case_statement")
is_loop = of("*for_statement", "while_statement")

@overload
def negate(predicate: Callable[[T], TypeGuard[R]]) -> Callable[[T], TypeGuard[R]]: ...
@overload
def negate(predicate: Callable[[T], bool]) -> Callable[[T], bool]: ...
def negate(predicate: Callable[[T], bool | TypeGuard[R]]) -> Callable[[T], bool | TypeGuard[R]]:
    def inner(value: T) -> bool | TypeGuard[R]:
        return not predicate(value)
    return inner

def not_of(*types: str, match: Literal["all", "any"] = "any") -> Callable[[Node], bool]:
    negations: dict[Literal["all", "any"], Literal["all", "any"]] = {"all": "any", "any": "all"}
    return negate(of(*types, match=negations[match]))

@overload
def dropuntil(predicate: Callable[[T], TypeGuard[R]], iterable: Iterable[T]) -> itertools.dropwhile[R]: ...
@overload
def dropuntil(predicate: Callable[[T], bool], iterable: Iterable[T]) -> itertools.dropwhile[T]: ...
def dropuntil(predicate: Any, iterable: Iterable[T]) -> Any:
    iterable = itertools.dropwhile(negate(predicate), iterable)
    next(iterable, None)
    return iterable

def is_command(name: str) -> Predicate:
    def inner(node: Node) -> bool:
        if node.type != "command":
            return False

        command_name = node.child_by_field_name("name")
        # logger.error(f"command_name: {command_name}")
        assert command_name, "command_name is required as per the grammar"
        # logger.error(f"command_name.children: {command_name.children}")
        command, *_ = command_name.children
        if command.type != "word":
            return False

        # NOTE: the tree cannot be edited
        return command.text.decode() == name
    return inner

@dataclass(frozen=True)
class Scope(metaclass=ABCMeta):
    type: ClassVar[str]
    node: Node
    parent: Scope | None = None

    def ancestors(self) -> Generator[Scope]:
        yield self
        if self.parent:
            yield from self.parent.ancestors()

    @property
    def exit(self) -> Node | Range:
        return self.node.children[-1]

    @functools.cached_property
    def statements(self) -> list[Node]:
        return list(filter(is_statement, self.node.children))

    def is_last_statement(self, statement: Node) -> bool:
        return statement == self.statements[-1]

    @property
    def end(self) -> Node | Range:
        """ The place where the control flow goes after executing the last statement. """
        return self.exit

    # TODO: cache?
    def next(self, statement: Node) -> Node | Range:
        predicate: Callable[[Node], bool] = lambda stmt: stmt == statement
        iterable = dropuntil(predicate, self.statements)
        value = next(iterable, None)
        return value or self.end

@dataclass(frozen=True)
class ProgramScope(Scope):
    type: ClassVar[str] = "program"

    @property
    def exit(self) -> Node | Range:
        return exit

@dataclass(frozen=True)
class FunctionScope(Scope):
    type: ClassVar[str] = "function_definition"

@dataclass(frozen=True)
class IfScope(Scope):
    type: ClassVar[str] = "if_statement"

@dataclass(frozen=True)
class LoopScope(Scope, metaclass=ABCMeta):
    @functools.cached_property
    def statements(self) -> list[Node]:
        body = self.node.child_by_field_name("body")
        assert body, "body is required as per the grammar"
        logger.debug(f"body: {body}")

        statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
        assert statements, "at least one statement is required in while_statement as per the grammar"
        return statements

    @property
    @abstractmethod
    def advance(self) -> Node | Range: ...

    @property
    def end(self) -> Node | Range:
        return self.advance

@dataclass(frozen=True)
class CStyleForLoopScope(LoopScope):
    type: ClassVar[str] = "c_style_for_statement"

    @property
    def advance(self) -> Node | Range:
        if updates := self.node.children_by_field_name("update"):
            return updates[0]
        return self.node.children[0]

@dataclass(frozen=True)
class ForLoopScope(LoopScope):
    type: ClassVar[str] = "for_statement"

    @property
    def advance(self) -> Node | Range:
        variable = self.node.child_by_field_name("variable")
        assert variable, "variable is required in for_statement as per the grammar"
        return variable

@dataclass(frozen=True)
class WhileLoopScope(LoopScope):
    type: ClassVar[str] = "while_statement"

    @property
    def advance(self) -> Node | Range:
        conditions = self.node.children_by_field_name("condition")
        assert conditions, "conditions are required in for_statement as per the grammar"
        return conditions[0]

@dataclass(frozen=True)
class SubShellScope(Scope):
    type: ClassVar[str] = "subshell"

@dataclass(frozen=True)
class CommandSubstitution(SubShellScope):
    type: ClassVar[str] = "command_substitution"

is_loop_scope: Callable[[Scope], TypeGuard[LoopScope]] = instance_of(LoopScope) # type: ignore[type-abstract]

def subclasses(cls: type[T]) -> Generator[type[T]]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from subclasses(subclass)

scope_types: dict[str, builtins.type[Scope]] = {
    cls.type: cls for cls in subclasses(Scope) if not inspect.isabstract(cls) and hasattr(cls, "type")
}

is_break = is_command("break")
is_continue = is_command("continue")
is_return = is_command("return")

# def next_statement(node: Node | None, scope: Scope) -> Range:
#     logger.debug(f"node: {node}")
#     while node and (node := node.next_sibling):
#         logger.debug(f"sibling: {node}")
#         if is_statement(node):
#             return node.range
#     return scope

class ShellFileParser:
    filename: str | os.PathLike[str]
    source: str
    exclude: str | None

    _parser: Parser = get_parser("bash")

    def __init__(self, filename: str | os.PathLike[str], source: str, exclude: str | None = None):
        self.filename = filename
        self.source = source
        self.exclude = exclude
        self._tree = self._parser.parse(self.source.encode("utf-8"))
        self.branches, self.branch_translations = self.analyse_branches(self._tree)

    @functools.cached_property
    def logger(self) -> logging.Logger:
        return logger.getChild(type(self).__name__).getChild(os.path.basename(self.filename))

    @functools.cached_property
    def trace_logger(self) -> logging.Logger:
        return logger.getChild(type(self).__name__).getChild("trace").getChild(os.path.basename(self.filename))


    @functools.cached_property
    def lines(self) -> set[TLineNo]:
        executable_lines = set(line for arc in self.arcs for line in arc if line >= 0)
        self.logger.debug(f"executable_lines: {executable_lines}")
        return executable_lines

    @functools.cached_property
    def excluded_lines(self) -> set[TLineNo]:
        excluded_lines: set[TLineNo] = set()

        for lineno, line in enumerate(self.source.split("\n"), 1):
            if self.exclude is not None and re.search(self.exclude, line):
                excluded_lines.add(lineno)

        return excluded_lines

    @property
    def bash_supports_case_item_trace(self) -> bool:
        # NOTE: bash's trace does not support showing case_items as of 5.2
        return False

    @property
    def bash_supports_case_item_termination_trace(self) -> bool:
        # NOTE: bash's trace does not support showing case_items as of 5.2
        return False

    @property
    def bash_supports_else_clause_trace(self) -> bool:
        # NOTE: bash's trace does not support showing else_clause as of 5.2
        return False

    def analyse_branches(self, tree: Tree) -> tuple[set[Branch], dict[Branch, set[Branch]]]:
        branches: set[Branch] = set()
        translations: dict[Branch, set[Branch]] = defaultdict(set)
        block_exits: dict[Node, set[Node]] = defaultdict(set)

        # TODO: stack
        # loop_context: list[tuple[Node, Node, list[Node]]] = []
        
        S = TypeVar("S", bound=Scope)
        def nearest_scope(scope: Scope, predicate: Callable[[Scope], TypeGuard[S]]) -> S | None:
            return next((ancestor for ancestor in scope.ancestors() if predicate(ancestor)), None)

        def parse_ast(node: Node, scope: Scope | None = None) -> None:
            if scope_type := scope_types.get(node.type):
                scope = scope_type(node, scope)

            assert scope

            if node.type == "if_statement":
                self.trace_logger.debug(f"if_statement: {node!r}")
                conditions = list(filter(of("command", "*_command", "compound_statement"), node.children_by_field_name("condition")))
                assert conditions, f"conditions are required as per the grammar: {node.children_by_field_name('condition')!r}"
                self.trace_logger.debug(f"conditions: {conditions}")

                elif_clauses = list(filter(of("elif_clause"), node.children))
                self.trace_logger.debug(f"elif_clauses: {elif_clauses}")
                else_clauses = list(filter(of("else_clause"), node.children))
                assert len(else_clauses) <= 1
                self.trace_logger.debug(f"else_clauses: {else_clauses}")
                clauses = {
                    node: list(filter(is_statement, dropuntil(of("then"), node.children))),
                    **{elif_clause: list(filter(is_statement, dropuntil(of("then"), elif_clause.children))) for elif_clause in elif_clauses},
                    **{else_clause: list(filter(is_statement, dropuntil(of("else"), else_clause.children))) for else_clause in else_clauses},
                }
                for clause, statements in clauses.items():
                    # logger.debug(f"clause: {clause}, statements: {statements}")
                    branch = (clause.start_point, statements[0].start_point)
                    branches.add(branch)
                    self.trace_logger.debug(f"(clause.start_point: {clause}, self.nearest_next_line(statements[0]: {statements[0]}.start_point)): {format_branch(branch)}, arc: {to_arc(branch)}")
                    # branches.add((clause.start_point[0] + 1, statements[0].start_point[0] + 1))

                    # if not is_break(statements[-1]) and not is_continue(statements[-1]):
                    #     branch = (statements[-1].end_point, node.end_point)
                    #     branches.add(branch)
                    #     self.trace_logger.debug(f"(statements[-1]: {statements[-1]}, node.end_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")

                for source, destination in pairwise(clauses.keys()):
                    # mismatch
                    branch = (source.start_point, destination.start_point)
                    # arc = (source.start_point[0] + 1, destination.start_point[0] + 1)
                    branches.add(branch)
                    self.trace_logger.debug(f"(source.start_point: {source}, destination.start_point: {destination}): {format_branch(branch)}, arc: {to_arc(branch)}")

                if else_clauses:
                    else_clause, = else_clauses
                    statements = clauses[else_clause]
                    if not self.bash_supports_else_clause_trace: # pragma: no branch
                        previous_clause = list(clauses.keys())[-2]
                        translations[(previous_clause.start_point, statements[0].start_point)] = {
                            (previous_clause.start_point, else_clause.start_point),
                            (else_clause.start_point, statements[0].start_point),
                        }

            elif node.type == "c_style_for_statement":
                self.trace_logger.debug(f"c_style_for_statement: {node!r}")
                body = node.child_by_field_name("body")
                assert body, "body is required as per the grammar"
                self.trace_logger.debug(f"body: {body}")

                initializers = node.children_by_field_name("initializer")
                self.trace_logger.debug(f"initializers: {initializers}")

                conditions = node.children_by_field_name("condition")
                self.trace_logger.debug(f"conditions: {conditions}")

                updates = node.children_by_field_name("update")
                self.trace_logger.debug(f"updates: {updates}")

                statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
                branch = (node.start_point, statements[0].start_point)
                branches.add(branch)
                self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")

                if updates:
                    branch = (statements[-1].end_point, updates[0].start_point)
                    branches.add(branch)
                    self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, updates[0].start_point: {updates[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
                    # loop_context.append((node, updates[0], statements))
                else:
                    branch = (statements[-1].end_point, node.start_point)
                    branches.add(branch)
                    self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
                    # loop_context.append((node, node, statements))

            elif node.type == "for_statement":
                self.trace_logger.debug(f"for_statement: {node!r}")

                variable = node.child_by_field_name("variable")
                self.trace_logger.debug(f"variable: {variable}")
                values = node.children_by_field_name("value")
                self.trace_logger.debug(f"values: {values}")

                body = node.child_by_field_name("body")
                assert body, "body is required as per the grammar"
                self.trace_logger.debug(f"body: {body}")

                statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
                assert statements, "at least one statement is required in for_statement as per the grammar"
                assert list(filter(not_of("comment"), statements)), "comments should not appear in statements"

                branch = (node.start_point, statements[0].start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                branch = (statements[-1].end_point, node.end_point)
                self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, node.end_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")

                # loop_context.append((node, node, statements))

            elif node.type == "while_statement":
                self.trace_logger.debug(f"while_statement: {node!r}")

                conditions = node.children_by_field_name("condition")
                assert conditions, "at least one condition is required for while_statement as per the grammar"
                self.trace_logger.debug(f"conditions: {conditions}")

                body = node.child_by_field_name("body")
                assert body, "body is required as per the grammar"
                self.trace_logger.debug(f"body: {body}")

                statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
                assert statements, "at least one statement is required in while_statement as per the grammar"

                branch = (node.start_point, statements[0].start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                branch = (statements[-1].end_point, conditions[0].start_point)
                self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, conditions[0].start_point: {conditions[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                # loop_context.append((node, conditions[0], statements))

            elif node.type == "case_statement":
                self.trace_logger.debug(f"case_statement: {node!r}")
                value = node.child_by_field_name("value")
                self.trace_logger.debug(f"value: {value}")
                case_items = list(filter(of("case_item"), node.children))
                self.trace_logger.debug(f"case_items: {case_items}")

                # TODO: find a way to be more precise on the branches for each condition in  each case_item

                # only bridge from case_statement to the first case_item
                branch = (node.start_point, case_items[0].start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, case_items[0].start_point: {case_items[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                for i, case_item in enumerate(case_items):
                    if statements := list(filter(is_statement, case_item.children)):
                        # matched
                        branch = (case_item.start_point, statements[0].start_point)
                        self.trace_logger.debug(f"(case_item.start_point: {case_item}, statements[0].start_point: {statements}): {format_branch(branch)}, arc: {to_arc(branch)}")
                        branches.add(branch)

                        if not self.bash_supports_case_item_trace: # pragma: no branch
                            # matched
                            branch = (node.start_point, statements[0].start_point)
                            self.trace_logger.debug(f"(node.start_point: {node}, statements[0].start_point: {statements[0]}): {format_branch(branch)}, arc: {to_arc(branch)}")
                            translations[branch] = {
                                # case_statement.start_point -> case_item[0] -> ... -> case_item
                                *((source.start_point, destination.start_point) for source, destination in pairwise([node] + case_items[:i])),
                                # *((source.start_point[0] + 1, destination.start_point[0] + 1) for source, destination in pairwise([node] + case_items[:i])),
                                # case_item -> statements[0]
                                (case_item.start_point, statements[0].start_point),
                                # (case_item.start_point[0] + 1, statements[0].start_point[0] + 1),
                            }

                        # termination
                        if termination := case_item.child_by_field_name("termination"):
                            branch = (termination.end_point, node.end_point)
                            self.trace_logger.debug(f"(termination.end_point: {termination}, node.end_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
                            branches.add(branch)

                            if not self.bash_supports_case_item_termination_trace: # pragma: no branch
                                # TODO: consider making fi, done, esac, etc. executable?
                                branch = (statements[-1].end_point, node.end_point)
                                self.trace_logger.debug(f"(statements[-1].end_point: {statements[-1]}, node.end_point: {node}, round=True): {format_branch(branch)}, arc: {to_arc(branch)}")

                                translations[branch] = {
                                    # statements[-1] -> case_statement.end_point
                                    (statements[-1].start_point, node.end_point),
                                    # (statements[-1].start_point[0] + 1, node.end_point[0] + 1),
                                    (node.end_point, node.end_point),
                                    # (node.end_point[0] + 1, self.nearest_next_line(node.end_point[0] + 1)),
                                }

                for source, destination in pairwise(case_items):
                    self.trace_logger.debug(f"source: {source}, source.children: {source.children}")
                    # TODO: consider dropping unilt ) and taking while not
                    if statements := list(filter(is_statement, source.children)):
                        self.trace_logger.debug(f"statements: {statements}")
                        # mismatch
                        branch = (source.start_point, destination.start_point)
                        self.trace_logger.debug(f"(source.start_point: {source}, destination.start_point: {destination}): {format_branch(branch)}, arc: {to_arc(branch)}")
                        branches.add(branch)

                    # fallthrough
                    if fallthrough := source.child_by_field_name("fallthrough"):
                        branch = (fallthrough.end_point, destination.start_point)
                        branches.add(branch)


                    assert fallthrough or source.child_by_field_name("termination"), f"case_item: {source!r} should have either fallthrough or termination"
            elif node.type == "negated_command":
                self.trace_logger.debug(f"negated_command: {node!r}")
                _, child = node.children
                self.trace_logger.debug(f"child: {child}")
                branch = (node.start_point, child.start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, child.start_point: {child}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

            elif node.type == "ternary_expression":
                self.trace_logger.debug(f"ternary_expression: {node!r}")
                condition = node.child_by_field_name("condition")
                assert condition, "condition is required as per the grammar"
                self.trace_logger.debug(f"condition: {condition}")

                consequence = node.child_by_field_name("consequence")
                assert consequence, "consequence is required as per the grammar"
                self.trace_logger.debug(f"consequence: {consequence}")

                branch = (node.start_point, consequence.start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, consequence.start_point: {consequence}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                alternative = node.child_by_field_name("alternative")
                assert alternative, "alternative is required as per the grammar"
                self.trace_logger.debug(f"alternative: {alternative}")

                branch = (node.start_point, alternative.start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, alternative.start_point: {alternative}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

            elif node.type == "binary_expression":
                self.trace_logger.debug(f"binary_expression: {node!r} {node.text!r}")

                left = node.child_by_field_name("left")
                assert left, "left is required for binary_expression as per the grammar"

                operator = node.child_by_field_name("operator")
                assert operator, "operator is required for binary_expression as per the grammar"

                right = node.child_by_field_name("right")
                assert right, "right is required for binary_expression as per the grammar"

                if operator.type in ("&&", "||"):
                    branch = (left.end_point, right.start_point)
                    self.trace_logger.debug(f"(left.end_point: {left} {left.text!r}, right.start_point: {right} {right.text!r}): {format_branch(branch)}, arc: {to_arc(branch)}")
                    branches.add(branch)

            elif node.type in "variable_assignment":
                self.trace_logger.debug(f"variable_assignment: {node!r}")

                name = node.child_by_field_name("name")
                assert name, "name is required for variable_assignment as per the grammar"

                value = node.child_by_field_name("value")
                assert value, "value is required for variable_assignment as per the grammar"

                self.trace_logger.debug(f"variable_assignment: name: {name} {name.text!r}, value: {value} {value.text!r}")

                branch = (node.start_point, scope.next(node).start_point)
                self.trace_logger.debug(f"(node.start_point: {node}, scope.next(node)).start_point: {scope.next(node).start_point}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                branch = (value.end_point, node.start_point)
                self.trace_logger.debug(f"(value.end_point: {value}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

                # TODO: do we need to handle previous -> node.end_point?

                branch = (value.end_point, scope.next(node).start_point)
                self.trace_logger.debug(f"(value.end_point: {value}, scope.next(node).start_point: {scope.next(node)}): {format_branch(branch)}, arc: {to_arc(branch)}")
                translations[branch] = {
                    (value.end_point, node.start_point),
                    (node.start_point, scope.next(node).start_point),
                }

                # self.trace_logger.debug(f"(node.end_point: {node}, node.parent.start_point: {node.parent}): {format_branch(branch)}, arc: {to_arc(branch)}")

            # elif node.type == "command_substitution":
            #     # TODO: should be the same in subshell?
            #     self.trace_logger.debug(f"command_substitution: {node!r}")

            #     # FIXME: this is pretty wrong
            #     branch = (node.end_point, cast(Node, node.parent).start_point)
            #     self.trace_logger.debug(f"(node.end_point: {node}, node.parent.start_point: {node.parent}): {format_branch(branch)}, arc: {to_arc(branch)}")
            #     branches.add(branch)

            elif node.type == "test_command":
                self.trace_logger.debug(f"test_command: {node!r}")
                # TODO

            elif node.type == "function_definition":
                self.trace_logger.debug(f"function_definition: {node!r}")
                branch = (entry.start_point, node.start_point)
                self.trace_logger.debug(f"(entry.start_point: {entry}, node.start_point: {node}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)
                branch = (node.end_point, exit.start_point)
                self.trace_logger.debug(f"(node.end_point, exit.start_point): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

            elif is_break(node):
                loop_scope = nearest_scope(scope, is_loop_scope)
                assert loop_scope, f"break can only be used inside a loop scope, scopes: {list(scope.ancestors())}"

                branch = (node.start_point, loop_scope.exit.end_point)
                self.trace_logger.debug(f"(node.start_point: {node}, loop_scope.exit.end_point: {loop_scope}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

            # elif is_break(node):
            #     assert loop_context, "break should only be used in loop context"
            #     loop, _, __ = loop_context[-1]
            #     branch = (node.start_point, loop.end_point)
            #     self.trace_logger.debug(f"(node.start_point: {node}, loop.end_point: {loop}, round=True): {format_branch(branch)}, arc: {to_arc(branch)}")
            #     branches.add(branch)

            elif is_continue(node):
                loop_scope = nearest_scope(scope, is_loop_scope)
                assert loop_scope, f"continue can only be used inside a loop scope, scopes: {list(scope.ancestors())}"

                branch = (node.start_point, loop_scope.advance.end_point)
                self.trace_logger.debug(f"(node.start_point: {node}, loop_scope.advance.end_point: {loop_scope}): {format_branch(branch)}, arc: {to_arc(branch)}")
                branches.add(branch)

            # elif is_continue(node):
            #     assert loop_context, "continue should only be used in loop context"
            #     _, update, __ = loop_context[-1]
            #     branch = (node.start_point, update.end_point)
            #     self.trace_logger.debug(f"(node.start_point: {node}, update.end_point: {update}, round=True): {format_branch(branch)}, arc: {to_arc(branch)}")
            #     branches.add(branch)

            elif node.type in set(EXECUTABLE_NODE_TYPES) - {"elif_clause", "else_clause"}:
                self.trace_logger.debug(f"{node.type}: {node!r}")
                branch = (node.end_point, scope.next(node).start_point)
                self.trace_logger.debug(f"(node.end_point: {node}, scope.next(node).start_point: {scope.next(node)})")
                branches.add(branch)
                # if scope.is_last_statement(node):
                #     branch = (node.end_point, scope.end.end_point)
                # else:
                #     sibling = next_statement(node)
                #     self.trace_logger.debug(f"node: {node}, text: {node.text!r}, next_statement(node): {sibling}") 
                #     branch = (node.end_point, sibling.end_point)
                #     self.trace_logger.debug(f"(node.end_point, sibling.end_point): {format_branch(branch)}, arc: {to_arc(branch)}")
                #     branches.add(branch)
            else:
                # self.trace_logger.debug(f"{node.type}: {node!r}")
                pass


            # if is_block(node):
            #     branch = (node.end_point, scope.next(node).start_point)
            #     self.trace_logger.debug("(node.end_point: {node}, scope.next(node).start_point: {next_statement(node)}: {to_arc(branch)}")
            #     branches.add(branch)

            for child in node.children:
                parse_ast(child, scope)

        predicate = lambda _: True # matches(of("program"), is_statement, match="any")
        self.trace_logger.debug("Tree:\n" + "\n".join(format_tree(tree.root_node, predicate=predicate, format=lambda node: repr(node) if node.type == "program" else f"{node!r} {node.text!s}")))
        parse_ast(tree.root_node, None)

        # extra_lines = set((lineno, (source, destination)) for source, destination in branches for lineno in (source, destination) if lineno not in self.lines)
        # self.trace_logger.debug(f"extra_lines: {extra_lines}")
        # assert not extra_lines, f"extra lines in branches: {extra_lines!r}"
        # self.trace_logger.debug(branches)

        branches.union(*translations.values())
        self.logger.debug(f"executable_branches: {pprint.pformat(set(map(format_branch, branches)))}")
        return branches, translations

    @functools.cached_property
    def arc_translations(self) -> dict[TArc, set[TArc]]:
        return {
            to_arc(branch): set(map(to_arc, translations))
            for branch, translations in self.branch_translations.items()
        }

    @functools.cached_property
    def arcs(self) -> set[TArc]:
        executable_arcs = set((to_line(source), to_line(destination)) for source, destination in self.branches)
        self.logger.debug(f"executable_arcs: {pprint.pformat(executable_arcs)}")
        return executable_arcs

    @functools.cached_property
    def exit_counts(self) -> dict[TLineNo, int]:
        lines: dict[TLineNo, set[int]] = defaultdict(set)

        for source, destination in self.arcs:
            lines[source].add(destination)

        return { source: len(destinations) for source, destinations in lines.items() }

    # def source_token_lines(self):
    #     pass

