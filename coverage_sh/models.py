from __future__ import annotations

import sys
import os
import inspect
import ctypes
import itertools
import functools
import re
import fnmatch
import pathlib
import shutil
import shlex
import subprocess
import logging

import abc
import pprint

import more_itertools
from sortedcontainers import SortedSet

from typing import TypeVar, ClassVar, Union, Any, Literal, Optional, overload, Protocol, runtime_checkable
if sys.version_info < (3, 10): # pragma: no-cover-if-python-gte-310
    from typing_extensions import TypeAlias, TypeGuard, Self
else: # pragma: no-cover-if-python-lt-310
    from typing import TypeAlias, TypeGuard, Self

from collections.abc import Callable, Mapping, Iterable, Generator, Collection, Sequence, Set
from collections import defaultdict

from dataclasses import dataclass, field

from tree_sitter import Tree, Node

from coverage.types import TConfigurable, TArc, TLineNo # , CodeRegion
from coverage_sh.utils import negate, dropuntil, instance_of, find_index

logger = logging.getLogger(__name__)

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
}

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S", bound="Scope")

# tree_sitter < 0.22
Point: TypeAlias = tuple[int, int]
Branch: TypeAlias = tuple[Point, Point]

MatchMode = Literal["all", "any"]

def of(*types: str, match: MatchMode = "any") -> Callable[[Node], bool]:
    return lambda node: {"all": all, "any": any}[match](fnmatch.fnmatch(node.type, type) for type in types)

def not_of(*types: str, match: Literal["all", "any"] = "any") -> Callable[[Node], bool]:
    negations: dict[Literal["all", "any"], Literal["all", "any"]] = {"all": "any", "any": "all"}
    return negate(of(*types, match=negations[match]))

def matches(*predicates: Callable[[T], bool], match: MatchMode) -> Callable[[T], bool]:
    return lambda node: {"all": all, "any": any}[match](predicate(node) for predicate in predicates)

def equals(target: T) -> Callable[[T], bool]:
    def inner(value: T) -> bool:
        logger.debug(f"value: {value} == target: {target}, result: {value == target}")
        return value == target
    return inner

def contains(target: Node, node: Node, level: int = 0) -> bool:
    # indent = '\t' * level
    # logger.debug(f"{indent}contains(target: {target}, node: {node})")
    if target in node.children:
        return True

    for child in node.children:
        if contains(target, child, level=level + 1):
            return True

    return False

is_statement = of(*EXECUTABLE_NODE_TYPES)
is_block = of("if_statement", "*for_statement", "while_statement", "case_statement")
is_loop = of("*for_statement", "while_statement")

@runtime_checkable
class Range(Protocol):
    @property
    def start_point(self) -> Point: ...

    @property
    def end_point(self) -> Point: ...

@dataclass(frozen=True)
class MadeupRange:
    start_point: Point
    end_point: Point

entry: Range = MadeupRange((-1, 0), (-1, 0))
exit: Range = MadeupRange((-1, 0), (-1, 0))

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

def ensure_sint32(n: int) -> int:
    return ctypes.c_int32(n & ctypes.c_uint32(-1).value).value

def to_line(point: Point, offset: int = 0) -> TLineNo:
    if (row := ensure_sint32(point[0])) < 0:
        return row
    return int(row) + 1 + offset

def node_to_line(node: Node) -> int:
    return to_line(node.start_point)

def to_arc(branch: Branch) -> TArc:
    source, destination = branch
    return (to_line(source), to_line(destination))

def format_point(point: Point) -> tuple[TLineNo, int]:
    _, column = point
    return (to_line(point), column + 1)

def format_branch(branch: Branch, offset: int = 0) -> tuple[tuple[TLineNo, int], tuple[TLineNo, int]]:
    source, destination = branch
    return (format_point(source), format_point(destination))

def is_command(name: str) -> Callable[[Node], bool]:
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
        assert command.text is not None
        return command.text.decode() == name
    return inner

is_break = is_command("break")
is_continue = is_command("continue")
is_return = is_command("return")

@dataclass(frozen=True)
class Scope(metaclass=abc.ABCMeta):
    type: ClassVar[str]
    node: Node
    parent: Optional[Scope]

    def ancestors(self) -> Generator[Scope]:
        yield self
        if self.parent:
            yield from self.parent.ancestors()

    @overload
    def nearest(self, predicate: Callable[[Scope], TypeGuard[S]]) -> Optional[S]: ...
    @overload
    def nearest(self, predicate: Callable[[Scope], bool]) -> Optional[Scope]: ...
    def nearest(self, predicate: Callable[[Scope], Union[bool, TypeGuard[S]]]) -> Optional[Union[Scope, S]]:
        logger.warning(f"nearest: {self}")
        if predicate(self):
            return self
        if self.parent:
            return self.parent.nearest(predicate)
        return None

    @property
    def entry(self) -> Node | Range:
        return self.node.children[0]

    @property
    def exit(self) -> Node | Range:
        return self.node.children[-1]

    # @functools.cached_property
    @property
    def statements(self) -> list[Node]:
        statements = list(filter(is_statement, self.node.children))
        comments = list(filter(of("comment"), statements))
        assert not comments, f"statements: {statements}, comments: {comments}"
        logger.debug(f"{type(self).__name__}: {statements}")
        return statements

    def is_first_statement(self, statement: Node) -> bool:
        return statement == self.statements[0]

    def is_last_statement(self, statement: Node) -> bool:
        return statement == self.statements[-1]

    @property
    def end(self) -> Node | Range:
        """ The place where the control flow goes after executing the last statement. """
        return self.exit

    # TODO: cache?
    def next(self, statement: Node) -> Node | Range:
        index = find_index(equals(statement), self.statements)
        assert index is not None, f"cannot find {statement} (text: {statement.text!r}) in self.statements: {self.statements}, self: {self}"

        if index + 1 >= len(self.statements):
            return self.end
        return self.statements[index + 1]

    def __contains__(self, node: Node) -> bool:
        return contains(node, self.node)

@dataclass(frozen=True)
class ProgramScope(Scope):
    type: ClassVar[str] = "program"

    @property
    def entry(self) -> Node | Range:
        return entry

    @property
    def exit(self) -> Node | Range:
        return exit

@dataclass(frozen=True)
class FunctionScope(Scope):
    type: ClassVar[str] = "function_definition"

def conditions(node: Node) -> list[Node]:
    if node.type == "if_statement":
        return node.children_by_field_name("condition")[:-1]
    if node.type == "elif_clause":
        logger.debug(f"node.children: {node.children}")
        # after elif but before ; and then
        elif_keyword = find_index(of("elif"), node.children)
        semicolon = find_index(of(";"), node.children)
        then = find_index(of("then"), node.children)
        assert elif_keyword is not None
        assert semicolon
        assert then
        assert elif_keyword == 0
        assert semicolon + 1 == then
        return node.children[elif_keyword + 1:semicolon]

    return []

def statements(node: Node) -> list[Node]:
    if node.type in ("if_statement", "elif_clause"):
        return list(filter(not_of("elif_clause", "else_clause"), filter(is_statement, dropuntil(of("then"), node.children))))
    if node.type == "else_clause":
        return list(filter(is_statement, node.children))
    return []

@dataclass(frozen=True)
class IfScope(Scope):
    type: ClassVar[str] = "if_statement"

    @functools.cached_property
    def clauses(self) -> list[Node]:
        return [self.node] + list(filter(of("elif_clause", "else_clause"), self.node.children))

    @functools.cached_property
    def conditions(self) -> dict[Node, list[Node]]:
        return { clause: conditions(clause) for clause in self.clauses }

    @functools.cached_property
    def condition_scopes(self) -> list[ConditionScope]:
        return [
            ConditionScope(
                node=clause,
                parent=self,
                conditions=conditions,
            )
            for clause, conditions in self.conditions.items()
        ]

    @functools.cached_property
    def clause_scopes(self) -> list[ClauseScope]:
        return [
            ClauseScope(
                node=clause,
                parent=self,
            )
            for clause in self.clauses
        ]

    def __contains__(self, target: Node) -> bool:
        # TODO: refactor to use condition_scopes
        for clause, conditions in self.conditions.items():
            for condition in conditions:
                if target == condition or contains(target, condition):
                    return True
        for clause_scope in self.clause_scopes:
            if target in clause_scope:
                return True
        return super().__contains__(target)

    def next(self, statement: Node) -> Node | Range:
        for condition_scope in self.condition_scopes:
            if statement in condition_scope:
                logger.debug(f"found condition_scope: {condition_scope} for {statement}")
                return condition_scope.next(statement)

        for clause_scope in self.clause_scopes:
            # logger.error(f"statement: {statement} in clause_scope: {clause_scope}: {statement in clause_scope}")
            if statement in clause_scope:
                return clause_scope.next(statement)

        return super().next(statement)

@dataclass(frozen=True)
class ConditionScope(Scope):
    """ Prevent returning or exiting if condition fails """
    conditions: Sequence[Node]
    type: ClassVar[str] = "<virtual>"

    def __contains__(self, target: Node) -> bool:
        return target in self.conditions

    def next(self, condition: Node) -> Node | Range:
        assert self.parent
        assert condition in self
        if condition == self.conditions[-1]:
            return self.node.children[0]
        
        return super().next(condition)

@dataclass(frozen=True)
class ClauseScope(Scope):
    type: ClassVar[str] = "<virtual>"

    def __contains__(self, target: Node) -> bool:
        logger.info(f"{target} in {self.statements}: {target in self.statements}")
        return target in self.statements

    @property
    def statements(self) -> list[Node]:
        results = statements(self.node)
        # logger.error(f"statements: {results}, self: {self}")
        return results

    def next(self, statement: Node) -> Node | Range:
        assert self.parent
        assert statement in self
        index = find_index(equals(statement), self.statements)
        assert index is not None
        if index + 1 >= len(self.statements):
            return self.parent.end
        return self.statements[index + 1]

@dataclass(frozen=True)
class LoopScope(Scope, metaclass=abc.ABCMeta):
    @functools.cached_property
    def statements(self) -> list[Node]:
        body = self.node.child_by_field_name("body")
        assert body, "body is required as per the grammar"
        logger.debug(f"body: {body}")

        statements = list(filter(is_statement, dropuntil(of("do"), body.children)))
        assert statements, "at least one statement is required in while_statement as per the grammar"
        return statements

    @property
    @abc.abstractmethod
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

scope_types: Mapping[str, builtins.type[Scope]] = {
    cls.type: cls for cls in subclasses(Scope) if not inspect.isabstract(cls) and hasattr(cls, "type")
}

@dataclass(frozen=True)
class Shebang:
    """ Shebang
    The informative section of the `POSIX specification for sh: Application Usage <https://pubs.opengroup.org/onlinepubs/9699919799/utilities/sh.html#tag_20_117_16>`_ states that you cannot rely on the sh executable being installed at /bin/sh
    """
    # NOTE: POSIX doesn't specify how exactly the shebang is to be interpreted
    interpreter: str
    argv: Sequence[str]

    @classmethod
    def parse(cls, source: str) -> Optional[Self]:
        """
        >>> Shebang("#!/usr/bin/env bash\n") # POSIX Line
        Shebang("/usr/bin/env", ("bash",))
        >>> Shebang("#!/usr/bin/env bash")
        Shebang("/usr/bin/env", ("bash",))
        >>> Shebang("#!/bin/sh")
        Shebang("/bin/sh", ())
        >>> Shebang("#! /bin/sh") # should work with space
        Shebang("/bin/sh", ())
        >>> Shebang("#! /bin/sh") # should work with space
        Shebang("/bin/sh", ())
        """
        # shebang MUST start at the very beginning of the file
        if match := re.match(r"^#!(?P<command>[^\n]*)[\n]?", source):
            interpreter, *argv = shlex.split(match["command"])
            return cls(
                interpreter,
                argv,
            )
        return None

    def expand(self, env: Mapping[str, str] = {}) -> Self:
        if pathlib.Path(self.interpreter).name == "env":
            flags, arguments = more_itertools.before_and_after(lambda argument: argument == "-i", self.argv)
            variable_assignments, arguments = more_itertools.before_and_after(lambda argument: "=" in argument, arguments)
            interpreter, *argv = arguments

            command = shutil.which("command", path=os.confstr("CS_PATH"))
            assert command is not None, "command should always be found in POSIX-compliant platforms"

            interpreter_path = subprocess.check_output(
                # ensure we are using the system version of command
                [command, "-v", interpreter],
                env={
                    **({} if "-i" in flags else env),
                    # NOTE: there is no shell expansion when the shebang is evaluated in the kernel
                    **{ name: value for variable_assignment in variable_assignments for (name, value) in [variable_assignment.split("=", maxsplit=1)] },
                },
                text=True,
            ).rstrip()
            return type(self)(
                interpreter_path,
                argv=argv,
            )
        return self

    @classmethod
    def infer_language(
        cls,
        shebang: Optional[Shebang],
    ) -> Optional[str]:
        if shebang is None:
            return None

        return pathlib.Path(shebang.expand().interpreter).name

@dataclass(frozen=True)
class Symbols:
    node: Node
    captures: dict[str, SortedSet[Node]] = field(compare=False)

    @functools.cached_property
    def children(self) -> dict[str, SortedSet[Node]]:
        nodes_by_type: dict[str, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
        for child in self.node.children:
            nodes_by_type[child.type].add(child)
        return nodes_by_type

    @functools.cached_property
    def nodes_by_type(self) -> dict[str, SortedSet[Node]]:
        nodes_by_type: dict[str, SortedSet[Node]] = defaultdict(lambda: SortedSet(key=node_to_line))
        for node in itertools.chain.from_iterable(self.captures.values()):
            nodes_by_type[node.type].add(node)
        return nodes_by_type

    @functools.cached_property
    def captures_by_node(self) -> dict[Node, dict[str, SortedSet[Node]]]:
        captures_by_node: dict[Node, dict[str, SortedSet[Node]]] = defaultdict(lambda: defaultdict(lambda: SortedSet(key=node_to_line)))
        for capture_name, nodes in self.captures.items():
            for node in nodes:
                if node.parent:
                    # logger.debug(f"node.parent: {node.parent!r}, capture_name: {capture_name}, node: {node}")
                    captures_by_node[node.parent][capture_name].add(node)

        return captures_by_node

    def __lt__(self, other: Symbols) -> bool:
        return node_to_line(self.node) < node_to_line(other.node)

    @overload
    def __getitem__(self, key: Node) -> Symbols: ...
    @overload
    def __getitem__(self, key: str) -> SortedSet[Symbols]: ...
    @overload
    def __getitem__(self, key: tuple[str, int]) -> Symbols: ...
    @overload
    def __getitem__(self, key: Sequence[tuple[str, int]]) -> Symbols: ...
    def __getitem__(self, key: Any) -> Any:
        # logger.info(f"key: {key!r}")
        # logger.info(f"key: {key!r}, self.node: {self.node!r}, self.captures: {self.captures}, self.captures_by_node: {self.captures_by_node}")
        if isinstance(key, Node):
            # logger.info(f"key: {key!r}, self.captures_by_node[key]: {self.captures_by_node[key]}")
            return Symbols(key, self.captures)

        if isinstance(key, str):
            if key == "parent":
                assert self.node.parent
                return SortedSet([self[self.node.parent]])

            # logger.info(f"self.node: {self.node!r}, self.captures_by_node[self.node]: {self.captures_by_node[self.node]}")
            # logger.info(f"self.node: {self.node!r}, self.nodes_by_type: {self.nodes_by_type}")
            nodes = [self[node] for node in (self.captures_by_node[self.node].get(key) or self.captures.get(key) or self.nodes_by_type[key])]
            # logger.info(f"key: {key!r}, nodes: {nodes}")
            results = SortedSet(nodes)
            # logger.info(f"results: {results}, self.captures: {self.captures}, self.nodes_by_type: {self.nodes_by_type}")
            return results

        capture_name: str
        index: int
        if isinstance(key, Sequence):
            if len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], int):
                capture_name, index = key
                return self[capture_name][index]

            if len(key) == 0:
                return self

            return functools.reduce(lambda symbols, key: symbols[key], key, self)

@dataclass(frozen=True)
class ShellFile:
    path: pathlib.Path
    language: str
    shebang: Optional[Shebang]
    source: str
    tree: Tree
    captures: dict[str, list[Node]]
    lines: set[TLineNo]
    excluded_lines: set[TLineNo]
    branches: set[Branch]
    branch_translations: dict[Branch, list[Branch]]

    @functools.cached_property
    def symbols(self) -> Symbols:
        return Symbols(self.tree.root_node, {
            capture_name: SortedSet(captures, key=node_to_line) for capture_name, captures in self.captures.items()
        })

    @functools.cached_property
    def arcs(self) -> set[TArc]:
        executable_arcs = set(map(to_arc, self.branches))
        logger.debug(f"[{self.path.name}] executable_arcs: {pprint.pformat(executable_arcs)}")
        return executable_arcs

    @functools.cached_property
    def arc_translations(self) -> dict[TArc, set[TArc]]:
        return {
            to_arc(branch): set(map(to_arc, translations))
            for branch, translations in self.branch_translations.items()
        }

    # @functools.cached_property
    # def lines(self) -> set[TLineNo]:
    #     return set(line for arc in self.arcs for line in arc if line >= 0)

    @functools.cached_property
    def exit_counts(self) -> dict[TLineNo, int]:
        lines: dict[TLineNo, set[int]] = defaultdict(set)

        for source, destination in self.arcs:
            lines[source].add(destination)

        return { source: len(destinations) for source, destinations in lines.items() }

    # def missing_arc_description(self, start: int, end: int, executed_arcs: Iterable[TArc] | None) -> str: ...

    # def arc_description(self, start: int, end: int) -> str: ...

    # @functools.cached_property
    # def source_token_lines(self) -> Iterable[list[tuple[TokenClass, str]]]: ...

    # def code_regions(self) -> Iterable[CodeRegion]: ...

    # def code_region_kinds(self) -> Iterable[tuple[str, str]]: ...
