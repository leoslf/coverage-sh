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
