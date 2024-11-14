from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from coverage_sh.searches import dijkstra

import pytest

@dataclass(frozen=True)
class Vec2:
    row: int
    column: int

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.row + other.row, self.column + other.column)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.row - other.row, self.column - other.column)

@dataclass(frozen=True)
class State:
    position: Vec2
    length: int = 0

    def __lt__(self, other: State) -> bool:
        return self.length < other.length

@pytest.mark.parametrize("maze,start,end,length", [
    ([[0]], Vec2(0, 0), Vec2(0, 0), 0),
    ([[1]], Vec2(0, 0), Vec2(0, 0), None),
    ([[0, 1], [0, 0]], Vec2(0, 0), Vec2(1, 1), 2),
    ([[0, 1], [1, 0]], Vec2(0, 0), Vec2(1, 1), None),
])
def test_dijkstra(maze: list[list[int]], start: Vec2, end: Vec2, length: Optional[int]) -> None:
    def is_safe(position: Vec2) -> bool:
        rows = len(maze)
        columns = len(maze[0])
        return 0 <= position.row < rows and 0 <= position.column < columns and not maze[position.row][position.column]

    def predicate(state: State) -> bool:
        return is_safe(state.position) and state.position == end

    def successors(state: State) -> list[State]:
        return [
            State(position, state.length + 1)
            for delta in [Vec2(0, 1), Vec2(1, 0), Vec2(0, -1), Vec2(-1, 0)]
            if is_safe(position := state.position + delta)
        ]

    expected = State(end, length) if length is not None else None

    actual = dijkstra(predicate, successors, [State(start)])

    assert actual == expected
