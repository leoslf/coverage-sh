from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Hashable, Optional, Any, overload
from collections.abc import Callable, Hashable, Sequence

from sortedcontainers import SortedList

if TYPE_CHECKING:
    from sortedcontainers.sortedset import SupportsHashableAndRichComparison 
else:
    SupportsHashableAndRichComparison: TypeAlias = Any

S = TypeVar("S")
R = TypeVar("R", bound=Hashable)
P = TypeVar("P", bound=SupportsHashableAndRichComparison)

@overload
def dijkstra(
    predicate: Callable[[S], bool],
    successors: Callable[[S], Sequence[S]],
    initials: Sequence[S],
    priority: Callable[[S], P],
    representation: Callable[[S], R],
) -> Optional[S]: ...
@overload
def dijkstra(
    predicate: Callable[[R], bool],
    successors: Callable[[R], Sequence[R]],
    initials: Sequence[R],
    priority: Callable[[R], P],
    representation: Callable[[R], R] = lambda s: s,
) -> Optional[S | R]: ...
@overload
def dijkstra(
    predicate: Callable[[P], bool],
    successors: Callable[[P], Sequence[P]],
    initials: Sequence[P],
    priority: Callable[[P], P] = lambda s: s,
    representation: Callable[[P], P] = lambda s: s,
) -> Optional[S | R | P]: ...
def dijkstra(
    predicate: Callable[[S], bool],
    successors: Callable[[S], Sequence[S]],
    initials: Sequence[S],
    priority: Any = lambda s: s,
    representation: Any = lambda s: s,
) -> Any:
    seen = set()
    # priority queue
    queue = SortedList(initials, key=priority)

    while queue:
        # dequeue
        state = queue.pop(0)

        # mark visited
        seen.add(representation(state))

        if predicate(state):
            # found
            return state

        # add successors
        queue.update(filter(lambda successor: representation(successor) not in seen,  successors(state)))
    return None

