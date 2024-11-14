from __future__ import annotations

from typing import TypeVar, Generic, Union, Optional, Any, overload
from collections.abc import Callable, Iterable, Generator, Mapping

import sys
import itertools

if sys.version_info < (3, 10): # pragma: no-cover-if-python-gte-310
    from typing_extensions import TypeGuard
else: # pragma: no-cover-if-python-lt-310
    from typing import TypeGuard

T = TypeVar("T")
R = TypeVar("R")
@overload
def negate(predicate: Callable[[T], TypeGuard[R]]) -> Callable[[T], TypeGuard[R]]: ...
@overload
def negate(predicate: Callable[[T], bool]) -> Callable[[T], bool]: ...
def negate(predicate: Any) -> Any:
    def inner(value: T) -> bool | TypeGuard[R]:
        return not predicate(value)
    return inner

def instance_of(type: type[R]) -> Callable[[T], TypeGuard[R]]:
    def inner(value: T) -> TypeGuard[R]:
        return isinstance(value, type)
    return inner

@overload
def dropuntil(predicate: Callable[[T], TypeGuard[R]], iterable: Iterable[T]) -> itertools.dropwhile[R]: ...
@overload
def dropuntil(predicate: Callable[[T], bool], iterable: Iterable[T]) -> itertools.dropwhile[T]: ...
def dropuntil(predicate: Any, iterable: Iterable[T]) -> Any:
    iterable = itertools.dropwhile(negate(predicate), iterable)
    next(iterable, None)
    return iterable

def find_index(predicate: Callable[[T], bool], iterable: Iterable[T]) -> Optional[int]:
    return next((i for i, value in enumerate(iterable) if predicate(value)), None)

