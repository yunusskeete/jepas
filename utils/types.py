from typing import Tuple, TypeVar, Union

Number = Union[int, float]

T = TypeVar("T")  # Generic type variable


def ensure_tuple(value: Union[T, Tuple[T, T]]) -> Tuple[T, T]:
    return value if isinstance(value, tuple) else (value, value)
