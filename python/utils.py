from typing import Sequence, NoReturn

import collections
import jax
from jax._src.typing import DType


def _check_shape(
    array: jax.Array, expected_shape: Sequence[int], name: str
) -> NoReturn:
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} should have shape: {expected_shape}, but got {array.shape}."
        )


def _check_dtype(
    array: jax.Array, dtypes: DType | Sequence[DType], name: str
) -> NoReturn:
    if not isinstance(dtypes, collections.abc.Sequence):
        dtypes = (dtypes,)

    if all(array.dtype != d for d in dtypes):
        raise ValueError(f"{name} must be of type {dtypes}, but is {array.dtype}.")


def _check_is_multiple(x: int, y: int, name: str) -> NoReturn:
    if x % y != 0:
        raise ValueError(f"{name} must be divisible by {y}")


def round_multiple(x: int, m: int) -> int:
    """Rounds x to the nearest multiple of m.

    Args:
      x: The number to round.
      m: The multiple to round to.

    Returns:
      The rounded number.
    """
    return (x + m - 1) // m * m
