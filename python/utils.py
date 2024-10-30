from typing import Sequence, NoReturn

import collections
import jax

from jax.typing import DTypeLike
import jax.numpy as jnp


def check_shape(array: jax.Array, expected_shape: Sequence[int], name: str) -> NoReturn:
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} should have shape: {expected_shape}, but got {array.shape}."
        )


def check_dtype(
    array: jax.Array, dtypes: DTypeLike | Sequence[DTypeLike], name: str
) -> NoReturn:
    if not isinstance(dtypes, collections.abc.Sequence):
        dtypes = (dtypes,)

    if jnp.dtype(array.dtype) not in map(jnp.dtype, dtypes):
        raise ValueError(f"{name} must be of type {dtypes}, but is {array.dtype}.")


def check_is_multiple(x: int, y: int, name: str) -> NoReturn:
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


def check_compute_capability(capability: str) -> bool:
    device = jax.local_devices(backend="gpu")[0]
    target = tuple(int(x) for x in capability.split("."))
    current = tuple(int(x) for x in device.compute_capability.split("."))

    if current < target:
        raise ValueError(
            f"Cuda device must have compute capability >= {target}, but is {current}."
        )
