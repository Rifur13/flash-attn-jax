from typing import Sequence, NoReturn

import jax
from jax._src.typing import DType

def _check_shape(array: jax.Array, expected_shape: Sequence[int], name: str) -> NoReturn:
  if array.shape != expected_shape:
    raise ValueError(f"{name} should have shape: {expected_shape}, but got {array.shape}.")

def _check_dtype(array: jax.Array, dtypes: DType | Sequence[DType], name: str) -> NoReturn:
  if isinstance(dtypes, DType):
    dtypes = (dtypes, )

  if all(array.dtype != d for d in dtypes):
    raise ValueError(f"{name} must be of type {dtypes}, but is {array.dtype}.")
