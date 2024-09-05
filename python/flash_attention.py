import jax.extend as jex
from typing import Sequence

from jax._src.typing import DType
import jax
import jax.numpy as jnp
import numpy as np

import flash_attn_jax_lib


def _check_shape(array: jax.Array, expected_shape: Sequence[int], name: str):
  if array.shape != expected_shape:
    raise ValueError(f"{name} should have shape: {expected_shape}, but got {array.shape}.")

def _check_dtype(array: jax.Array, dtypes: DType | Sequence[DType], name: str):
  if isinstance(dtypes, DType):
    dtypes = (dtypes, )

  if all(array.dtype != d for d in dtypes):
    raise ValueError(f"{name} must be of type {dtypes}, but is {array.dtype}.")

def flash_attention_hopper_fwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    softmax_scale: float,
    causal: bool):
  # (TODO pobudzey): check for sm90

  if not (query.ndim == 4 and key.ndim == 4 and value.ndim == 4):
    raise ValueError(f"query/key/value should be 4-dim, but are {query.ndim}/{key.ndim}/{value.ndim}")

  batch_size, seq_len_q, num_heads_q, head_dim = query.shape
  seq_len_kv, num_heads_kv = key.shape[1], key.shape[2]

  _check_dtype(query, [jnp.float16], "query")
  _check_dtype(key, query.dtype, "key")
  _check_dtype(value, query.dtype, "value")

  _check_shape(key, (batch_size, seq_len_kv, num_heads_kv, head_dim), "key")
  _check_shape(value, (batch_size, seq_len_kv, num_heads_kv, head_dim), "value")

  if seq_len_q != seq_len_kv:
    raise ValueError("Query and Key/Value sequence lengths must be equal, but got {seq_len_q} vs {seq_len_kv}.")

  if head_dim not in [64, 128, 256]:
    raise ValueError(f"head_dim must be one of [64, 128, 256], but got {head_dim}.")

  if num_heads_q % num_heads_kv != 0:
    raise ValueError(f"The number of query heads must be a multiple of "
                     f"key/value heads, but got {num_heads_q} vs {num_heads_kv}")

  tile_count_semaphore = jnp.zeros((1,), jnp.int32) if causal else jnp.empty((1,), jnp.int32)

  out_type = [
      jax.ShapeDtypeStruct(query.shape, query.dtype), # out
      jax.ShapeDtypeStruct([batch_size, num_heads_q, seq_len_q], jnp.float32) # softmax_lse
  ]

  jex.ffi.register_ffi_target(
    "flash_attention_hopper_f16_fwd", flash_attn_jax_lib.flash_attention_hopper_f16_fwd(), platform="CUDA")

  out, softmax_lse = jex.ffi.ffi_call(
      "flash_attention_hopper_f16_fwd",
      out_type,
      query,
      key,
      value,
      tile_count_semaphore,
      softmax_scale=np.float32(softmax_scale),
      is_causal=np.bool_(causal),
      vectorized=False,
  )

  return out, (query, key, value, out, softmax_lse)
