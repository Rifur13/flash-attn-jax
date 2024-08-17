import ctypes
from pathlib import Path
import jax.extend as jex
from typing import Sequence

from jax._src.typing import ArrayLike
import jax
import jax.numpy as jnp
import numpy as np

import flash_attn_jax_lib


def _check_shape(arr: ArrayLike, expected_shape: Sequence[int], name: str):
  if arr.shape != expected_shape:
    raise ValueError(f"{name} should have shape: {expected_shape}, but got {arr.shape}.")

def flash_attention_hopper_fwd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    *,
    softmax_scale: float,
    causal: bool):
  # (TODO pobudzey): check for sm90

  if not (query.ndim == 4 and key.ndim == 4 and value.ndim == 4):
    raise ValueError(f"query/key/value should be 4-dim, but are {query.ndim}/{key.ndim}/{value.ndim}")

  batch_size, seq_len_q, num_heads_q, head_dim = query.shape
  seq_len_kv, num_heads_kv = key.shape[1], key.shape[2]

  ###############
  # FlashAttention-3 supported configs.

  if query.dtype != jnp.float16:
    raise ValueError("Only float16 are supported right now. Please expand the FA3 custom call.")
  
  if head_dim != 128:
    raise ValueError("Only head_dims of 128 are supported right now. Please expand the FA3 custom call.")

  if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
    raise ValueError("Only sequence lenghts that are multiples of 128 are supported right now. Please expand the FA3 custom call.")

  if num_heads_q % 32 != 0 or num_heads_kv % 32 != 0:
    raise ValueError("Only num_heads that are multiples of 32 are supported right now. Please expand the FA3 custom call.")
  
  ###############

  _check_shape(key, (batch_size, seq_len_kv, num_heads_kv, head_dim), "key")
  _check_shape(value, (batch_size, seq_len_kv, num_heads_kv, head_dim), "value")
  
  if num_heads_q % num_heads_kv != 0:
    raise ValueError(f"The number of query heads must be a multiple of "
                     f"key/value heads, but got {num_heads_q} vs {num_heads_kv}")
  
  if not (query.dtype == key.dtype == value.dtype):
    raise ValueError(f"query/key/value should have the same shape, but got "
                     f"{query.dtype} vs {key.dtype} vs {value.dtype}.")
  

  tile_count_sempahore = jnp.zeros((1,), jnp.int32) if causal else jnp.empty((1,), jnp.int32) 

  out_type = [
      jax.ShapeDtypeStruct(query.shape, query.dtype), # out
      jax.ShapeDtypeStruct(query.shape[0:3], jnp.float32) # softmax_lse
  ]

  jex.ffi.register_ffi_target(
    "flash_attention_hopper_f16_fwd", flash_attn_jax_lib.flash_attention_hopper_f16_fwd(), platform="CUDA")
  
  out, softmax_lse =  jex.ffi.ffi_call(
      "flash_attention_hopper_f16_fwd",
      out_type,
      query,
      key,
      value,
      tile_count_sempahore,
      softmax_scale=np.float32(softmax_scale),
      is_causal=np.bool_(causal),        
      vectorized=False,
  )

  return [out, query, key, value, softmax_lse]


