import jax.extend as jex

import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
import functools

from utils import (
    check_dtype,
    check_shape,
    check_is_multiple,
    check_compute_capability,
    round_multiple,
)

import flash_attn_jax_lib

jex.ffi.register_ffi_target(
    "flash_attention_hopper_f16_fwd",
    flash_attn_jax_lib.flash_attention_hopper_f16_fwd(),
    platform="CUDA",
)

jex.ffi.register_ffi_target(
    "flash_attention_hopper_f16_varlen_fwd",
    flash_attn_jax_lib.flash_attention_hopper_f16_varlen_fwd(),
    platform="CUDA",
)

jex.ffi.register_ffi_target(
    "flash_attention_hopper_f16_bwd",
    flash_attn_jax_lib.flash_attention_hopper_f16_bwd(),
    platform="CUDA",
)

jex.ffi.register_ffi_target(
    "flash_attention_hopper_f16_varlen_bwd",
    flash_attn_jax_lib.flash_attention_hopper_f16_varlen_bwd(),
    platform="CUDA",
)


def get_k_block_m(head_dim: int):
    if head_dim <= 64:
        return 128
    elif head_dim < 256:
        return 64
    else:
        return 32


@functools.partial(jax.custom_vjp, nondiff_argnums=range(3, 8))
@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "causal",
        "window_size_left",
        "window_size_right",
        "deterministic",
    ],
)
def flash_attention_hopper(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = True,
):
    check_compute_capability("9.0")

    out, _ = _flash_attention_hopper_fwd(
        query,
        key,
        value,
        window_size_left,
        window_size_right,
        softmax_scale,
        causal,
        deterministic,
    )

    return out


def _flash_attention_hopper_fwd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    window_size_left: int,
    window_size_right: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
):
    del window_size_left, window_size_right, deterministic

    if not (query.ndim == 4 and key.ndim == 4 and value.ndim == 4):
        raise ValueError(
            f"query/key/value should be 4-dim, but are {query.ndim}/{key.ndim}/{value.ndim}"
        )

    batch_size, seq_len_q, num_heads_q, head_dim = query.shape
    seq_len_kv, num_heads_kv = key.shape[1], key.shape[2]

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim)

    check_dtype(query, [jnp.float16], "query")
    check_dtype(key, query.dtype, "key")
    check_dtype(value, query.dtype, "value")

    check_shape(key, (batch_size, seq_len_kv, num_heads_kv, head_dim), "key")
    check_shape(value, (batch_size, seq_len_kv, num_heads_kv, head_dim), "value")

    if seq_len_q != seq_len_kv:
        raise ValueError(
            "Query and Key/Value sequence lengths must be equal, but got {seq_len_q} vs {seq_len_kv}."
        )

    if head_dim not in [64, 128, 256]:
        raise ValueError(
            f"head_dim must be one of [64, 128, 256] for the forward pass, but got {head_dim}."
        )

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"The number of query heads must be a multiple of "
            f"key/value heads, but got {num_heads_q} vs {num_heads_kv}"
        )

    tile_count_semaphore = (
        jnp.zeros((1,), jnp.int32) if causal else jnp.empty((1,), jnp.int32)
    )

    out_type = [
        jax.ShapeDtypeStruct(query.shape, query.dtype),  # out
        jax.ShapeDtypeStruct(
            [batch_size, num_heads_q, seq_len_q], jnp.float32
        ),  # softmax_lse
    ]

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


def _flash_attention_hopper_bwd(
    window_size_left: int,
    window_size_right: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
    res,
    dout: jax.Array,
):
    query, key, value, out, softmax_lse = res

    check_dtype(query, [jnp.float16], "query")
    check_dtype(key, query.dtype, "key")
    check_dtype(value, query.dtype, "value")
    check_dtype(dout, query.dtype, "dout")
    check_dtype(out, query.dtype, "out")

    if not (
        query.ndim == 4
        and key.ndim == 4
        and value.ndim == 4
        and dout.ndim == 4
        and out.ndim == 4
    ):
        raise ValueError(
            f"query/key/value/dout/out should be 4-dim, but are {query.ndim}/{key.ndim}/{value.ndim}/{dout.ndim}/{out.ndim}"
        )

    batch_size, seq_len_q, num_heads_q, head_dim = query.shape
    seq_len_kv, num_heads_kv = key.shape[1], key.shape[2]

    k_block_m = get_k_block_m(head_dim)
    seqlen_q_rounded = round_multiple(seq_len_q, k_block_m)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim)

    if head_dim not in [64, 128]:
        raise ValueError(
            f"head_dim must be one of [64, 128] for the backwards pass, but got {head_dim}."
        )

    if num_heads_q != num_heads_kv:
        raise NotImplementedError("MQA / GQA not implemented yet.")

    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"The number of query heads must be a multiple of "
            f"key/value heads, but got {num_heads_q} vs {num_heads_kv}"
        )

    check_is_multiple(seq_len_q, 128, "seq_len_q")
    check_is_multiple(seq_len_kv, 128, "seq_len_kv")

    check_shape(dout, (batch_size, seq_len_q, num_heads_q, head_dim), "dout")
    check_shape(out, (batch_size, seq_len_q, num_heads_q, head_dim), "out")
    check_shape(key, (batch_size, seq_len_kv, num_heads_kv, head_dim), "key")
    check_shape(value, (batch_size, seq_len_kv, num_heads_kv, head_dim), "value")

    dq_semaphore = jnp.zeros(
        ((seq_len_q + k_block_m - 1) // 64, batch_size, num_heads_q), jnp.int32
    )

    softmax_lse_log2 = jnp.empty(
        (batch_size, num_heads_q, seqlen_q_rounded), dtype=jnp.float32
    )
    dq_accum = jnp.empty(
        (batch_size, num_heads_q, seqlen_q_rounded, head_dim), dtype=jnp.float32
    )

    out_type = [
        jax.ShapeDtypeStruct(
            [batch_size, seq_len_q, num_heads_q, head_dim], query.dtype
        ),  # dq
        jax.ShapeDtypeStruct(
            [batch_size, seq_len_kv, num_heads_q, head_dim], key.dtype
        ),  # dk
        jax.ShapeDtypeStruct(
            [batch_size, seq_len_kv, num_heads_q, head_dim], value.dtype
        ),  # dv
        jax.ShapeDtypeStruct(
            [batch_size, num_heads_q, seqlen_q_rounded], jnp.float32
        ),  # softmax_d
    ]

    dq, dk, dv, _ = jex.ffi.ffi_call(
        "flash_attention_hopper_f16_bwd",
        out_type,
        dout,
        query,
        key,
        value,
        out,
        softmax_lse,
        softmax_lse_log2,
        dq_accum,
        dq_semaphore,
        window_size_left=np.int32(window_size_left),
        window_size_right=np.int32(window_size_right),
        softmax_scale=np.float32(softmax_scale),
        causal=np.bool_(causal),
        deterministic=np.bool_(deterministic),
        vectorized=False,
    )

    if num_heads_q != num_heads_kv:
        dk = jnp.sum(
            dk.reshape(
                batch_size,
                seq_len_kv,
                num_heads_kv,
                num_heads_q // num_heads_kv,
                head_dim,
            ),
            axis=3,
        )
        dv = jnp.sum(
            dv.reshape(
                batch_size,
                seq_len_kv,
                num_heads_kv,
                num_heads_q // num_heads_kv,
                head_dim,
            ),
            axis=3,
        )

    return [dq, dk, dv]


flash_attention_hopper.defvjp(_flash_attention_hopper_fwd, _flash_attention_hopper_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 12))
@functools.partial(
    jax.jit,
    static_argnames=[
        "max_seqlen_q",
        "max_seqlen_k",
        "window_size_left",
        "window_size_right",
        "softmax_scale",
        "causal",
        "deterministic",
    ],
)
def flash_attention_hopper_varlen(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    cu_seqlens_q: ArrayLike,
    cu_seqlens_k: ArrayLike,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = True,
):
    check_compute_capability("9.0")

    out, res = _flash_attention_hopper_varlen_fwd(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size_left,
        window_size_right,
        softmax_scale,
        causal,
        deterministic,
    )

    query, key, value, cu_seqlens_q, cu_seqlens_k, out, softmax_lse = res

    return out, softmax_lse


def _flash_attention_hopper_varlen_fwd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    cu_seqlens_q: ArrayLike,
    cu_seqlens_k: ArrayLike,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
):
    del deterministic

    if not (query.ndim == 3 and key.ndim == 3 and value.ndim == 3):
        raise ValueError(
            f"query/key/value should be 3-dim, but are {query.ndim}/{key.ndim}/{value.ndim}"
        )

    batch_size = cu_seqlens_q.size - 1
    total_q, num_heads, head_dim = query.shape
    total_k, num_heads_k, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim)

    check_dtype(query, [jnp.float16], "query")
    check_dtype(key, query.dtype, "key")
    check_dtype(value, query.dtype, "value")

    check_dtype(cu_seqlens_q, jnp.int32, "cu_seqlens_q")
    check_dtype(cu_seqlens_k, jnp.int32, "cu_seqlens_k")

    if head_dim not in [64, 128, 256]:
        raise ValueError(f"head_dim must be one of [64, 128, 256], but got {head_dim}.")

    if num_heads % num_heads_k != 0:
        raise ValueError(
            f"The number of query heads must be a multiple of "
            f"key/value heads, but got {num_heads} vs {num_heads_k}"
        )

    check_shape(query, (total_q, num_heads, head_dim), "query")
    check_shape(key, (total_k, num_heads_k, head_dim), "key")
    check_shape(value, (total_k, num_heads_k, head_dim), "value")

    check_shape(cu_seqlens_q, (batch_size + 1,), "cu_seqlens_q")
    check_shape(cu_seqlens_k, (batch_size + 1,), "cu_seqlens_k")

    out_type = [
        jax.ShapeDtypeStruct(query.shape, query.dtype),  # out
        jax.ShapeDtypeStruct([num_heads, total_q], jnp.float32),  # softmax_lse
    ]

    out, softmax_lse = jex.ffi.ffi_call(
        "flash_attention_hopper_f16_varlen_fwd",
        out_type,
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=np.int32(max_seqlen_q),
        max_seqlen_k=np.int32(max_seqlen_k),
        window_size_left=np.int32(window_size_left),
        window_size_right=np.int32(window_size_right),
        softmax_scale=np.float32(softmax_scale),
        is_causal=np.bool_(causal),
        vectorized=False,
    )

    return out, (query, key, value, cu_seqlens_q, cu_seqlens_k, out, softmax_lse)


def _flash_attention_hopper_varlen_bwd(
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
    res,
    dout: jax.Array,
):
    query, key, value, cu_seqlens_q, cu_seqlens_k, out, softmax_lse = res

    if not (
        query.ndim == 3
        and key.ndim == 3
        and value.ndim == 3
        and dout.ndim == 3
        and out.ndim == 3
    ):
        raise ValueError(
            f"query/key/value/dout/out should be 3-dim, but are {query.ndim}/{key.ndim}/{value.ndim}/{dout.ndim}/{out.ndim}"
        )

    batch_size = cu_seqlens_q.size - 1
    total_q, num_heads, head_dim = query.shape
    total_k, num_heads_k, _ = key.shape

    k_block_m = get_k_block_m(head_dim)
    total_q_padded_rounded = round_multiple(total_q + batch_size * 128, 128)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim)

    check_dtype(query, [jnp.float16], "query")
    check_dtype(key, query.dtype, "key")
    check_dtype(value, query.dtype, "value")
    check_dtype(dout, query.dtype, "dout")
    check_dtype(out, query.dtype, "out")

    check_dtype(cu_seqlens_q, jnp.int32, "cu_seqlens_q")
    check_dtype(cu_seqlens_k, jnp.int32, "cu_seqlens_k")

    if head_dim not in [64, 128]:
        raise ValueError(f"head_dim must be one of [64, 128], but got {head_dim}.")

    if num_heads != num_heads_k:
        raise ValueError(
            f"Only multi-headed attention is supported. Query heads must equal key/value heads, but got {num_heads} vs {num_heads_k}."
        )

    check_shape(query, (total_q, num_heads, head_dim), "query")
    check_shape(key, (total_k, num_heads_k, head_dim), "key")
    check_shape(value, (total_k, num_heads_k, head_dim), "value")

    check_shape(cu_seqlens_q, (batch_size + 1,), "cu_seqlens_q")
    check_shape(cu_seqlens_k, (batch_size + 1,), "cu_seqlens_k")

    dq_semaphore = jnp.zeros(
        ((max_seqlen_q + k_block_m - 1) // k_block_m, batch_size, num_heads), jnp.int32
    )

    softmax_lse_log2 = jnp.empty((num_heads, total_q_padded_rounded), dtype=jnp.float32)

    dq_accum = jnp.empty(
        (num_heads, total_q_padded_rounded, head_dim), dtype=jnp.float32
    )

    out_type = [
        jax.ShapeDtypeStruct([total_q, num_heads, head_dim], query.dtype),  # dq
        jax.ShapeDtypeStruct([total_k, num_heads, head_dim], key.dtype),  # dk
        jax.ShapeDtypeStruct([total_k, num_heads, head_dim], value.dtype),  # dv
        jax.ShapeDtypeStruct(
            [num_heads, total_q_padded_rounded], jnp.float32
        ),  # softmax_d
    ]

    dq, dk, dv, _ = jex.ffi.ffi_call(
        "flash_attention_hopper_f16_varlen_bwd",
        out_type,
        dout,
        query,
        key,
        value,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_lse,
        softmax_lse_log2,
        dq_accum,
        dq_semaphore,
        max_seqlen_q=np.int32(max_seqlen_q),
        max_seqlen_k=np.int32(max_seqlen_k),
        window_size_left=np.int32(window_size_left),
        window_size_right=np.int32(window_size_right),
        softmax_scale=np.float32(softmax_scale),
        causal=np.bool_(causal),
        deterministic=np.bool_(deterministic),
        vectorized=False,
    )

    return dq, dk, dv, None, None


flash_attention_hopper_varlen.defvjp(
    _flash_attention_hopper_varlen_fwd, _flash_attention_hopper_varlen_bwd
)
