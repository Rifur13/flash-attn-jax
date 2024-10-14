import jax
import jax.numpy as jnp

from einops import rearrange, repeat
import functools
import math


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
):
    row_idx = rearrange(jnp.arange(seqlen_q, dtype=jnp.int64), "s -> s 1")
    col_idx = jnp.arange(seqlen_k, dtype=jnp.int64)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = jnp.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return jnp.logical_or(
            col_idx > jnp.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


@functools.partial(jax.jit, static_argnames=["softmax_scale", "causal", "window_size"])
def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    softmax_scale=1.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32)
    if key_padding_mask is not None:
        scores = jnp.where(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"), scores
        )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
        )
        local_mask = jnp.broadcast_to(local_mask, scores.shape)
        scores = jnp.where(~local_mask, scores, float("-inf"))

    attention = jax.nn.softmax(scores * softmax_scale)

    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = jnp.where(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0, attention
        )

    output = jnp.einsum("bhts,bshd->bthd", attention, v)
    if query_padding_mask is not None:
        output = jnp.where(
            rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0, output
        )
    if key_padding_mask is not None:
        output = jnp.where(
            rearrange(
                jnp.logical_not(jnp.any(key_padding_mask, axis=1)), "b -> b 1 1 1"
            ),
            0.0,
            output,
        )
    return output.astype(dtype_og)
