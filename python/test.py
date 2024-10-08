import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jax.nn as nn

from einops import rearrange

from jax._src.typing import DTypeLike
from flash_attention import (
    flash_attention_hopper_fwd,
    flash_attention_hopper_varlen_fwd,
)
from test_utils import (
    generate_random_padding_mask,
    unpad_input,
    pad_input,
    generate_qkv,
)

ABS_TOL = 5e-3
REL_TOL = 1e-1


def test_pad_input():
    batch_size = 9
    seq_len = 10

    x = jax.random.normal(
        jax.random.key(0), (batch_size, seq_len, 32, 128), jnp.float16
    )
    x_mask = generate_random_padding_mask(batch_size, seq_len)

    x_unpad, indices, cu_seqlens, max_seqlen = unpad_input(x, x_mask)
    out = pad_input(x_unpad, indices, batch_size, seq_len)

    x_masked = x * jnp.expand_dims(x_mask, (2, 3))

    np.testing.assert_allclose(x_masked, out)


@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 64),
        (113, 113),
        (128, 128),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_fwd(
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    mha_type: str,
    causal: bool,
    dtype: DTypeLike,
):
    batch_size = 9
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(k1, (batch_size, seqlen_q, nheads, head_dim), dtype)
    key = jax.random.normal(k2, (batch_size, seqlen_k, nheads_kv, head_dim), dtype)
    value = jax.random.normal(k3, (batch_size, seqlen_k, nheads_kv, head_dim), dtype)

    out, *_ = flash_attention_hopper_fwd(
        query, key, value, softmax_scale=None, causal=causal
    )
    out_ref = nn.dot_product_attention(query, key, value, scale=None, is_causal=causal)

    np.testing.assert_allclose(out, out_ref, atol=ABS_TOL, rtol=REL_TOL)


@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 64),
        (113, 113),
        (128, 128),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_fwd_varlen(
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    mha_type: str,
    causal: bool,
    dtype: DTypeLike,
):
    batch_size = 9
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(k1, (batch_size, seqlen_q, nheads, head_dim), dtype)
    k = jax.random.normal(k2, (batch_size, seqlen_k, nheads_kv, head_dim), dtype)
    v = jax.random.normal(k3, (batch_size, seqlen_k, nheads_kv, head_dim), dtype)

    query_padding_mask = generate_random_padding_mask(batch_size, seqlen_q)
    key_padding_mask = generate_random_padding_mask(batch_size, seqlen_k)

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    out_unpad, _ = flash_attention_hopper_varlen_fwd(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size_left=-1,
        window_size_right=-1,
        softmax_scale=None,
        causal=causal,
    )

    out = output_pad_fn(out_unpad)
    out_ref = nn.dot_product_attention(q, k, v, scale=None, is_causal=causal)

    q_zero_masking = jnp.logical_not(rearrange(query_padding_mask, "b s -> b s 1 1"))
    out_ref_masked = jnp.where(q_zero_masking, 0.0, out_ref)

    np.testing.assert_allclose(out, out_ref_masked, atol=ABS_TOL, rtol=REL_TOL)
