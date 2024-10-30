import jax
import jax.numpy as jnp
import numpy as np
import pytest
import functools

from flash_attention_reference import attention_ref

from jax._src.typing import DTypeLike
from flash_attention import flash_attention_hopper_varlen, flash_attention_hopper
from test_utils import (
    generate_random_padding_mask,
    unpad_input,
    pad_input,
    generate_qkv,
)


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

    impl = functools.partial(flash_attention_hopper, softmax_scale=1.0, causal=causal)
    out = impl(query, key, value)
    out_ref = attention_ref(query, key, value, softmax_scale=1.0, causal=causal)

    np.testing.assert_allclose(out, out_ref, atol=5e-3, rtol=1e-1)


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
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    out_unpad, _ = flash_attention_hopper_varlen(
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
    out_ref = attention_ref(
        q, k, v, query_padding_mask, key_padding_mask, softmax_scale=None, causal=causal
    )

    np.testing.assert_allclose(out, out_ref, atol=5e-3, rtol=1e-1)


@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_bwd(
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

    def f(q, k, v):
        impl = functools.partial(
            flash_attention_hopper, softmax_scale=None, causal=causal
        )
        return impl(q, k, v).sum()

    def f_ref(q, k, v):
        impl = functools.partial(
            attention_ref,
            query_padding_mask=None,
            key_padding_mask=None,
            softmax_scale=None,
            causal=causal,
        )
        return impl(q, k, v).sum()

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(query, key, value)
    dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(query, key, value)

    np.testing.assert_allclose(dq, dq_ref, atol=1e-1)
    np.testing.assert_allclose(dk, dk_ref, atol=1e-1)
    np.testing.assert_allclose(dv, dv_ref, atol=1e-1)


@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("deterministic", [True, False])
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
def test_bwd_varlen(
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    mha_type: str,
    causal: bool,
    deterministic: bool,
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
        _,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    def f(q, k, v, cu_seqlens_q, cu_seqlens_k):
        impl = functools.partial(
            flash_attention_hopper_varlen,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            window_size_left=-1,
            window_size_right=-1,
            softmax_scale=None,
            causal=causal,
            deterministic=deterministic,
        )
        return impl(q, k, v, cu_seqlens_q, cu_seqlens_k).sum()

    def f_ref(q, k, v):
        impl = functools.partial(
            attention_ref,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            softmax_scale=None,
            causal=causal,
        )
        return impl(q, k, v).sum()

    dq_unpad, dk_unpad, dv_unpad = jax.grad(f, argnums=(0, 1, 2))(
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k
    )
    dq = dq_pad_fn(dq_unpad)
    dk = dk_pad_fn(dk_unpad)
    dv = dk_pad_fn(dv_unpad)

    dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)

    np.testing.assert_allclose(dq, dq_ref, atol=1e-1)
    np.testing.assert_allclose(dk, dk_ref, atol=1e-1)
    np.testing.assert_allclose(dv, dv_ref, atol=1e-1)
