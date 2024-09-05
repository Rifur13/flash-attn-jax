import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jax.nn as nn

from jax._src.typing import DTypeLike
from flash_attention import flash_attention_hopper_fwd


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
def test_fwd(seqlen_q: int, seqlen_k: int, head_dim: int, mha_type: str, causal: bool, dtype: DTypeLike):
    batch_size = 9
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    softmax_scale = 0.2

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(k1, (batch_size, seqlen_q, nheads, head_dim), dtype)
    key = jax.random.normal(k2, (batch_size, seqlen_k, nheads_kv, head_dim), dtype)
    value = jax.random.normal(k3, (batch_size, seqlen_k, nheads_kv, head_dim), dtype)

    out, *_ = flash_attention_hopper_fwd(query, key, value, softmax_scale=softmax_scale, causal=causal)
    out_ref = nn.dot_product_attention(query, key, value, scale=softmax_scale, is_causal=causal)

    np.testing.assert_allclose(out, out_ref, atol=5e-3, rtol=1e-1)
