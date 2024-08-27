import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jax.nn as nn

from jax._src.typing import DTypeLike
from flash_attention import flash_attention_hopper_fwd

@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("softmax_scale", [0.2, 0.5])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd(softmax_scale: float, causal: bool, dtype: DTypeLike):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    query = jax.random.uniform(k1, (2, 128, 32, 128), dtype)
    key = jax.random.uniform(k2, (2, 128, 32, 128), dtype)
    value = jax.random.uniform(k3, (2, 128, 32, 128), dtype)

    out, *_ = flash_attention_hopper_fwd(query, key, value, softmax_scale=softmax_scale, causal=causal)
    out_ref = nn.dot_product_attention(query, key, value, scale=softmax_scale, is_causal=causal)

    np.testing.assert_allclose(out, out_ref, rtol=5e-3)
