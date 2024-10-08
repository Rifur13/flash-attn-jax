import jax
import jax.numpy as jnp
from einops import rearrange
from utils import _check_shape


def generate_random_padding_mask(batch_size: int, max_seq_len: int) -> jax.Array:
    """
    Generates a random padding mask for a batch of sequences.

    Args:
    batch_size: The number of sequences in the batch.
    max_seq_len: The maximum length of the sequences.

    Returns:
    A 2D boolean jax.Array of shape (batch_size, max_seq_len) where `True` indicates a valid token and `False` indicates padding.
    """
    key = jax.random.key(0)
    lengths = jax.random.randint(
        key, (batch_size, 1), max(1, max_seq_len - 20), max_seq_len + 1
    )
    grid = jnp.tile(jnp.arange(max_seq_len), batch_size).reshape(
        (batch_size, max_seq_len)
    )
    padding_mask = grid < lengths

    return padding_mask


def unpad_input(hidden_states: jax.Array, attention_mask: jax.Array):
    """Removes padding from a batch of sequences and returns relevant indexing information.

    Args:
      hidden_states: A jax.Array representing the hidden states of a batch of sequences.
      attention_mask: A 2D boolean jax.Array where `True` indicates a valid token and `False` indicates padding.

    Returns:
      A tuple containing:
        - unpadded_hidden_states: A jax.Array with the padding tokens removed.
        - indices: A jax.Array with the indices of valid tokens in the flattened input sequence.
        - cu_seqlens: A jax.Array with the cumulative sequence lengths.
        - max_seqlen_in_batch: An integer representing the maximum sequence length in the batch.
    """

    seqlens_in_batch = jnp.sum(attention_mask, axis=-1, dtype=jnp.int32)
    indices = jnp.nonzero(attention_mask.flatten())
    max_seqlen_in_batch = jnp.max(seqlens_in_batch).item()
    cu_seqlens = jnp.pad(jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), (1, 0))

    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch_size, seq_len):
    """
    Given a packed input and the indices at which there's a valid token, return the padded input.

    Args:
        hidden_states:  (total_tokens, ...), where total_tokens = number of valid tokens in a batch of sequence lengths.
        indices: The indices where the hidden states should be placed in the output.
        batch_size: The batch size.
        seq_len: The sequence length.

    Returns:
        The padded hidden states.
    """
    output = jnp.zeros(
        (batch_size * seq_len, *hidden_states.shape[1:]), dtype=hidden_states.dtype
    )
    output = output.at[indices].set(hidden_states)

    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


def generate_qkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    query_padding_mask: jax.Array,
    key_padding_mask: jax.Array,
):
    batch_size, seqlen_q, _, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    _check_shape(k, (batch_size, seqlen_k, nheads_k, d), "key")
    _check_shape(v, (batch_size, seqlen_k, nheads_k, d), "value")

    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)

    def output_pad_fn(output_unpad):
        return pad_input(output_unpad, indices_q, batch_size, seqlen_q)

    k_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
    v_unpad, _, _, _ = unpad_input(v, key_padding_mask)

    return (
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
    )
