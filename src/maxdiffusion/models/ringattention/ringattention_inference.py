"""
Dot product attention
"""

from functools import partial
import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange


def _ring_attention_inference_fwd(q, k, v, attn_mask, axis_name, float32_logits: bool):
    """
    To be executed under shard map, each device computes full attention for its query shard. 
    q: Query matrix shape (batch, q_seq_len, num_heads, head_dim)
    k: Key matric shape (batch, kv_seq_len, num_heads, head_dim)
    v
    attn_mask
    axis_name
    float32_logits: bool if true cast q k to float 32
    """

    if float32_logits: # cast q k to float 32
        q, k = q.astype(jnp.float32), k.astype(jnp.float32) 

    
    batch, q_len, num_heads, _ = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    # e^(QK^t-1/srt(d_head)) * V
    # softmax numerator(e^(score−max_score)) * V 
    # Replicated on each device
    # Local shape for query shard
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    # sigma e^(QK^t-1/srt(d_head))
    # sigma(softmax numerator)
    # Local shape for query shard
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    # all reduce sum to get all devices (var 1 , summed over parallelization degree)
    axis_size = lax.psum(1, axis_name)
    # element wise sqrt over head_dim
    scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        # unpack from comms sent from previous device in the axis.
        # max score from score head, q_seq_len, k_seq_len for single key seq shard
        # numerator
        prev_max_score, numerator, denominator, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        # QK^t/scale for current K sequence shard
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        
        # apply mask?
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        
        # New max score
        # compare previous K shard's max score for each query with
        # Max score from attention scores with current K shard for each query
        # batch, num_heads, q_len
        max_score = jnp.maximum(prev_max_score, jnp.max(attn_weights, axis=-1))
        # softmax score numerator
        exp_weights = jnp.exp(attn_weights - max_score[..., None])
        
        # correction to be multiplied with previous scores
        # e^(x-a) = z
        # e^(x-b) = ze^(a-b) this is e^(a-b) term
        # rearrange and (add dim)/broadcast to multiply with z. (b q h 1)
        # a is max_score term in attention
        correction = rearrange(jnp.exp(prev_max_score - max_score), 'b h q -> b q h')[..., None]
        # ze^(a-b) is corrected weighted query values from previous shards
        # jnp.einsum("bhqk,bkhd->bqhd", exp_weights, v) is the current kv shards
        # weighted attention output
        # to get the e^
        numerator = numerator * correction + jnp.einsum("bhqk,bkhd->bqhd", exp_weights, v)
        # Denominator's exp expressions max_score term corrected
        denominator = denominator * jnp.exp(prev_max_score - max_score) + jnp.sum(exp_weights, axis=-1)
        # permute kv from i device to i+1 in ring of size = axis size
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
   
    # Ring attention outer loop, permute/rotate ring degree times aka full rotation
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        f = scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v),
        xs=jnp.arange(0, axis_size),
    )
    # compute final attention output
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_mask, numerator, denominator, max_score)


def _ring_attention_inference_bwd(axis_name, float32_logits, res, g):
    del float32_logits
    axis_size = lax.psum(1, axis_name)
    output, q, k, v, attn_mask, numerator, denominator, max_score = res
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    batch, kv_len, num_heads, dim_per_head = k.shape
    scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        ds = jnp.einsum("bqhd,bkhd->bhqk", g, v)
        dl = (ds - jnp.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        dq = dq + jnp.einsum("bhqk,bkhd->bqhd", dl, k) / scale
        dk = dk + jnp.einsum("bqhd,bhqk->bkhd", q, dl) / scale
        dv = dv + jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None


@partial(jax.custom_vjp, nondiff_argnums=[4, 5])
def ring_attention_inference(q, k, v, attn_mask, axis_name, float32_logits=True):
    y, _ = _ring_attention_inference_fwd(q, k, v, attn_mask, axis_name, float32_logits)
    return y


ring_attention_inference.defvjp(_ring_attention_inference_fwd, _ring_attention_inference_bwd)
