import functools
import math
import time
from typing import Optional, Callable, Tuple
import flax.linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from einops import rearrange
from enum import Enum
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from flax.linen import partitioning as nn_partitioning
from padded_flash_attn import make_dense_padded_attention


Mesh = jax.sharding.Mesh
AxisNames = tuple[str, ...]
BlockSizes = splash_attention_kernel.BlockSizes

class Masking(Enum):
    FULL = 1
    PADDING = 2
    SEGMENT = 3
    
def _reshape_heads_to_head_dim(tensor):
  # takes a tensor of shape [b, h, s, d] and reshapes to [b, s, h * d]
  # This is used to transform the output of flash attention back into the format of other attention outputs
  b, h, s, d = tensor.shape
  tensor = jnp.transpose(tensor, axes=[0, 2, 1, 3])
  reshaped_tensor = jnp.reshape(tensor, (b, -1, h * d))
  return jax.lax.with_sharding_constraint(reshaped_tensor, PartitionSpec("data", "fsdp", "tensor"))
    
def _unflatten_heads(tensor, heads):
  # reshapes from [b, s, h * d] to [b, h, s, d] (input format to flash format)
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  # Transpose to ('batch', 'heads', 'length', 'kv')
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _reshape_data_for_flash(tensor, heads):
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
  """
  if tensor.ndim != 4:
    tensor = _unflatten_heads(tensor, heads)
  return tensor

def _pad_data_for_flash(tensor, heads, flash_block_size, num_shards: int = 1):
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
  """
  tensor = _reshape_data_for_flash(tensor, heads)

  # Pad head_dim to 128 if less than that.
  kv_size = tensor.shape[-1]
  head_dim_pad = 0
  if kv_size < 128:
    head_dim_pad = 128 - kv_size

  # Pad seq_len with sharding constraints.
  seq_len = tensor.shape[2]

  # 1. First, pad seq_len to be a multiple of flash_block_size
  rem = seq_len % flash_block_size
  if rem != 0:
    seq_len_padded_pre = seq_len + (flash_block_size - rem)
  else:
    seq_len_padded_pre = seq_len

  # 2. Ensure num_blocks is divisible by num_shards
  num_blocks = seq_len_padded_pre // flash_block_size
  if num_blocks % num_shards != 0:
    num_blocks += num_shards - (num_blocks % num_shards)

  final_padded_len = num_blocks * flash_block_size
  seq_len_pad = final_padded_len - seq_len

  if kv_size < 128 or seq_len_pad != 0:
    npad = ((0, 0), (0, 0), (0, seq_len_pad), (0, head_dim_pad))
    tensor = jnp.pad(tensor, npad)

  return tensor, kv_size, seq_len

NUM_LANES = 128
def pad_kv_seq_to_lanes(tensor):
    seq_len = tensor.shape[2]
    if seq_len % NUM_LANES != 0:
        seq_len_pad = seq_len + (NUM_LANES - (seq_len % NUM_LANES))
    npad = ((0, 0), (0, 0), (0, seq_len_pad), (0, 0))
    tensor  = jnp.pad(tensor, npad)
    return tensor, seq_len


@functools.partial(jax.jit, static_argnames=("heads",
                                             "mesh",
                                             "axis_names_q",
                                             "axis_names_kv",
                                             "flash_block_sizes",
                                             "dtype",
                                             "attention_kernel",
                                             "mask_type"
                                             ))
def _tpu_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    attention_kernel: str = "flash",
    mask_type: Masking = Masking.FULL
) -> jax.Array:
    
    num_fsdp_shards = mesh.shape["fsdp"]
    query = _reshape_data_for_flash(query, heads)
    key = _reshape_data_for_flash(key, heads)
    value = _reshape_data_for_flash(value, heads)
    q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
    kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
    block_sizes = flash_block_sizes
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
        out_specs=q_axis_names,
        check_rep=False,
    )
    def wrap_flash_attention(query, key, value):

        query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_sizes.block_q)
        key, _, key_seq_len = _pad_data_for_flash(key, heads, block_sizes.block_kv)
        value, _, _ = _pad_data_for_flash(value, heads, block_sizes.block_kv)

        q_padded_len = query.shape[2]
        kv_padded_len = key.shape[2]
        jax.debug.print("q_orig_len {q_orig_len}, padded_len: {q_padded_len}, kv_orig_len {kv_orig_len}, padded_len: {kv_padded_len}", 
                        q_orig_len=query_seq_len, 
                        q_padded_len=q_padded_len, 
                        kv_orig_len=key_seq_len, 
                        kv_padded_len=kv_padded_len,
                       )

        if mask_type == Masking.FULL and attention_kernel != "dense_padded":
            mask = splash_attention_mask.FullMask(_shape=(q_padded_len, kv_padded_len))
            multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])
            segment_ids = None
        elif mask_type == Masking.PADDING and attention_kernel != "dense_padded":
            padding_mask = splash_attention_mask.PaddingMask(
                shape=(q_padded_len, kv_padded_len),
                q_seq_len=query_seq_len,
                kv_seq_len=key_seq_len
            )
            multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(padding_mask,) * query.shape[1])
            segment_ids = None
        elif mask_type == Masking.SEGMENT and attention_kernel != "dense_padded":
            q_indices = jax.lax.broadcasted_iota(jnp.int32, (q_padded_len,), 0)
            q_segment_ids = (q_indices < query_seq_len).astype(jnp.int32)
            kv_indices = jax.lax.broadcasted_iota(jnp.int32, (kv_padded_len,), 0)
            kv_segment_ids = (kv_indices < key_seq_len).astype(jnp.int32)
            segment_ids = splash_attention_kernel.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
            mask = splash_attention_mask.FullMask(_shape=(q_padded_len, kv_padded_len))
            multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

        # jax.debug.print("Is cross attention: {is_cross_attention}, q_padded_len: {q_padded_len}, kv_padded_len: {kv_padded_len}", is_cross_attention=is_cross_attention, q_padded_len=q_padded_len, kv_padded_len=kv_padded_len)
        # make_splash_mha is wrapped around shardmap and seq and head is already
        # sharded based on in_specs, therefore setting head_shards=1 and q_seq_shards=1.
        

        if attention_kernel == "flash":
            splash_kernel = splash_attention_kernel.make_splash_mha(
                mask=multi_head_mask,
                head_shards=1,  # the sizes of the axis is sharding over heads
                q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
                block_sizes=block_sizes,
                save_residuals=True if attention_kernel == "ring" else False
                )
            vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0), out_axes=0)
            if segment_ids:
                vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None), out_axes=0)
                attention_output = vmapped_splash(query, key, value, segment_ids)
            else:
                vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0), out_axes=0)
                attention_output = vmapped_splash(query, key, value)
        elif attention_kernel == "ring":
            splash_kernel = splash_attention_kernel.make_splash_mha(
                mask=multi_head_mask,
                head_shards=1,  # the sizes of the axis is sharding over heads
                q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
                block_sizes=block_sizes,
                save_residuals=True if attention_kernel == "ring" else False
                )
            if num_fsdp_shards > 1:
                out, (lse,) = vmapped_splash(query, key, value, segment_ids)
                m = lse.astype(jnp.float32)
                l = jnp.exp(lse - m)
                o = out.astype(jnp.float32) * l[..., None]

                perm = [(j, (j + 1) % num_fsdp_shards) for j in range(num_fsdp_shards)]

                k1 = jax.lax.ppermute(key, axis_name="fsdp", perm=perm)
                v1 = jax.lax.ppermute(value, axis_name="fsdp", perm=perm)

                def ring_scan_body(carry, _):
                    m, l, o, k_current, v_current = carry
                    k_next = jax.lax.ppermute(k_current, axis_name="fsdp", perm=perm)
                    v_next = jax.lax.ppermute(v_current, axis_name="fsdp", perm=perm)

                    out_chunk, (lse_chunk,) = vmapped_splash(query, k_current, v_current, segment_ids)

                    m_chunk = lse_chunk.astype(jnp.float32)
                    m_old = m
                    m = jnp.maximum(m_old, m_chunk)

                    exp_m_diff = jnp.exp(m_old - m)
                    exp_m_chunk_diff = jnp.exp(m_chunk - m)

                    l = l * exp_m_diff + jnp.exp(lse_chunk - m)
                    o = o * exp_m_diff[..., None]
                    o += exp_m_chunk_diff[..., None] * out_chunk.astype(jnp.float32)

                    # Return the updated state for the next iteration
                    return (m, l, o, k_next, v_next), None

                initial_carry = (m, l, o, k1, v1)
                (m_final, l_final, o_final, _, _), _ = jax.lax.scan(ring_scan_body, initial_carry, None, length=num_fsdp_shards - 1)

            attention_output = o_final / l_final[..., None]
        elif attention_kernel == "dense_padded":
            padded_kv_len = key.shape[1] - key_seq_len
            dense_padded_attention_kernel = make_dense_padded_attention(block_sizes=block_sizes, kv_padding=padded_kv_len)
            vmapped_splash = jax.vmap(dense_padded_attention_kernel, in_axes=(0, 0, 0), out_axes=0)
            attention_output, _ = vmapped_splash(query, key, value)
        else:
            raise ValueError(f"Unknown attention kernel: {attention_kernel}")

        return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)
    
    devices_in_data_fsdp = mesh.shape["data"] * mesh.shape["fsdp"]
    x = wrap_flash_attention(query, key, value)
    x = _reshape_heads_to_head_dim(x)

    return x
    
# MESH AXES
DATA = "data"
FSDP = "fsdp"
TENSOR = "tensor"
# LOGICAL AXES
BATCH = "activation_batch"
D_KV = "activation_kv"
ATTN_HEAD = "activation_attn_heads"
ATTN_Q_LENGTH = "activation_attn_q_length"
ATTN_KV_LENGTH = "activation_attn_kv_length"
# LOGICAL AXES mapping to qkv tensor axes
axis_names_q = (BATCH, ATTN_HEAD, ATTN_Q_LENGTH, D_KV)
axis_names_kv = (BATCH, ATTN_HEAD, ATTN_KV_LENGTH, D_KV)

### LOGICAL AXES TO PHYSICAL AXES MAPPING ###
RING_ATTENTION_AXIS_RULES = [
        [BATCH, DATA],  
        [ATTN_HEAD, None],
        [ATTN_Q_LENGTH, FSDP],
        [ATTN_KV_LENGTH, FSDP],
        [D_KV, None]
        
]

SEQUENCE_PARALLEL_AXIS_RULES = [
        [BATCH, DATA],
        [ATTN_HEAD, None],
        [ATTN_Q_LENGTH, FSDP],
        [ATTN_KV_LENGTH, None],
        [D_KV, None ]
]

TENSOR_PARALLEL_AXIS_RULES = [
        [BATCH, DATA],
        [ATTN_HEAD, TENSOR],
        [ATTN_Q_LENGTH, None],
        [ATTN_KV_LENGTH, None],
        [D_KV, None]
]
    
    
def main():
    BQ = [3024]
    BKV = [2048]
    BKV_COMPUTE = [1024]
    rng = jax.random.key(1)
    query = jax.random.normal(rng,(2, 40, 75600, 128), dtype=jnp.bfloat16)
    rng = jax.random.key(2)
    key = jax.random.normal(rng,(2, 40, 75600, 128), dtype=jnp.bfloat16)
    rng = jax.random.key(3)
    value = jax.random.normal(rng,(2, 40, 75600, 128), dtype=jnp.bfloat16)
    # query = jnp.ones((2, 4, 3024, 128))
    # key = jnp.ones((2, 4, 2048, 128))
    # value = jnp.ones((2, 4, 2048, 128))
    data=2
    fsdp=1
    tensor=4
    mesh_devices = mesh_utils.create_device_mesh((data, fsdp, tensor), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('data','fsdp','tensor'))
   
    for bq in BQ:
        for bk in BKV:
            for bk_compute in BKV_COMPUTE:
                block_sizes = splash_attention_kernel.BlockSizes(
                    block_q=bq,
                    block_kv_compute=bk_compute,
                    block_kv=bk)
                print(block_sizes)
                for mask in Masking:
                    for attn in ["dense_padded","flash" ]:
                        if mask != Masking.FULL and attn == "dense_padded":
                            print("==========SKIP NON FULL MASK DENSE PADDED ATTN")
                            continue
                        with mesh, nn_partitioning.axis_rules(TENSOR_PARALLEL_AXIS_RULES):
                            print (f"==========CASE {bq} {bk} {bk_compute} mask {mask} attn {attn}==========")
                            print("==========COMPILE==========")
                            lhs = _tpu_flash_attention(
                                            query, 
                                            key, 
                                            value, 
                                            heads=40, 
                                            mesh=mesh,  
                                            axis_names_q=axis_names_q, 
                                            axis_names_kv=axis_names_kv, 
                                            flash_block_sizes=block_sizes, 
                                            dtype=jnp.bfloat16,
                                            attention_kernel=attn,
                                            mask_type=mask
                                            )
                            jax.block_until_ready(
                                        lhs
                                    )
                            rhs = _tpu_flash_attention(
                                            query, 
                                            key, 
                                            value, 
                                            heads=40, 
                                            mesh=mesh,  
                                            axis_names_q=axis_names_q, 
                                            axis_names_kv=axis_names_kv, 
                                            flash_block_sizes=block_sizes, 
                                            dtype=jnp.bfloat16,
                                            attention_kernel="flash",
                                            mask_type=Masking.SEGMENT
                                            )
                            jax.block_until_ready(
                                        rhs
                                    )
                            
                            allclose = jnp.allclose(lhs, rhs, rtol=1e-3, atol=1e-3)
                            mean_diff = jnp.mean(jnp.abs(lhs - rhs))
                            print(f"==========All close {allclose} mean diff {mean_diff}==========")
                            start = time.perf_counter()
                            print("==========PROFILE==========")
                            jax.block_until_ready(
                                        _tpu_flash_attention(
                                            query, 
                                            key, 
                                            value, 
                                            heads=40, 
                                            mesh=mesh,  
                                            axis_names_q=axis_names_q, 
                                            axis_names_kv=axis_names_kv, 
                                            flash_block_sizes=block_sizes, 
                                            dtype=jnp.bfloat16,
                                            attention_kernel=attn,
                                            mask_type=mask
                                            )
                                    )
                            end = time.perf_counter()
                            print("==========RESULT========")
                            print(f"=========={end - start}s block {bq} {bk} {bk_compute} mask {mask} attn {attn}==========")
                            print("==========END========")
                    
                    
if __name__ == "__main__":
    main()                    