# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementation of Ring Attention."""

import functools
from typing import Any

import jax
from jax import lax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
from . import base
from . import splash_attention_kernel as splash_kernel
from . import splash_attention_mask as mask_lib
from . import splash_attention_mask_info as mask_info_lib

P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial

SegmentIds = base.SegmentIds
SplashConfig = splash_kernel.SplashConfig
SplashResidualsType = base.SplashResidualsType
SplashCustomReturnType = base.SplashCustomReturnType
MaskFunctionType = splash_kernel.MaskFunctionType
_splash_attention_forward = splash_kernel._splash_attention_forward  # pylint: disable=protected-access
_splash_attention_bwd = splash_kernel._splash_attention_bwd  # pylint: disable=protected-access


def _dynamic_slice_mask_info(
    mask_info: MaskInfo, kv_shard_idx: jax.Array, ring_size: int
) -> MaskInfo:
  """Slices MaskInfo for the current ring step."""

  def slice_if_exists(arr: jax.Array | None):
    if arr is None:
      return None

    shard_len = int(arr.shape[-1]) // ring_size
    start_idx = kv_shard_idx * shard_len
    return lax.dynamic_slice_in_dim(arr, start_idx, shard_len, axis=-1)

  return MaskInfo(
      mask_next=slice_if_exists(mask_info.mask_next),
      active_rows=slice_if_exists(mask_info.active_rows),
      active_cols=slice_if_exists(mask_info.active_cols),
      num_active_blocks=slice_if_exists(mask_info.num_active_blocks),
      block_mask=slice_if_exists(mask_info.block_mask),
      partial_mask_blocks=mask_info.partial_mask_blocks,  # partial mask blocks are global
      q_sequence=mask_info.q_sequence,  # Q sequence stays stationary
  )


def _ring_attention_forward(
    fwd_mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    *,
    sinks: jax.Array | None = None,
    ring_axis: str,
    rotate_segment_ids: bool = True,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:

  if q.shape[-1] != k.shape[-1]:
    raise NotImplementedError(
        "Queries and keys must have the same head dimension."
    )

  if sinks is not None:
    raise NotImplementedError("Sinks aren't supportd yet.")

  ring_axis_size = lax.axis_size(ring_axis)
  ring_axis_idx = lax.axis_index(ring_axis)

  shift = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
  )
  # for example, if ring size is 4
  # Device 3 => permute_idx 0, offset (3-0) % 4 = 3,
  #             permute_idx 1, offset (3-1) % 4 = 2, etc.
  # Device 2 => permute_idx 0, offset (2-0) % 4 = 2,
  #             permute_idx 1, offset (2-1) % 4 = 1, etc.
  # Device 1 => permute_idx 0, offset (1-0) % 4 = 1,
  #             permute_idx 1, offset (1-1) % 4 = 0, etc.
  # Device 0 => permute_idx 0, offset (0-0) % 4 = 0,
  #             permute_idx 1, offset (0-1) % 4 = 3, etc.

  splash_fwd_partial = partial(
      _splash_attention_forward,
      save_residuals=True,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=None,
  )
  # Initial accumulator values
  o_shape = q.shape
  o_init = jnp.zeros(o_shape, dtype=jnp.float32)
  l_init = jnp.zeros((o_shape[0], o_shape[1]), jnp.float32)
  m_init = jnp.full_like(l_init, mask_value, dtype=jnp.float32)

  def body(carry, i: int)-> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, SegmentIds | None], None]:
    m_prev, l_prev, o_prev, k_current, v_current, segment_ids_current = carry

    current_kv_shard_idx = (ring_axis_idx - i) % ring_axis_size
    local_fwd_mask_info = _dynamic_slice_mask_info(
        fwd_mask_info, current_kv_shard_idx, ring_axis_size
    )
    k_next = shift(k_current)
    v_next = shift(v_current)

    if segment_ids is not None and rotate_segment_ids:
      kv_segment_ids_next = shift(segment_ids_current.kv)
      segment_ids_next = SegmentIds(segment_ids.q, kv_segment_ids_next)
    else:
      segment_ids_next = segment_ids_current

    out_curr, stats = splash_fwd_partial(
        local_fwd_mask_info,
        q,
        k_current,
        v_current,
        segment_ids=segment_ids_current,
        sinks=sinks,
    )
    lse_curr = stats["logsumexp"]
    m_curr = stats["max_logits"]
    l_curr = jnp.exp(lse_curr - m_curr)
    o_curr = out_curr.astype(jnp.float32) * l_curr[..., None]
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    o_next = alpha[..., None] * o_prev + beta[..., None] * o_curr
    return (m_next, l_next, o_next, k_next, v_next, segment_ids_next), None

  # Use lax.scan to get the final carry AND the collected sequence of (k,v)
  # pairs
  initial_carry = (m_init, l_init, o_init, k, v, segment_ids)
  (m_final, l_final, o_final, _, _, _), _ = lax.scan(
      body,
      initial_carry,
      xs=jnp.arange(0, ring_axis_size),
      length=ring_axis_size,
      unroll=True,
  )  # type: ignore[arg-type]
  # Final normalization
  assert l_final.dtype == jnp.float32
  l_inv = jnp.where(l_final == 0.0, 0.0, 1.0 / l_final)
  out = (o_final * l_inv[..., None]).astype(q.dtype)
  # Final logsumexp for residuals
  lse = jnp.log(l_final) + m_final
  lse = jnp.where(l_final == 0.0, mask_value, lse)

  return out, (lse, m_final)


def _ring_attention_bwd(
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    rotate_segment_ids: bool,
    # Residuals and gradients
    res: Any,
    do: jax.Array,
):
  del save_residuals
  (q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info) = res
  do = do.astype(jnp.float32)

  ring_axis_size = lax.axis_size(ring_axis)
  ring_axis_idx = lax.axis_index(ring_axis)

  shift = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
  )
  dq_accum = jnp.zeros_like(q, dtype=jnp.float32)
  dk_accum = jnp.zeros_like(k, dtype=jnp.float32)
  dv_accum = jnp.zeros_like(v, dtype=jnp.float32)
  dsinks = sinks

  def body(carry, i: int):
    (
        dq_accum,
        dk_accum,
        dv_accum,
        k_current,
        v_current,
        segment_ids_current,
        _,
    ) = carry
    k_next = shift(k_current)
    v_next = shift(v_current)

    current_kv_shard_idx = (ring_axis_idx - i) % ring_axis_size
    local_dkv_mask_info = _dynamic_slice_mask_info(
        dkv_mask_info, current_kv_shard_idx, ring_axis_size
    )
    if segment_ids is not None and rotate_segment_ids:
      kv_segment_ids_next = shift(segment_ids_current.kv)
      segment_ids_next = SegmentIds(segment_ids.q, kv_segment_ids_next)
    else:
      segment_ids_next = segment_ids_current

    residuals_for_chunk = (
        q,
        k_current,
        v_current,
        segment_ids_current,
        sinks,
        out,
        logsumexp,
        local_dkv_mask_info,
    )

    attn_bwd = functools.partial(
        _splash_attention_bwd,
        save_residuals=False,
        mask_value=mask_value,
        is_mqa=is_mqa,
        config=config,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        dkv_mask_sparsity=dkv_mask_sparsity,
    )
    _, _, dq_i, dk_i, dv_i, _, dsinks, _ = attn_bwd(
        res=residuals_for_chunk, do=do
    )
    dv_next = shift(dv_accum + dv_i.astype(dv_accum.dtype))
    dk_next = shift(dk_accum + dk_i.astype(dk_accum.dtype))
    dq_accum = dq_accum + dq_i.astype(dq_accum.dtype)

    return (
        dq_accum,
        dk_next,
        dv_next,
        k_next,
        v_next,
        segment_ids_next,
        dsinks,
    ), None

  initial_carry = (dq_accum, dk_accum, dv_accum, k, v, segment_ids, dsinks)
  (dq, dk, dv, _, _, _, dsinks), _ = lax.scan(
      body,
      initial_carry,
      xs=jnp.arange(ring_axis_size),
      length=ring_axis_size,
      unroll=True,
  )

  if sinks is not None:
    dsinks = jax.lax.psum(dsinks, axis_name=ring_axis)

  return (
      None,  # fwd_mask_info
      None,  # dkv_mask_info
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      dsinks,
      None,
  )


def _ring_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    # nondiff_args
    mask_value: float,  # 1
    is_mqa: bool,  # 2
    config: SplashConfig | None,  # 3
    mask_function: MaskFunctionType | None,  # 4
    fwd_mask_sparsity: float,  # 5
    dkv_mask_sparsity: float,  # 6
    save_residuals: bool,  # 7
    ring_axis: str,  # 8
    rotate_segment_ids: bool,  # 9
) -> tuple[jax.Array, SplashResidualsType]:
  """Forward pass for the custom VJP of ring attention.

  This function is used by `jax.custom_vjp` to define the forward pass
  of the ring attention computation, also returning residuals needed for
  the backward pass.

  Args:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    q: Query array.
    k: Key array.
    v: Value array.
    segment_ids: Optional segment IDs for packed sequences.
    sinks: Optional sink tokens.
    mask_value: The value used for masked-out attention scores.
    is_mqa: Whether Multi-Query Attention is used.
    config: SplashAttention configuration.
    mask_function: Optional function to apply additional masking.
    fwd_mask_sparsity: Sparsity level of the forward mask.
    save_residuals: Whether to save residuals for the backward pass.
    ring_axis: The name of the jax axis used for the ring.

  Returns:
    A tuple containing:
      - The output of the ring attention computation.
      - Residuals needed for the backward pass (`SplashResidualsType`).
  """
  del dkv_mask_sparsity
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported.")

  out, (logsumexp, max_logits) = _ring_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      ring_axis=ring_axis,
      rotate_segment_ids=rotate_segment_ids,
  )
  residuals = (q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info)
  return out, residuals


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "save_residuals",
        "ring_axis",
        "rotate_segment_ids",
    ),
)
def _ring_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    rotate_segment_ids: bool ,
) -> SplashCustomReturnType:
  """Performs ring attention with a custom VJP.

  This function is a wrapper around `_ring_attention_forward` and is used
  to define the custom gradient for ring attention.

  Args:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    q: Query array.
    k: Key array.
    v: Value array.
    segment_ids: Optional segment IDs for packed sequences.
    sinks: Optional sink tokens.
    mask_value: The value used for masked-out attention scores.
    is_mqa: Whether Multi-Query Attention is used.
    config: SplashAttention configuration.
    mask_function: Optional function to apply additional masking.
    fwd_mask_sparsity: Sparsity level of the forward mask.
    save_residuals: Whether to save residuals for the backward pass.
    ring_axis: The name of the jax axis used for the ring.
    rotate_segment_ids: Whether to rotate segment IDs along with K/V in ring attention.
        This only possible when segment id for all KV shards are same, i.e ring attention is called in shard map.
  Returns:
    The output of the ring attention computation.
  """
  del dkv_mask_info, dkv_mask_sparsity
  out, _ = _ring_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      ring_axis=ring_axis,
      rotate_segment_ids=rotate_segment_ids,
  )
  return out


_ring_attention_custom.defvjp(_ring_attention_fwd, _ring_attention_bwd)


def _has_axis(axis_name: str) -> bool:
  try:
    # We try to access the size of the axis.
    # If it doesn't exist, JAX raises a NameError (or similar) immediately
    # during tracing.
    lax.axis_size(axis_name)
    return True
  except (NameError, ValueError):
    return False


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "save_residuals",
        "ring_axis",
        "rotate_segment_ids",
    ],
)
def _ring_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool = False,
    ring_axis: str,
    rotate_segment_ids: bool = True,
) -> SplashCustomReturnType:
  """Performs ring attention using SplashAttention kernels.

  This function orchestrates the ring attention mechanism by iterating through
  shards of keys and values across devices along the specified `ring_axis`,
  using `_splash_attention_forward` for each chunk.

  Args:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    q: Query array.
    k: Key array.
    v: Value array.
    segment_ids: Optional segment IDs for packed sequences.
    sinks: Optional sink tokens.
    is_mqa: Whether Multi-Query Attention is used.
    config: SplashAttention configuration.
    mask_value: The value used for masked-out attention scores.
    mask_function: Optional function to apply additional masking.
    fwd_mask_sparsity: Sparsity level of the forward mask.
    save_residuals: Whether to save residuals for the backward pass.
    ring_axis: The name of the jax axis used for the ring.
      rotate_segment_ids: Whether to rotate segment IDs along with K/V in ring attention

  Returns:
    The output of the ring attention computation.

  Raises:
    ValueError: If the specified `ring_axis` does not exist.
  """
  if not _has_axis(ring_axis):
    raise ValueError(f"Ring axis {ring_axis} does not exist")

  return _ring_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      is_mqa=is_mqa,
      config=config,
      mask_value=mask_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
      save_residuals=save_residuals,
      ring_axis=ring_axis,
      rotate_segment_ids=rotate_segment_ids,
  )


@jax.tree_util.register_pytree_node_class
class RingSplashAttentionKernel:
  """Implements Ring Attention using SplashAttention for sequence parallelism.

  This kernel computes global attention by keeping Keys and Values distributed
  across the `ring_axis`. Instead of gathering full sequences, it rotates K/V 
  shards between devices and accumulates results incrementally. This allows
  processing sequence lengths that exceed single-device memory limits.

  Attributes:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    ring_axis: The name of the jax axis used for the ring.
    kwargs: Additional keyword arguments passed to the SplashAttentionKernel
      constructor.
  """

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      ring_axis: str,
      rotate_segment_ids: bool ,
      **kwargs,
  ):
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info
    self.ring_axis = ring_axis
    self.rotate_segment_ids = rotate_segment_ids
    self.kwargs = kwargs

  def __call__(self, *args, **kwargs):
    return _ring_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **kwargs,
        **self.kwargs,
        ring_axis=self.ring_axis,
        rotate_segment_ids=self.rotate_segment_ids,
    )

  def manual_sharding_spec(self):
    """Ring attention expects MaskInfo to be sharded by `q_seq_shards`.

    Each q shard will need all the k/v shard's MaskInfo eventually, so we don't
    shard it, but instead dynamic_slice the chunk that we need at each
    iteration.
    """

    spec = jax.sharding.PartitionSpec(self.ring_axis)
    _resolve_spec = lambda x: spec if x is not None else None

    mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
        mask_next=_resolve_spec(self.fwd_mask_info.mask_next),
        active_rows=_resolve_spec(self.fwd_mask_info.active_rows),
        active_cols=_resolve_spec(self.fwd_mask_info.active_cols),
        num_active_blocks=_resolve_spec(self.fwd_mask_info.num_active_blocks),
        block_mask=_resolve_spec(self.fwd_mask_info.block_mask),
        partial_mask_blocks=jax.sharding.PartitionSpec(),  # replicated
        q_sequence=_resolve_spec(self.fwd_mask_info.q_sequence),
    )
    return RingSplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dkv_mask_info is not None else None,
        ring_axis=self.ring_axis,
        **self.kwargs,
    )

  def tree_flatten(self):
    children = (self.fwd_mask_info, self.dkv_mask_info)
    aux_data = self.kwargs.copy()
    aux_data["ring_axis"] = self.ring_axis
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    fwd_mask_info, dkv_mask_info = children
    dkv_mask_info = (
        mask_info_lib.MaskInfo(*dkv_mask_info)
        if dkv_mask_info is not None
        else None
    )
    return cls(
        mask_info_lib.MaskInfo(*fwd_mask_info),
        dkv_mask_info,
        **aux_data,
    )


def make_ring_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    ring_axis: str,
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
    rotate_segment_ids: bool = True,
):
  """Creates a RingSplashAttentionKernel.

  Args:
    mask: The attention mask.
    config: SplashAttention configuration. If None, uses the default config.
    is_mqa: Whether the model uses Multi-Query Attention.
    save_residuals: Whether to save residuals for the backward pass.
    mask_value: The value to use for masked-out attention scores.
    downcast_smem_data: Whether to downcast data in shared memory.
    partial_mask_blocks_dtype: The dtype for partial mask blocks.
    ring_axis: The name of the jax scan axis used for the ring.
    q_seq_shards: The number of shards for the query sequence dimension.
    kv_seq_shards: The number of shards for the key/value sequence dimension.
    rotate_segment_ids: Whether to rotate segment IDs along with K/V in ring attention
    This only possible when segment id for all KV shards are same, i.e ring attention is called in shard map.
    Common scenario being padding applied to each shard independently, so all shards have same segment ids.
  Returns:
    A RingSplashAttentionKernel instance.

  Raises:
    ValueError: If the mask shape is unexpected or ring_axis is not specified
  """

  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = mask_lib.NumpyMask(mask)

  if not isinstance(mask, (mask_lib.NumpyMask, mask_lib.FullMask)):
    raise NotImplementedError(
        f"Only NumpyMask and FullMask are supported, but got {type(mask)}."
    )

  if config is None:
    config = SplashConfig.get_default()

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
      kv_seq_shards=kv_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  dkv_mask_sparsity = 0.0
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)
    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

  return RingSplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      ring_axis=ring_axis,
      rotate_segment_ids=rotate_segment_ids,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )
