from functools import partial
import math
import jax
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import BlockSizes
from jaxtyping import Array, Float16
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import numpy as jnp, lax
import numpy as np


NUM_LANES=128
NUM_SUBLANES=8
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
QK_DOT_DIM_NUMBERS_SEQ_MAJOR = (((1,), (1,)), ((), ()))  # RHS transposed
SV_DOT_DIM_NUMBERS_SEQ_MAJOR = (((1,), (0,)), ((), ()))  # standard matmul
save_residuals=False

DensePaddedAttentionReturnType = (
    Float16[Array, "heads q_seq_len head_dim"] | 
    tuple[ 
        Float16[Array, "heads q_seq_len head_dim"] , # out
        tuple[
            Float16[Array, "heads q_seq_len lanes"], # l
                Float16[Array, "heads q_seq_len lanes"] # m 
            ]
        ]
)

def _dense_padded_flash_attn_fwd_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    mask_scratch_ref,
    qk_scratch_ref,
    # statics
    *,
    kv_padding: int,
    mask_value:float,
    kv_steps: int,
    bkv_compute:int,
    head_dim_v: int,
):
    h = pl.program_id(0)
    bq_i = pl.program_id(1)
    bkv_j = pl.program_id(2)
    
    # initialize accumulation tensors in scratch every bq_i
    should_initialize = bkv_j == 0 
    is_last_k_block = bkv_j == kv_steps - 1
    padding_exists = kv_padding > 0
    masking_is_needed_for_block = is_last_k_block & padding_exists
    @pl.when(should_initialize)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        
        @pl.when(masking_is_needed_for_block)
        def init_actual_mask():
             # Initialize the full (bq, bkv) mask for the last block
            col_indices = jnp.arange(bkv_compute)
            # Calculate how many tokens in this chunk are NOT padding
            num_real_tokens = bkv_compute - kv_padding  # Shape (block_kv,)
            mask_row = col_indices < num_real_tokens # True for real, False for padding
            mask_scratch_ref[...] = jnp.broadcast_to(mask_row, mask_scratch_ref.shape)
        
        
    def mask(qk, bkv_slice):
        mask_arr = mask_scratch_ref[:, bkv_slice]
        qk = jnp.where(mask_arr, qk, mask_value)
        return qk
    
    should_write = bkv_j == kv_steps - 1
    padding_exists = kv_padding > 0
    masking_is_needed_for_block = should_write & padding_exists
    num_iters = (
      k_ref.shape[0] // bkv_compute
    )
    # body for lax.fori loop over bkv compute blocks
    def body(kv_compute_index, _):
        # Reads from VMEM to VREG
        # compute BKV_COMPUTE slice from BK k sequence
        # idea compute slices before in prefetch scalars
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        # Softmax stats
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        # BQ read
        q = q_ref[...]
        # BKV_COMPUTE read 
        k = k_ref[slice_k, :] # HEAD DIM minor
        # QK
        qk_dims = QK_DOT_DIM_NUMBERS_SEQ_MAJOR # TODO: option to support K transpose
        qk = lax.dot_general(q, k, qk_dims, preferred_element_type=jnp.float32)
        qk_scratch_ref[...] = qk
        
        @pl.when(jnp.logical_and(kv_compute_index==num_iters-1,should_write))
        def mask():
            # mask_arr = ]
            # qk_t = qk_scratch_ref[...]
            qk_scratch_ref[...] = jnp.where(mask_scratch_ref[...], qk_scratch_ref[...], mask_value)
            
        qk = qk_scratch_ref[...]
        # Running max
        m_curr = qk.max(axis=-1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)
        
        # Current numerator
        bkv_repeats = bkv_compute//NUM_LANES
        exp = jnp.exp
        s_curr = exp(qk - pltpu.repeat(m_next, bkv_repeats, axis=1))
        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))

        # Correction factor
        alpha = exp(m_prev - m_next)
        
        # Accumulate denominator 
        l_next = l_curr + alpha * l_prev
        
        # Update softmax stats
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
        
        # numerator * V 
        sv_dims = SV_DOT_DIM_NUMBERS_SEQ_MAJOR
        v = v_ref[slice_k, :]
        v = v.astype(jnp.float32)
        o_curr = lax.dot_general(s_curr, v, sv_dims)
        
        # Accumulate unnormalized O
        head_dim_v_repeats = head_dim_v//NUM_LANES
        alpha_o = pltpu.repeat(alpha, head_dim_v_repeats, axis=1)
        o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
        
    
    lax.fori_loop(
        lower=0, 
        upper=num_iters, 
        body_fun=body, 
        init_val=None, 
        unroll=True)
    
    @pl.when(should_write)
    def end():
        l = l_scratch_ref[...]
        head_dim_v_repeats = head_dim_v//NUM_LANES
        l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=1)
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
        
        # UNCOMMENT FOR SAVING SOFTMAX STATS
        # if logsumexp_ref is not None:
        #     log = jnp.log # allow log base 2
        #     logsumexp = m_scratch_ref[...] + log(l)
        #     logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
        # if max_logits_ref is not None:
        #     max_logits_ref[...] = m_scratch_ref[...].astype(max_logits_ref.dtype)
    

    
def _dense_padded_flash_attn_custom_fwd(
    q: Float16[Array, "heads q_seq_len head_dim"],
    k: Float16[Array, "heads kv_seq_len head_dim"],
    v: Float16[Array, "heads kv_seq_len head_dim"],
    block_sizes: BlockSizes,
    kv_padding: int,
)-> DensePaddedAttentionReturnType:
    head_dim = q.shape[-1]
    num_heads = q.shape[0]
    q_sequence_len = q.shape[1]
    kv_sequence_len = k.shape[1]

    # Block specs
    # Input block specs 
    in_specs = [
        # BQ
        pl.BlockSpec(
            block_shape= (None, block_sizes.block_q, head_dim),
            index_map= lambda h, bq_i, bkv_j : (h, bq_i, 0)
        ),
        # BK
        pl.BlockSpec(
            block_shape= (None, block_sizes.block_kv, head_dim),
            index_map= lambda h, bq_i, bkv_j : (h, bkv_j, 0)
        ),
        # BV
        pl.BlockSpec(
            block_shape= (None, block_sizes.block_kv, head_dim),
            index_map= lambda h, bq_i, bkv_j : (h, bkv_j, 0)
        ),
    ]
    # Output block specs
    out_specs = [
        # out
        pl.BlockSpec(
            block_shape= (None, block_sizes.block_q, head_dim),
            index_map= lambda h, bq_i, bkv_j : (h, bq_i, 0)
        ),
    ]
    # Output Shape
    out_shapes = [
        # out
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
    ]
    
    if save_residuals:
        out_specs += [pl.BlockSpec( # logsumexp
            block_shape=(None, block_sizes.block_q, NUM_LANES), 
            index_map= lambda h, bq_i, bkv_j : (h, bq_i, 0)
            ),
        # max_logits
        pl.BlockSpec(
            block_shape=(None, block_sizes.block_q, NUM_LANES), 
            index_map= lambda h, bq_i, bkv_j : (h, bq_i, 0)
            )]

        out_shapes += [
            # logsumexp
            jax.ShapeDtypeStruct((num_heads, q_sequence_len, NUM_LANES), jnp.float32),
            # max_logits
            jax.ShapeDtypeStruct((num_heads, q_sequence_len, NUM_LANES), jnp.float32),
        ]
    else:
        out_specs += [None, None]
        out_shapes += [None, None]
    
    # Scratch shapes m,l,o,mask    
    scratch_shapes = [
        pltpu.VMEM( # m_scratch
            shape=(block_sizes.block_q, NUM_LANES), 
            dtype=jnp.float32),  
        pltpu.VMEM(  # l_scratch
            shape=(block_sizes.block_q, NUM_LANES), 
            dtype=jnp.float32), 
        pltpu.VMEM( # o_scratch
            shape=(block_sizes.block_q, head_dim), 
            dtype=jnp.float32),  
        pltpu.VMEM( # mask
            shape=(block_sizes.block_q, block_sizes.block_kv_compute), 
            dtype=jnp.bool),
        pltpu.VMEM(
            shape=(block_sizes.block_q, block_sizes.block_kv_compute), 
            dtype=jnp.float32
        )
        ]
    
    # Grid
    
    num_bq_blocks = q_sequence_len // block_sizes.block_q
    num_bkv_blocks = kv_sequence_len // block_sizes.block_kv
    grid = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0, 
        grid = (num_heads, num_bq_blocks, num_bkv_blocks),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes
        )
    
    # Compiler Params
    compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        )
    
    
    # Cost estimate
    def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
        if x is None:
            return 0
        if jnp.issubdtype(x.dtype, jnp.floating):
            info = jnp.finfo
        elif jnp.issubdtype(x.dtype, jnp.integer):
            info = jnp.iinfo
        else:
            raise ValueError(f"Unsupported dtype: {x.dtype}")
        return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)

    def _fwd_cost_estimate(
        q: Float16[Array, "heads q_seq_len head_dim"],
        k: Float16[Array, "heads kv_seq_len head_dim"],
        v: Float16[Array, "heads kv_seq_len head_dim"],
        out_shapes: list[jax.ShapeDtypeStruct],
    ) -> pl.CostEstimate | None:
        num_q_heads, q_seq_len, head_dim_qk = q.shape
        kv_seq_len, head_dim_v = v.shape[-2:]

        matmul_flops = (
            2 * q_seq_len * kv_seq_len * head_dim_qk
            + 2 * kv_seq_len * kv_seq_len * head_dim_v
        )

        # This is an upper bound because `mask_sparsity` is actually the mean
        # sparsity of the non-fully masked **blocks**.
        total_flops = num_q_heads * matmul_flops 

        # Count expensive exp() calls
        transcendentals = num_q_heads * q_seq_len * kv_seq_len

        inputs_ = [q, k, v]
        input_bytes = sum(map(_bytes, inputs_))
        output_bytes = sum(map(_bytes, out_shapes))
        return pl.CostEstimate(
            flops=int(total_flops),
            transcendentals=int(transcendentals),
            bytes_accessed=int(input_bytes + output_bytes),
        )
    
    vmem_inputs = [
      q,
      k,
      v,
    ]
    cost_estimate = _fwd_cost_estimate(*vmem_inputs, out_shapes)
    
    ## Pallas call
    kv_steps = kv_sequence_len//block_sizes.block_kv
    kv_compute_iters = block_sizes.block_kv//block_sizes.block_kv_compute
    dense_padded_attn_fwd_kernel = partial(
        _dense_padded_flash_attn_fwd_kernel,
        kv_padding=kv_padding,
        mask_value = DEFAULT_MASK_VALUE,
        kv_steps=kv_steps,
        bkv_compute=block_sizes.block_kv_compute,
        head_dim_v=head_dim
    )
    with jax.named_scope("dense_padded_flash_attn_fwd"):
        all_out = pl.pallas_call(
            kernel=dense_padded_attn_fwd_kernel,
            grid_spec = grid,
            compiler_params=compiler_params,
            cost_estimate=cost_estimate,
            out_shape=out_shapes,
            name="dense_padded_flash_attn_fwd",
        )(
            q, k, v
        )
    out, logsumexp, max_logits = all_out
    return out, (logsumexp, max_logits)



def _dense_padded_flash_attn_custom_vjp(
    q: Float16[Array, "heads q_seq_len head_dim"],
    k: Float16[Array, "heads kv_seq_len head_dim"],
    v: Float16[Array, "heads kv_seq_len head_dim"],
    block_sizes: BlockSizes,
    kv_padding: int,
)-> DensePaddedAttentionReturnType:
    return _dense_padded_flash_attn_custom_fwd(
        q, k, v, block_sizes, kv_padding
    )


@partial(
    jax.jit,
    static_argnames=("block_sizes", "kv_padding"),
)
def _dense_padded_flash_attention(
    q: Float16[Array, "heads q_seq_len head_dim"],
    k: Float16[Array, "heads kv_seq_len head_dim"],
    v: Float16[Array, "heads kv_seq_len head_dim"],
    *,
    block_sizes: BlockSizes,
    kv_padding: int,
)-> DensePaddedAttentionReturnType:
    return _dense_padded_flash_attn_custom_vjp(
        q, k, v, block_sizes, kv_padding
    )
    

@jax.tree_util.register_pytree_node_class
class DensePaddedAttention:
    
    def __init__(self, block_sizes: BlockSizes, kv_padding: int):
        self.block_sizes = block_sizes
        self.kv_padding = kv_padding
    def __call__(self, 
                q: Float16[Array, "heads q_seq_len head_dim"],
                k: Float16[Array, "heads kv_seq_len head_dim"],
                v: Float16[Array, "heads kv_seq_len head_dim"],
                ):
        return _dense_padded_flash_attention(
            q, k, v, 
            block_sizes=self.block_sizes, 
            kv_padding=self.kv_padding)
        
    def tree_flatten(self):
        """Flattens the PyTree.
        
        Returns:
            A tuple of dynamic children (none) and static auxiliary data.
        """
        # All attributes are static, so they go into aux_data
        aux_data = (self.block_sizes, self.kv_padding)
        # No dynamic children
        children = ()
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs the PyTree from static and dynamic data."""
        # Unpack the static data
        block_sizes, kv_padding = aux_data
        # No dynamic children to unpack
        return cls(block_sizes, kv_padding)

def make_dense_padded_attention(block_sizes: BlockSizes, kv_padding: int):
    return DensePaddedAttention(block_sizes=block_sizes, kv_padding=kv_padding)
        
