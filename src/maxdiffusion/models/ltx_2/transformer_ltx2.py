
from typing import Optional, Tuple, Any, Dict, Union
import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.ltx_2.attention_ltx2 import Attention
from maxdiffusion.models.ltx_2.rope import LTX2AudioVideoRotaryPosEmbed
from maxdiffusion.models.attention_flax import NNXSimpleFeedForward
from maxdiffusion.models.embeddings_flax import NNXPixArtAlphaCombinedTimestepSizeEmbeddings, NNXPixArtAlphaTextProjection
from maxdiffusion.models.gradient_checkpoint import GradientCheckpointType
from maxdiffusion.common_types import BlockSizes

def print_shape(name: str, tensor: Optional[jax.Array]):
    if tensor is not None:
        print(f"MAXDIFF - Shape of {name}: {tensor.shape}")
    else:
        print(f"MAXDIFF - Shape of {name}: None")


class LTX2AdaLayerNormSingle(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, embedding_dim: int, num_mod_params: int = 6, use_additional_conditions: bool = False, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32):
        self.num_mod_params = num_mod_params
        self.use_additional_conditions = use_additional_conditions
        self.emb = NNXPixArtAlphaCombinedTimestepSizeEmbeddings(
            rngs=rngs,
            embedding_dim=embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
            dtype=dtype,
            weights_dtype=weights_dtype
        )
        self.silu = nnx.silu
        self.linear = nnx.Linear(
            rngs=rngs,
            in_features=embedding_dim,
            out_features=num_mod_params * embedding_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=weights_dtype,
            kernel_init=nnx.initializers.zeros, 
            bias_init=nnx.initializers.zeros
        )

    def __call__(
        self, 
        timestep: jax.Array, 
        added_cond_kwargs: Optional[Dict[str, jax.Array]] = None,
        batch_size: Optional[int] = None, # Unused in JAX path usually inferred
        hidden_dtype: Optional[jnp.dtype] = None
    ) -> Tuple[jax.Array, jax.Array]:
        
        resolution = None
        aspect_ratio = None
        if self.use_additional_conditions:
             if added_cond_kwargs is None:
                 raise ValueError("added_cond_kwargs must be provided when use_additional_conditions is True")
             resolution = added_cond_kwargs.get("resolution", None)
             aspect_ratio = added_cond_kwargs.get("aspect_ratio", None)

        embedded_timestep = self.emb(
            timestep, 
            resolution=resolution, 
            aspect_ratio=aspect_ratio, 
            hidden_dtype=hidden_dtype
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class LTX2VideoTransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        activation_fn: str = "gelu",
        qk_norm: str = "rms_norm_across_heads",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        rope_type: str = "interleaved",
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        mesh: jax.sharding.Mesh = None,
        remat_policy: str = "None",
        precision: jax.lax.Precision = None,
        names_which_can_be_saved: list = [],
        names_which_can_be_offloaded: list = [],
    ):
        self.dim = dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        
        # 1. Self-Attention (video and audio)
        self.norm1 = nnx.RMSNorm(self.dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.attn1 = Attention(
            rngs=rngs,
            query_dim=self.dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=0.0,
            use_bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope_type=rope_type,
            dtype=dtype,
            param_dtype=weights_dtype
        )
        
        self.audio_norm1 = nnx.RMSNorm(audio_dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.audio_attn1 = Attention(
            rngs=rngs,
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            dropout=0.0,
            use_bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope_type=rope_type,
            dtype=dtype,
            param_dtype=weights_dtype
        )
        
        # 2. Prompt Cross-Attention
        self.norm2 = nnx.RMSNorm(self.dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.attn2 = Attention(
            rngs=rngs,
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=0.0,
            use_bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope_type=rope_type,
            dtype=dtype,
            param_dtype=weights_dtype
        )
        
        self.audio_norm2 = nnx.RMSNorm(audio_dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.audio_attn2 = Attention(
            rngs=rngs,
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            dropout=0.0,
            use_bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope_type=rope_type,
            dtype=dtype,
            param_dtype=weights_dtype
        )
        
        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        self.audio_to_video_norm = nnx.RMSNorm(dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.audio_to_video_attn = Attention(
            rngs=rngs,
            query_dim=dim,
            cross_attention_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            dropout=0.0,
            use_bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope_type=rope_type,
            dtype=dtype,
            param_dtype=weights_dtype
        )
        
        self.video_to_audio_norm = nnx.RMSNorm(audio_dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.video_to_audio_attn = Attention(
            rngs=rngs,
            query_dim=audio_dim,
            cross_attention_dim=dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            dropout=0.0,
            use_bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope_type=rope_type,
            dtype=dtype,
            param_dtype=weights_dtype
        )
        
        # 4. Feed Forward
        self.norm3 = nnx.RMSNorm(dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.ff = NNXSimpleFeedForward(
            rngs=rngs,
            dim=dim,
            dim_out=dim,
            activation_fn=activation_fn, # Diffusers uses gelu
            dtype=dtype,
            weights_dtype=weights_dtype
        )
        
        self.audio_norm3 = nnx.RMSNorm(audio_dim, epsilon=self.norm_eps, use_scale=self.norm_elementwise_affine, rngs=rngs, dtype=dtype, param_dtype=weights_dtype)
        self.audio_ff = NNXSimpleFeedForward(
             rngs=rngs,
             dim=audio_dim,
             dim_out=audio_dim,
             activation_fn=activation_fn,
             dtype=dtype, 
             weights_dtype=weights_dtype
        )
        

        key = rngs.params()
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.scale_shift_table = nnx.Param(
            jax.random.normal(k1, (6, self.dim), dtype=weights_dtype) / jnp.sqrt(self.dim)
        )
        self.audio_scale_shift_table = nnx.Param(
            jax.random.normal(k2, (6, audio_dim), dtype=weights_dtype) / jnp.sqrt(audio_dim)
        )
        self.video_a2v_cross_attn_scale_shift_table = nnx.Param(
            jax.random.normal(k3, (5, self.dim), dtype=weights_dtype)
        )
        self.audio_a2v_cross_attn_scale_shift_table = nnx.Param(
            jax.random.normal(k4, (5, audio_dim), dtype=weights_dtype)
        )

    def __call__(
        self,
        hidden_states: jax.Array,  # Video
        audio_hidden_states: jax.Array,  # Audio
        encoder_hidden_states: jax.Array,  # Context (Text)
        audio_encoder_hidden_states: jax.Array,  # Audio Context

        # Timestep embeddings for AdaLN
        temb: jax.Array,
        temb_audio: jax.Array,
        temb_ca_scale_shift: jax.Array,
        temb_ca_audio_scale_shift: jax.Array,
        temb_ca_gate: jax.Array,
        temb_ca_audio_gate: jax.Array,

        # RoPE
        video_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None,
        audio_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None,
        ca_video_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None,
        ca_audio_rotary_emb: Optional[Tuple[jax.Array, jax.Array]] = None,

        attention_mask: Optional[jax.Array] = None, # Unused in the provided forward pass
        encoder_attention_mask: Optional[jax.Array] = None,
        audio_encoder_attention_mask: Optional[jax.Array] = None,
        a2v_cross_attention_mask: Optional[jax.Array] = None,
        v2a_cross_attention_mask: Optional[jax.Array] = None,

    ) -> Tuple[jax.Array, jax.Array]:
        batch_size = hidden_states.shape[0]
        print_shape("Block Input hidden_states", hidden_states)
        print_shape("Block Input audio_hidden_states", audio_hidden_states)
        print_shape("Block Input encoder_hidden_states", encoder_hidden_states)
        print_shape("Block Input audio_encoder_hidden_states", audio_encoder_hidden_states)
        print_shape("Block Input temb", temb)
        print_shape("Block Input temb_audio", temb_audio)
        print_shape("Block Input temb_ca_scale_shift", temb_ca_scale_shift)
        print_shape("Block Input temb_ca_audio_scale_shift", temb_ca_audio_scale_shift)
        print_shape("Block Input temb_ca_gate", temb_ca_gate)
        print_shape("Block Input temb_ca_audio_gate", temb_ca_audio_gate)

        # 1. Video and Audio Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        print_shape("norm1 hidden_states", norm_hidden_states)

        # Calculate Video AdaLN values
        num_ada_params = self.scale_shift_table.shape[0]
        # table shape: (6, dim) -> (1, 1, 6, dim)
        scale_shift_table_reshaped = jnp.expand_dims(self.scale_shift_table, axis=(0, 1))
        # temb shape: (batch, temb_dim) -> (batch, 1, 6, dim)  (assuming temb_dim is num_ada_params * dim)
        temb_reshaped = temb.reshape(batch_size, 1, num_ada_params, -1)
        ada_values = scale_shift_table_reshaped + temb_reshaped
        print_shape("ada_values", ada_values)

        shift_msa = ada_values[:, :, 0, :]
        scale_msa = ada_values[:, :, 1, :]
        gate_msa = ada_values[:, :, 2, :]
        shift_mlp = ada_values[:, :, 3, :]
        scale_mlp = ada_values[:, :, 4, :]
        gate_mlp = ada_values[:, :, 5, :]
        print_shape("shift_msa", shift_msa)
        print_shape("scale_msa", scale_msa)
        print_shape("gate_msa", gate_msa)
        print_shape("shift_mlp", shift_mlp)
        print_shape("scale_mlp", scale_mlp)
        print_shape("gate_mlp", gate_mlp)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        print_shape("mod_norm_hidden_states (Self-Attn)", norm_hidden_states)

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=video_rotary_emb,
        )
        print_shape("attn1 output", attn_hidden_states)
        hidden_states = hidden_states + attn_hidden_states * gate_msa
        print_shape("hidden_states after attn1", hidden_states)

        # Calculate Audio AdaLN values
        norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)
        print_shape("audio_norm1 audio_hidden_states", norm_audio_hidden_states)

        num_audio_ada_params = self.audio_scale_shift_table.shape[0]
        audio_scale_shift_table_reshaped = jnp.expand_dims(self.audio_scale_shift_table, axis=(0, 1))
        temb_audio_reshaped = temb_audio.reshape(batch_size, 1, num_audio_ada_params, -1)
        audio_ada_values = audio_scale_shift_table_reshaped + temb_audio_reshaped
        print_shape("audio_ada_values", audio_ada_values)

        audio_shift_msa = audio_ada_values[:, :, 0, :]
        audio_scale_msa = audio_ada_values[:, :, 1, :]
        audio_gate_msa = audio_ada_values[:, :, 2, :]
        audio_shift_mlp = audio_ada_values[:, :, 3, :]
        audio_scale_mlp = audio_ada_values[:, :, 4, :]
        audio_gate_mlp = audio_ada_values[:, :, 5, :]
        print_shape("audio_shift_msa", audio_shift_msa)
        print_shape("audio_scale_msa", audio_scale_msa)
        print_shape("audio_gate_msa", audio_gate_msa)
        print_shape("audio_shift_mlp", audio_shift_mlp)
        print_shape("audio_scale_mlp", audio_scale_mlp)
        print_shape("audio_gate_mlp", audio_gate_mlp)

        norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa
        print_shape("mod_norm_audio_hidden_states (Self-Attn)", norm_audio_hidden_states)

        attn_audio_hidden_states = self.audio_attn1(
            hidden_states=norm_audio_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=audio_rotary_emb,
        )
        print_shape("audio_attn1 output", attn_audio_hidden_states)
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * audio_gate_msa
        print_shape("audio_hidden_states after audio_attn1", audio_hidden_states)

        # 2. Video and Audio Cross-Attention with the text embeddings
        norm_hidden_states = self.norm2(hidden_states)
        print_shape("norm2 hidden_states", norm_hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        print_shape("attn2 output", attn_hidden_states)
        hidden_states = hidden_states + attn_hidden_states
        print_shape("hidden_states after attn2", hidden_states)

        norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
        print_shape("norm2 audio_hidden_states", norm_audio_hidden_states)
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=audio_encoder_attention_mask,
        )
        print_shape("attn2 audio output", attn_audio_hidden_states)
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states
        print_shape("audio_hidden_states after attn2 audio", audio_hidden_states)

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        norm_hidden_states = self.audio_to_video_norm(hidden_states)
        print_shape("audio_to_video_norm hidden_states", norm_hidden_states)
        norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)
        print_shape("video_to_audio_norm audio_hidden_states", norm_audio_hidden_states)

        # Calculate Cross-Attention Modulation values
        # Video
        video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
        video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]

        # table: (4, dim) -> (1, 1, 4, dim)
        video_ca_scale_shift_table = jnp.expand_dims(video_per_layer_ca_scale_shift, axis=(0, 1)) + \
                                    temb_ca_scale_shift.reshape(batch_size, 1, 4, -1)
        print_shape("video_ca_scale_shift_table", video_ca_scale_shift_table)

        video_a2v_ca_scale = video_ca_scale_shift_table[:, :, 0, :]
        video_a2v_ca_shift = video_ca_scale_shift_table[:, :, 1, :]
        video_v2a_ca_scale = video_ca_scale_shift_table[:, :, 2, :]
        video_v2a_ca_shift = video_ca_scale_shift_table[:, :, 3, :]
        print_shape("video_a2v_ca_scale", video_a2v_ca_scale)
        print_shape("video_a2v_ca_shift", video_a2v_ca_shift)
        print_shape("video_v2a_ca_scale", video_v2a_ca_scale)
        print_shape("video_v2a_ca_shift", video_v2a_ca_shift)


        # table: (1, dim) -> (1, 1, 1, dim)
        a2v_gate = (jnp.expand_dims(video_per_layer_ca_gate, axis=(0, 1)) + \
                    temb_ca_gate.reshape(batch_size, 1, 1, -1))[:, :, 0, :]
        print_shape("a2v_gate", a2v_gate)

        # Audio
        audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
        audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]

        audio_ca_scale_shift_table = (jnp.expand_dims(audio_per_layer_ca_scale_shift, axis=(0, 1)) + \
                                    temb_ca_audio_scale_shift.reshape(batch_size, 1, 4, -1))
        print_shape("audio_ca_scale_shift_table", audio_ca_scale_shift_table)

        audio_a2v_ca_scale = audio_ca_scale_shift_table[:, :, 0, :]
        audio_a2v_ca_shift = audio_ca_scale_shift_table[:, :, 1, :]
        audio_v2a_ca_scale = audio_ca_scale_shift_table[:, :, 2, :]
        audio_v2a_ca_shift = audio_ca_scale_shift_table[:, :, 3, :]
        print_shape("audio_a2v_ca_scale", audio_a2v_ca_scale)
        print_shape("audio_a2v_ca_shift", audio_a2v_ca_shift)
        print_shape("audio_v2a_ca_scale", audio_v2a_ca_scale)
        print_shape("audio_v2a_ca_shift", audio_v2a_ca_shift)

        v2a_gate = (jnp.expand_dims(audio_per_layer_ca_gate, axis=(0, 1)) + \
                    temb_ca_audio_gate.reshape(batch_size, 1, 1, -1))[:, :, 0, :]
        print_shape("v2a_gate", v2a_gate)

        # Audio-to-Video Cross Attention: Q: Video; K,V: Audio
        mod_norm_hidden_states = norm_hidden_states * (1 + video_a2v_ca_scale) + video_a2v_ca_shift
        print_shape("mod_norm_hidden_states_a2v", mod_norm_hidden_states)
        mod_norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_a2v_ca_scale) + audio_a2v_ca_shift
        print_shape("mod_norm_audio_hidden_states_a2v", mod_norm_audio_hidden_states)

        a2v_attn_hidden_states = self.audio_to_video_attn(
            mod_norm_hidden_states,
            encoder_hidden_states=mod_norm_audio_hidden_states,
            query_rotary_emb=ca_video_rotary_emb,
            key_rotary_emb=ca_audio_rotary_emb,
            attention_mask=a2v_cross_attention_mask,
        )
        print_shape("a2v_attn_hidden_states", a2v_attn_hidden_states)
        hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states
        print_shape("hidden_states after a2v", hidden_states)

        # Video-to-Audio Cross Attention: Q: Audio; K,V: Video
        mod_norm_hidden_states_v2a = norm_hidden_states * (1 + video_v2a_ca_scale) + video_v2a_ca_shift
        print_shape("mod_norm_hidden_states_v2a", mod_norm_hidden_states_v2a)
        mod_norm_audio_hidden_states_v2a = norm_audio_hidden_states * (1 + audio_v2a_ca_scale) + audio_v2a_ca_shift
        print_shape("mod_norm_audio_hidden_states_v2a", mod_norm_audio_hidden_states_v2a)

        v2a_attn_hidden_states = self.video_to_audio_attn(
            mod_norm_audio_hidden_states_v2a,
            encoder_hidden_states=mod_norm_hidden_states_v2a,
            query_rotary_emb=ca_audio_rotary_emb,
            key_rotary_emb=ca_video_rotary_emb,
            attention_mask=v2a_cross_attention_mask,
        )
        print_shape("v2a_attn_hidden_states", v2a_attn_hidden_states)
        audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states
        print_shape("audio_hidden_states after v2a", audio_hidden_states)


        # 4. Feedforward
        norm_hidden_states = self.norm3(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        print_shape("mod_norm_hidden_states (FF)", norm_hidden_states)
        ff_output = self.ff(norm_hidden_states)
        print_shape("ff_output", ff_output)
        hidden_states = hidden_states + ff_output * gate_mlp
        print_shape("hidden_states after ff", hidden_states)

        norm_audio_hidden_states = self.audio_norm3(audio_hidden_states)
        norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_mlp) + audio_shift_mlp
        print_shape("mod_norm_audio_hidden_states (FF)", norm_audio_hidden_states)
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        print_shape("audio_ff_output", audio_ff_output)
        audio_hidden_states = audio_hidden_states + audio_ff_output * audio_gate_mlp
        print_shape("audio_hidden_states after ff", audio_hidden_states)

        print_shape("Block Output hidden_states", hidden_states)
        print_shape("Block Output audio_hidden_states", audio_hidden_states)

        return hidden_states, audio_hidden_states


class LTX2VideoTransformer3DModel(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        in_channels: int = 128,  # Video Arguments
        out_channels: Optional[int] = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: Optional[int] = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        mesh: jax.sharding.Mesh = None,
        remat_policy: str = "None",
        precision: jax.lax.Precision = None,
        names_which_can_be_saved: list = [],
        names_which_can_be_offloaded: list = [],
        scan_layers: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.vae_scale_factors = vae_scale_factors
        self.pos_embed_max_pos = pos_embed_max_pos
        self.base_height = base_height
        self.base_width = base_width
        self.audio_in_channels = audio_in_channels
        self.audio_out_channels = audio_out_channels
        self.audio_patch_size = audio_patch_size
        self.audio_patch_size_t = audio_patch_size_t
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_attention_head_dim = audio_attention_head_dim
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.audio_scale_factor = audio_scale_factor
        self.audio_pos_embed_max_pos = audio_pos_embed_max_pos
        self.audio_sampling_rate = audio_sampling_rate
        self.audio_hop_length = audio_hop_length
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.qk_norm = qk_norm
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.caption_channels = caption_channels
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.rope_theta = rope_theta
        self.rope_double_precision = rope_double_precision
        self.causal_offset = causal_offset
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.cross_attn_timestep_scale_multiplier = cross_attn_timestep_scale_multiplier
        self.rope_type = rope_type
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.mesh = mesh
        self.remat_policy = remat_policy
        self.precision = precision
        self.names_which_can_be_saved = names_which_can_be_saved
        self.names_which_can_be_offloaded = names_which_can_be_offloaded
        self.scan_layers = scan_layers

        _out_channels = self.out_channels or self.in_channels
        _audio_out_channels = self.audio_out_channels or self.audio_in_channels
        inner_dim = self.num_attention_heads * self.attention_head_dim
        audio_inner_dim = self.audio_num_attention_heads * self.audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nnx.Linear(
            self.in_channels,
            inner_dim,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",))
        )
        self.audio_proj_in = nnx.Linear(
            self.audio_in_channels,
            audio_inner_dim,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",))
        )

        # 2. Prompt embeddings
        self.caption_projection = NNXPixArtAlphaTextProjection(
            rngs=rngs, in_features=self.caption_channels, hidden_size=inner_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        self.audio_caption_projection = NNXPixArtAlphaTextProjection(
            rngs=rngs, in_features=self.caption_channels, hidden_size=audio_inner_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        # 3. Timestep Modulation Params and Embedding
        self.time_embed = LTX2AdaLayerNormSingle(
            rngs=rngs, embedding_dim=inner_dim, num_mod_params=6, use_additional_conditions=False, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            rngs=rngs, embedding_dim=audio_inner_dim, num_mod_params=6, use_additional_conditions=False, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
            rngs=rngs, embedding_dim=inner_dim, num_mod_params=4, use_additional_conditions=False, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
            rngs=rngs, embedding_dim=audio_inner_dim, num_mod_params=4, use_additional_conditions=False, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
            rngs=rngs, embedding_dim=inner_dim, num_mod_params=1, use_additional_conditions=False, dtype=self.dtype, weights_dtype=self.weights_dtype
        )
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
            rngs=rngs, embedding_dim=audio_inner_dim, num_mod_params=1, use_additional_conditions=False, dtype=self.dtype, weights_dtype=self.weights_dtype
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        param_rng = rngs.params()
        self.scale_shift_table = nnx.Param(
            jax.random.normal(param_rng, (2, inner_dim), dtype=self.weights_dtype) / jnp.sqrt(inner_dim),
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "embed"))
        )
        self.audio_scale_shift_table = nnx.Param(
            jax.random.normal(param_rng, (2, audio_inner_dim), dtype=self.weights_dtype) / jnp.sqrt(audio_inner_dim),
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "embed"))
        )

        # 4. Rotary Positional Embeddings (RoPE)
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            base_num_frames=self.pos_embed_max_pos,
            base_height=self.base_height,
            base_width=self.base_width,
            scale_factors=self.vae_scale_factors,
            theta=self.rope_theta,
            causal_offset=self.causal_offset,
            modality="video",
            double_precision=self.rope_double_precision,
            rope_type=self.rope_type,
            num_attention_heads=self.num_attention_heads,
            dtype=self.dtype,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim, # Per head dim
            patch_size=self.audio_patch_size,
            patch_size_t=self.audio_patch_size_t,
            base_num_frames=self.audio_pos_embed_max_pos,
            sampling_rate=self.audio_sampling_rate,
            hop_length=self.audio_hop_length,
            scale_factors=[self.audio_scale_factor],
            theta=self.rope_theta,
            causal_offset=self.causal_offset,
            modality="audio",
            double_precision=self.rope_double_precision,
            rope_type=self.rope_type,
            num_attention_heads=self.audio_num_attention_heads,
            dtype=self.dtype,
        )

        cross_attn_pos_embed_max_pos = max(self.pos_embed_max_pos, self.audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.audio_cross_attention_dim,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=self.base_height,
            base_width=self.base_width,
            theta=self.rope_theta,
            causal_offset=self.causal_offset,
            modality="video",
            double_precision=self.rope_double_precision,
            rope_type=self.rope_type,
            num_attention_heads=self.num_attention_heads,
             dtype=self.dtype,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.audio_cross_attention_dim,
            patch_size=self.audio_patch_size,
            patch_size_t=self.audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=self.audio_sampling_rate,
            hop_length=self.audio_hop_length,
            theta=self.rope_theta,
            causal_offset=self.causal_offset,
            modality="audio",
            double_precision=self.rope_double_precision,
            rope_type=self.rope_type,
            num_attention_heads=self.audio_num_attention_heads,
             dtype=self.dtype,
        )

        # 5. Transformer Blocks
        # 5. Transformer Blocks
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.vmap(in_axes=None, out_axes=0, axis_size=self.num_layers, transform_metadata={nnx.PARTITION_NAME: "layers"})
        def init_block(rngs):
            return LTX2VideoTransformerBlock(
                rngs=rngs,
                dim=inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
                cross_attention_dim=self.cross_attention_dim,
                audio_dim=audio_inner_dim,
                audio_num_attention_heads=self.audio_num_attention_heads,
                audio_attention_head_dim=self.audio_attention_head_dim,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                activation_fn=self.activation_fn,
                qk_norm=self.qk_norm,
                attention_bias=self.attention_bias,
                attention_out_bias=self.attention_out_bias,
                norm_elementwise_affine=self.norm_elementwise_affine,
                norm_eps=self.norm_eps,
                rope_type=self.rope_type,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                mesh=self.mesh,
                remat_policy=self.remat_policy,
                precision=self.precision,
                names_which_can_be_saved=self.names_which_can_be_saved,
                names_which_can_be_offloaded=self.names_which_can_be_offloaded,
            )

        if self.scan_layers:
            self.transformer_blocks = init_block(rngs)
        else:
            blocks = []
            for _ in range(self.num_layers):
                block = LTX2VideoTransformerBlock(
                    rngs=rngs,
                    dim=inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    cross_attention_dim=self.cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=self.audio_num_attention_heads,
                    audio_attention_head_dim=self.audio_attention_head_dim,
                    audio_cross_attention_dim=self.audio_cross_attention_dim,
                    activation_fn=self.activation_fn,
                    qk_norm=self.qk_norm,
                    attention_bias=self.attention_bias,
                    attention_out_bias=self.attention_out_bias,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                    rope_type=self.rope_type,
                    dtype=self.dtype,
                    weights_dtype=self.weights_dtype,
                    mesh=self.mesh,
                    remat_policy=self.remat_policy,
                    precision=self.precision,
                    names_which_can_be_saved=self.names_which_can_be_saved,
                    names_which_can_be_offloaded=self.names_which_can_be_offloaded,
                )
                blocks.append(block)
            self.transformer_blocks = nnx.data(blocks)

        # 6. Output layers
        self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
        self.norm_out = nnx.LayerNorm(inner_dim, epsilon=1e-6, use_scale=False, rngs=rngs, dtype=self.dtype, param_dtype=self.weights_dtype)
        self.proj_out = nnx.Linear(
            inner_dim,
            _out_channels,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",))
        )

        self.audio_norm_out = nnx.LayerNorm(audio_inner_dim, epsilon=1e-6, use_scale=False, rngs=rngs, dtype=self.dtype, param_dtype=self.weights_dtype)
        self.audio_proj_out = nnx.Linear(
            audio_inner_dim,
            _audio_out_channels,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",))
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        audio_hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        audio_encoder_hidden_states: jax.Array,
        timestep: jax.Array,
        audio_timestep: Optional[jax.Array] = None,
        encoder_attention_mask: Optional[jax.Array] = None,
        audio_encoder_attention_mask: Optional[jax.Array] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[jax.Array] = None,
        audio_coords: Optional[jax.Array] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Any: # Should be AudioVisualModelOutput or Tuple

        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        if encoder_attention_mask is not None: print_shape("encoder_attention_mask input", encoder_attention_mask)
        if audio_encoder_attention_mask is not None: print_shape("audio_encoder_attention_mask input", audio_encoder_attention_mask)


        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.astype(self.dtype)) * -10000.0
            encoder_attention_mask = jnp.expand_dims(encoder_attention_mask, axis=1)
        if encoder_attention_mask is not None: print_shape("encoder_attention_mask bias", encoder_attention_mask)

        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.astype(self.dtype)) * -10000.0
            audio_encoder_attention_mask = jnp.expand_dims(audio_encoder_attention_mask, axis=1)
        if audio_encoder_attention_mask is not None: print_shape("audio_encoder_attention_mask bias", audio_encoder_attention_mask)

        batch_size = hidden_states.shape[0]
        print_shape("Model Input hidden_states", hidden_states)
        print_shape("Model Input audio_hidden_states", audio_hidden_states)
        print_shape("Model Input encoder_hidden_states", encoder_hidden_states)
        print_shape("Model Input audio_encoder_hidden_states", audio_encoder_hidden_states)
        print_shape("Model Input timestep", timestep)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, fps=fps
            )
            print_shape("video_coords", video_coords)
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames
            )
            print_shape("audio_coords", audio_coords)

        video_rotary_emb = self.rope(video_coords)
        if video_rotary_emb: print_shape("video_rotary_emb[0]", video_rotary_emb[0]); print_shape("video_rotary_emb[1]", video_rotary_emb[1])
        audio_rotary_emb = self.audio_rope(audio_coords)
        if audio_rotary_emb: print_shape("audio_rotary_emb[0]", audio_rotary_emb[0]); print_shape("audio_rotary_emb[1]", audio_rotary_emb[1])

        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :])
        if video_cross_attn_rotary_emb: print_shape("video_cross_attn_rotary_emb[0]", video_cross_attn_rotary_emb[0]); print_shape("video_cross_attn_rotary_emb[1]", video_cross_attn_rotary_emb[1])
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(audio_coords[:, 0:1, :])
        if audio_cross_attn_rotary_emb: print_shape("audio_cross_attn_rotary_emb[0]", audio_cross_attn_rotary_emb[0]); print_shape("audio_cross_attn_rotary_emb[1]", audio_cross_attn_rotary_emb[1])

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        print_shape("hidden_states after proj_in", hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)
        print_shape("audio_hidden_states after audio_proj_in", audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.cross_attn_timestep_scale_multiplier / self.timestep_scale_multiplier
        )

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            hidden_dtype=hidden_states.dtype,
        )
        print_shape("temb flat", temb)
        print_shape("embedded_timestep flat", embedded_timestep)
        temb = temb.reshape(batch_size, -1, temb.shape[-1])
        embedded_timestep = embedded_timestep.reshape(batch_size, -1, embedded_timestep.shape[-1])
        print_shape("temb reshaped", temb)
        print_shape("embedded_timestep reshaped", embedded_timestep)

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            hidden_dtype=audio_hidden_states.dtype,
        )
        print_shape("temb_audio flat", temb_audio)
        print_shape("audio_embedded_timestep flat", audio_embedded_timestep)
        temb_audio = temb_audio.reshape(batch_size, -1, temb_audio.shape[-1])
        audio_embedded_timestep = audio_embedded_timestep.reshape(batch_size, -1, audio_embedded_timestep.shape[-1])
        print_shape("temb_audio reshaped", temb_audio)
        print_shape("audio_embedded_timestep reshaped", audio_embedded_timestep)

        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            timestep.flatten(),
            hidden_dtype=hidden_states.dtype,
        )
        print_shape("video_cross_attn_scale_shift flat", video_cross_attn_scale_shift)
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            timestep.flatten() * timestep_cross_attn_gate_scale_factor,
            hidden_dtype=hidden_states.dtype,
        )
        print_shape("video_cross_attn_a2v_gate flat", video_cross_attn_a2v_gate)
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.reshape(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        print_shape("video_cross_attn_scale_shift reshaped", video_cross_attn_scale_shift)
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.reshape(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])
        print_shape("video_cross_attn_a2v_gate reshaped", video_cross_attn_a2v_gate)

        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_timestep.flatten(),
            hidden_dtype=audio_hidden_states.dtype,
        )
        print_shape("audio_cross_attn_scale_shift flat", audio_cross_attn_scale_shift)
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_timestep.flatten() * timestep_cross_attn_gate_scale_factor,
            hidden_dtype=audio_hidden_states.dtype,
        )
        print_shape("audio_cross_attn_v2a_gate flat", audio_cross_attn_v2a_gate)
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.reshape(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        print_shape("audio_cross_attn_scale_shift reshaped", audio_cross_attn_scale_shift)
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.reshape(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])
        print_shape("audio_cross_attn_v2a_gate reshaped", audio_cross_attn_v2a_gate)

        # 4. Prepare prompt embeddings
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        print_shape("encoder_hidden_states pre reshape", encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.reshape(batch_size, -1, hidden_states.shape[-1])
        print_shape("encoder_hidden_states after projection", encoder_hidden_states)

        audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
        print_shape("audio_encoder_hidden_states pre reshape", audio_encoder_hidden_states)
        audio_encoder_hidden_states = audio_encoder_hidden_states.reshape(batch_size, -1, audio_hidden_states.shape[-1])
        print_shape("audio_encoder_hidden_states after projection", audio_encoder_hidden_states)

        # 5. Run transformer blocks
        def scan_fn(carry, block):
            hidden_states, audio_hidden_states, rngs_carry = carry
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=video_cross_attn_rotary_emb,
                ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )
            return (hidden_states, audio_hidden_states, rngs_carry), None

        if self.scan_layers:
            rematted_scan_fn = self.gradient_checkpoint.apply(
                scan_fn, self.names_which_can_be_saved, self.names_which_can_be_offloaded, prevent_cse=not self.scan_layers
            )
            carry = (hidden_states, audio_hidden_states, nnx.Rngs(0)) # Placeholder RNGs for now if not used in block
            (hidden_states, audio_hidden_states, _), _ = nnx.scan(
                rematted_scan_fn,
                length=self.num_layers,
                in_axes=(nnx.Carry, 0),
                out_axes=(nnx.Carry, 0),
                transform_metadata={nnx.PARTITION_NAME: "layers"}
            )(carry, self.transformer_blocks)
        else:
             for block in self.transformer_blocks:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                )
        print_shape("Model hidden_states after blocks", hidden_states)
        print_shape("Model audio_hidden_states after blocks", audio_hidden_states)

        # 6. Output layers
        scale_shift_values = jnp.expand_dims(self.scale_shift_table, axis=(0, 1)) + jnp.expand_dims(embedded_timestep, axis=2)
        print_shape("Output scale_shift_values", scale_shift_values)
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        print_shape("Output shift", shift)
        print_shape("Output scale", scale)

        hidden_states = self.norm_out(hidden_states)
        print_shape("hidden_states after norm_out", hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        print_shape("hidden_states after output mod", hidden_states)
        output = self.proj_out(hidden_states)
        print_shape("Model Output sample", output)

        audio_scale_shift_values = jnp.expand_dims(self.audio_scale_shift_table, axis=(0, 1)) + jnp.expand_dims(audio_embedded_timestep, axis=2)
        print_shape("Audio Output scale_shift_values", audio_scale_shift_values)
        audio_shift = audio_scale_shift_values[:, :, 0, :]
        audio_scale = audio_scale_shift_values[:, :, 1, :]
        print_shape("Audio Output shift", audio_shift)
        print_shape("Audio Output scale", audio_scale)

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        print_shape("audio_hidden_states after norm_out", audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        print_shape("audio_hidden_states after output mod", audio_hidden_states)
        audio_output = self.audio_proj_out(audio_hidden_states)
        print_shape("Model Output audio_sample", audio_output)

        if not return_dict:
            return (output, audio_output)
        # return AudioVisualModelOutput(sample=output, audio_sample=audio_output)
        return {"sample": output, "audio_sample": audio_output} # Placeholder
