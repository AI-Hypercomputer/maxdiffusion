# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from maxdiffusion.models.attention_flax import FlaxFluxAttention as FluxAttention
from maxdiffusion.models.embeddings_flax import FluxPosEmbed, CombinedTimestepGuidanceTextProjEmbeddings as CombinedTimestepGuidanceTextEmbeddings
from maxdiffusion.models.normalization_flax import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from maxdiffusion.models.modeling_flax_utils import FlaxModelMixin
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxSingleTransformerBlock
from maxdiffusion.configuration_utils import ConfigMixin, flax_register_to_config
from maxdiffusion.utils import BaseOutput


@flax.struct.dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: jnp.ndarray


class FlaxSwiGluFeedForward(nn.Module):
    dim: int
    hidden_dim: int
    out_dim: int
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: float = None

    def setup(self):
        self.linear_in = nn.Dense(
            2 * self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,
        )
        self.linear_out = nn.Dense(
            self.out_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,
        )

    def __call__(self, x):
        x = self.linear_in(x)
        x1, x2 = jnp.split(x, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        return self.linear_out(hidden)


class Flux2KleinSingleTransformerBlock(nn.Module):
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    attention_kernel: str = "dot_product"
    flash_min_seq_length: int = 512
    flash_block_sizes: Optional[Dict[str, int]] = None
    mesh: Optional[jax.sharding.Mesh] = None
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: float = None
    mlp_ratio: float = 4.0
    use_global_modulation: bool = False

    def setup(self):
        if self.use_global_modulation:
            self.norm = nn.LayerNorm(
                use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
            )
        else:
            self.norm = AdaLayerNormZeroSingle(
                self.dim,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
            )
        self.attn = FluxAttention(
            query_dim=self.dim,
            heads=self.num_attention_heads,
            dim_head=self.attention_head_dim,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
        )
        self.mlp = FlaxSwiGluFeedForward(
            self.dim,
            int(self.dim * self.mlp_ratio),
            self.dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
        )

    def __call__(self, hidden_states, temb, image_rotary_emb=None, temb_mod=None):
        if self.use_global_modulation:
            shift, scale, gate = jnp.split(temb_mod, 3, axis=-1)
            shift = jnp.expand_dims(shift, axis=1)
            scale = jnp.expand_dims(scale, axis=1)
            gate = jnp.expand_dims(gate, axis=1)
            norm_hidden_states = self.norm(hidden_states) * (1.0 + scale) + shift
        else:
            norm_hidden_states, gate = self.norm(hidden_states, temb)

        attn_output, _ = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        mlp_output = self.mlp(norm_hidden_states)
        hidden_states = hidden_states + gate * (attn_output + mlp_output)

        return hidden_states


class Flux2KleinTransformerBlock(nn.Module):
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    attention_kernel: str = "dot_product"
    flash_min_seq_length: int = 512
    flash_block_sizes: Optional[Dict[str, int]] = None
    mesh: Optional[jax.sharding.Mesh] = None
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: float = None
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    use_global_modulation: bool = False

    def setup(self):
        if self.use_global_modulation:
            self.norm1 = nn.LayerNorm(
                use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
            )
            self.norm1_context = nn.LayerNorm(
                use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
            )
            self.norm2 = nn.LayerNorm(
                use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
            )
            self.norm2_context = nn.LayerNorm(
                use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
            )
        else:
            self.norm1 = AdaLayerNormZero(
                self.dim,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
            )
            self.norm1_context = AdaLayerNormZero(
                self.dim,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
            )
        self.attn = FluxAttention(
            query_dim=self.dim,
            heads=self.num_attention_heads,
            dim_head=self.attention_head_dim,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
            qkv_bias=self.qkv_bias,
        )
        self.ff = FlaxSwiGluFeedForward(
            self.dim,
            int(self.dim * self.mlp_ratio),
            self.dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
        )
        self.ff_context = FlaxSwiGluFeedForward(
            self.dim,
            int(self.dim * self.mlp_ratio),
            self.dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
        )

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb=None,
        temb_mod_img=None,
        temb_mod_txt=None,
    ):
        if self.use_global_modulation:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(temb_mod_img, 6, axis=-1)
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = jnp.split(temb_mod_txt, 6, axis=-1)

            shift_msa = jnp.expand_dims(shift_msa, axis=1)
            scale_msa = jnp.expand_dims(scale_msa, axis=1)
            gate_msa = jnp.expand_dims(gate_msa, axis=1)
            shift_mlp = jnp.expand_dims(shift_mlp, axis=1)
            scale_mlp = jnp.expand_dims(scale_mlp, axis=1)
            gate_mlp = jnp.expand_dims(gate_mlp, axis=1)

            c_shift_msa = jnp.expand_dims(c_shift_msa, axis=1)
            c_scale_msa = jnp.expand_dims(c_scale_msa, axis=1)
            c_gate_msa = jnp.expand_dims(c_gate_msa, axis=1)
            c_shift_mlp = jnp.expand_dims(c_shift_mlp, axis=1)
            c_scale_mlp = jnp.expand_dims(c_scale_mlp, axis=1)
            c_gate_mlp = jnp.expand_dims(c_gate_mlp, axis=1)

            norm_hidden_states = self.norm1(hidden_states) * (1.0 + scale_msa) + shift_msa
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states) * (1.0 + c_scale_msa) + c_shift_msa
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, temb
            )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output

        if self.use_global_modulation:
            norm_hidden_states = self.norm2(hidden_states) * (1.0 + scale_mlp) + shift_mlp
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (1.0 + c_shift_mlp) + c_scale_mlp
        else:
            norm_hidden_states, gate_mlp, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            norm_encoder_hidden_states, c_gate_mlp, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, temb
            )

        mlp_output = self.ff(norm_hidden_states)
        encoder_mlp_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp * mlp_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * encoder_mlp_output

        return hidden_states, encoder_hidden_states


@flax_register_to_config
class Flux2KleinTransformer2DModel(nn.Module, FlaxModelMixin, ConfigMixin):
    patch_size: int = 1
    in_channels: int = 64
    num_layers: int = 5
    num_single_layers: int = 20
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = True
    axes_dim: Tuple[int, ...] = (16, 56, 56)
    theta: int = 10000
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    use_global_modulation: bool = True
    scale_shift_order: str = "shift_scale"
    proj_out_bias: bool = True
    joint_attention_bias: bool = False
    x_embedder_bias: bool = False
    use_swiglu: bool = True
    axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32)
    attention_kernel: str = "dot_product"
    flash_min_seq_length: int = 512
    flash_block_sizes: Optional[Dict[str, int]] = None
    mesh: Optional[jax.sharding.Mesh] = None
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: float = None

    def setup(self):
        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        self.time_text_embed = CombinedTimestepGuidanceTextEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.pooled_projection_dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
        )

        if self.use_global_modulation:
            self.double_stream_modulation_img = nn.Dense(
                6 * self.inner_dim,
                dtype=self.dtype,
                param_dtype=self.weights_dtype,
                precision=self.precision,
            )
            self.double_stream_modulation_txt = nn.Dense(
                6 * self.inner_dim,
                dtype=self.dtype,
                param_dtype=self.weights_dtype,
                precision=self.precision,
            )
            self.single_stream_modulation = nn.Dense(
                3 * self.inner_dim,
                dtype=self.dtype,
                param_dtype=self.weights_dtype,
                precision=self.precision,
            )

        self.context_embedder = nn.Dense(
            self.inner_dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,
        )
        self.x_embedder = nn.Dense(
            self.inner_dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,
        )

        self.pos_embed = FluxPosEmbed(
            theta=self.theta,
            axes_dim=self.axes_dim,
            return_tuple=True,
        )

        double_blocks = []
        for _ in range(self.num_layers):
            double_block = Flux2KleinTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
                attention_kernel=self.attention_kernel,
                flash_min_seq_length=self.flash_min_seq_length,
                flash_block_sizes=self.flash_block_sizes,
                mesh=self.mesh,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                use_global_modulation=self.use_global_modulation,
            )
            double_blocks.append(double_block)
        self.double_blocks = double_blocks

        single_blocks = []
        for _ in range(self.num_single_layers):
            single_block = FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
                attention_kernel=self.attention_kernel,
                flash_min_seq_length=self.flash_min_seq_length,
                flash_block_sizes=self.flash_block_sizes,
                mesh=self.mesh,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
                mlp_ratio=self.mlp_ratio,
                use_global_modulation=self.use_global_modulation,
                use_swiglu=True,
            )
            single_blocks.append(single_block)
        self.single_blocks = single_blocks

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
            scale_shift_order=self.scale_shift_order,
        )

        self.proj_out = nn.Dense(
            self.in_channels,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,
            use_bias=self.proj_out_bias,
        )

    def timestep_embedding(self, t: jax.Array, dim: int, max_period=10000, time_factor: float = 1.0) -> jax.Array:
        t = time_factor * t
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        return_dict: bool = True,
        train: bool = False,
        return_intermediates: bool = False,
    ):
        hidden_states = self.x_embedder(hidden_states)
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        temb = temb.astype(hidden_states.dtype)

        if self.use_global_modulation:
            temb_silu = nn.silu(temb)
            double_stream_mod_img = self.double_stream_modulation_img(temb_silu)
            double_stream_mod_txt = self.double_stream_modulation_txt(temb_silu)
            single_stream_mod = self.single_stream_modulation(temb_silu)
        else:
            double_stream_mod_img, double_stream_mod_txt, single_stream_mod = None, None, None

        if encoder_hidden_states is not None and hasattr(self, "context_embedder") and self.context_embedder is not None:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = jnp.concatenate((txt_ids, img_ids), axis=0)
        image_rotary_emb = self.pos_embed(ids)

        intermediates = {}
        if return_intermediates:
            intermediates["temb"] = temb
            intermediates["global_modulation"] = (double_stream_mod_img, double_stream_mod_txt, single_stream_mod)
            intermediates["double_block_inputs"] = []
            intermediates["double_block_outputs"] = []
            intermediates["single_block_outputs"] = []

        for double_block in self.double_blocks:
            if return_intermediates:
                intermediates["double_block_inputs"].append((hidden_states, encoder_hidden_states))
            encoder_hidden_states, hidden_states = double_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
            )
            if return_intermediates:
                intermediates["double_block_outputs"].append((hidden_states, encoder_hidden_states))

        hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

        for single_block in self.single_blocks:
            hidden_states = single_block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                temb_mod=single_stream_mod,
            )
            if return_intermediates:
                intermediates["single_block_outputs"].append(hidden_states)

        if return_intermediates:
            intermediates["before_split"] = hidden_states

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_intermediates:
            return output, intermediates

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
