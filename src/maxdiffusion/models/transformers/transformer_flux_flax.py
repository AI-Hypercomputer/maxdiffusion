"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

'''This script is used an example of how to shard the UNET on TPU.'''

import jax.numpy as jnp
import flax.linen as nn
from ..normalization_flax import FlaxAdaLayerNormZeroSingle
from ..attention_flax import FlaxAttention

class FlaxFluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    mlp_ratio: int = 4.0

    def setup(self):
        super.__init__()
        self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        self.norm = FlaxAdaLayerNormZeroSingle(self.dim)
        self.proj_mlp = nn.Dense(self.mlp_hidden_dim)
        self.act_mlp = nn.GELU
        self.proj_out = nn.Dense(self.dim)
        self.attn = FlaxAttention(
            query_dim=self.dim,
            heads=self.num_attention_heads,
            dim_head=self.attention_head_dim,


        )
    
    def __call__(self, hidden_states, temb, image_rotary_emb=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        hidden_states = jnp.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = jnp.expand_dims(x, axis=1)