"""
Copyright 2026 Google LLC

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

"""Checkpointer for loading PRXPixel model weights and pipeline components."""

import os
from typing import Any, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from maxdiffusion import max_logging
from maxdiffusion.models.prx_pixel import (
    FlaxPRXPixelConfig,
    NNXPRXPixelTransformer2DModel,
    load_prx_pixel_weights,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, NNXFlaxQwen3Model
from maxdiffusion.models.qwen3_utils import load_qwen3_weights
from maxdiffusion.pipelines.prx_pixel import FlaxPRXPixelPipeline
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler


class PRXPixelCheckpointer:
  """Handles loading and initializing FlaxPRXPixelPipeline from Hugging Face or local directories."""

  @classmethod
  def load_pipeline(
      cls,
      model_path: str,
      dtype: Any = jnp.bfloat16,
      device: Optional[str] = None,
      revision: Optional[str] = "bcd5e63f072257a220c5d0ba039c97657398b1c2",
  ) -> FlaxPRXPixelPipeline:
    """Loads all components and constructs the FlaxPRXPixelPipeline."""
    max_logging.log(f"Loading PRXPixel from: {model_path} (dtype={dtype})...")

    # 1. Tokenizer
    if os.path.exists(os.path.join(model_path, "tokenizer")):
      tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    else:
      tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    # 2. Qwen3-VL Text Encoder
    qwen_config = FlaxQwen3Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        max_position_embeddings=256,
        max_layer_to_run=None,
        is_causal=True,
        dtype=dtype,
    )
    text_encoder = NNXFlaxQwen3Model(rngs=nnx.Rngs(0), config=qwen_config)
    te_path = os.path.join(model_path, "text_encoder/model.safetensors") if os.path.exists(os.path.join(model_path, "text_encoder")) else model_path
    
    _, te_state = nnx.split(text_encoder)
    te_eval_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, dtype), te_state.to_pure_dict())
    te_params = load_qwen3_weights(te_path, te_eval_shapes, device=device)
    nnx.update(text_encoder, te_params)
    max_logging.log("✅ Loaded Qwen3-VL text encoder!")

    # 3. PRXPixel Transformer
    transformer_config = FlaxPRXPixelConfig(
        in_channels=3,
        patch_size=16,
        context_in_dim=2048,
        hidden_size=3584,
        mlp_ratio=3.5,
        num_heads=28,
        depth=24,
        axes_dim=(64, 64),
        theta=10000,
        time_factor=1000.0,
        time_max_period=10000,
        dtype=dtype,
    )
    transformer = NNXPRXPixelTransformer2DModel(config=transformer_config, rngs=nnx.Rngs(0))
    tr_path = os.path.join(model_path, "transformer") if os.path.exists(os.path.join(model_path, "transformer")) else model_path

    _, tr_state = nnx.split(transformer)
    tr_eval_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, dtype), tr_state.to_pure_dict())
    tr_params = load_prx_pixel_weights(tr_path, tr_eval_shapes, device=device)
    nnx.update(transformer, tr_params)
    max_logging.log("✅ Loaded PRXPixel transformer!")

    # 4. Scheduler
    scheduler = FlaxFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=3.0,
    )

    pipeline = FlaxPRXPixelPipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        dtype=dtype,
    )
    max_logging.log("🎉 PRXPixel Pipeline loaded successfully!")
    return pipeline
