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

import os
import unittest
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoConfig, Qwen2TokenizerFast

from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.flux.util import load_and_convert_flux_klein_nnx_weights
from maxdiffusion.models.qwen3_flax import FlaxQwen3Model, FlaxQwen3Config
from maxdiffusion.models.qwen3_utils import load_and_convert_qwen3_weights
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.pipelines.flux.flux2klein_image_edit_pipeline import FlaxFlux2KleinImageEditPipeline


class TestFlux2KleinImageEditPipeline(unittest.TestCase):
  """Verifies Phase 6: Standalone FlaxFlux2KleinImageEditPipeline end-to-end inference."""

  def setUp(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    self.golden_dir = "/mnt/data/golden_image_edit_data" if os.path.exists("/mnt/data/golden_image_edit_data") else "golden_image_edit_data"
    self.assertTrue(os.path.exists(self.golden_dir), f"Golden data directory not found: {self.golden_dir}")

    candidates = [
        "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        "/mnt/hyperdisk_weights/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots",
        os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"),
    ]
    self.model_dir = None
    for c in candidates:
      if os.path.exists(c):
        snaps = os.listdir(c)
        if snaps:
          self.model_dir = os.path.join(c, snaps[0])
          self.transformer_path = os.path.join(self.model_dir, "transformer")
          self.vae_path = os.path.join(self.model_dir, "vae", "diffusion_pytorch_model.safetensors")
          self.text_encoder_path = os.path.join(self.model_dir, "text_encoder")
          self.tokenizer_path = os.path.join(self.model_dir, "tokenizer")
          if os.path.exists(self.transformer_path) and os.path.exists(self.vae_path):
            break
    self.assertIsNotNone(self.model_dir, "FLUX.2-Klein 4B model directory not found!")

  def test_pipeline_image_edit_execution(self):
    """Initializes FlaxFlux2KleinImageEditPipeline and executes 4-image conditioning image edit."""
    print("\n" + "=" * 80)
    print("🧪 Running Phase 6: Standalone FlaxFlux2KleinImageEditPipeline Test...")
    print("=" * 80)

    # 1. Instantiate NNX Transformer
    print(" -> Loading NNX Transformer...")
    rngs = nnx.Rngs(0)
    transformer = NNXFlux2KleinTransformer2DModel(
        rngs=rngs,
        patch_size=1,
        in_channels=128,
        num_layers=5,
        num_single_layers=20,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=7680,
        pooled_projection_dim=None,
        guidance_embeds=False,
        axes_dim=(32, 32, 32, 32),
        scale_shift_order="scale_shift",
        dtype=jnp.bfloat16,
        weights_dtype=jnp.bfloat16,
    )
    t_state = load_and_convert_flux_klein_nnx_weights(
        self.transformer_path,
        nnx.state(transformer, nnx.Param),
        num_double_layers=5,
        num_single_layers=20,
        dtype=jnp.bfloat16,
    )
    nnx.update(transformer, t_state)

    # 2. Instantiate NNX VAE
    print(" -> Loading NNX VAE...")
    nnx_vae = NNXAutoencoderKLFlux2(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    bn_mean, bn_std = load_and_convert_flux2klein_nnx_vae_weights(
        self.vae_path, nnx_vae, dtype=jnp.bfloat16
    )

    # 3. Instantiate Qwen3 Text Encoder
    print(" -> Loading Qwen3 Text Encoder...")
    pt_config = AutoConfig.from_pretrained(self.text_encoder_path)
    qwen3_config = FlaxQwen3Config(
        vocab_size=pt_config.vocab_size,
        hidden_size=pt_config.hidden_size,
        intermediate_size=pt_config.intermediate_size,
        num_hidden_layers=pt_config.num_hidden_layers,
        num_attention_heads=pt_config.num_attention_heads,
        num_key_value_heads=pt_config.num_key_value_heads,
        max_position_embeddings=pt_config.max_position_embeddings,
        rms_norm_eps=pt_config.rms_norm_eps,
        rope_theta=pt_config.rope_theta,
        dtype=jnp.bfloat16,
        max_layer_to_run=27,
    )
    text_encoder = FlaxQwen3Model(config=qwen3_config)
    key = jax.random.PRNGKey(0)
    abstract_q_vars = text_encoder.init(key, jnp.zeros((1, 512), dtype=jnp.int32), jnp.zeros((1, 512), dtype=jnp.int32))
    q_params = load_and_convert_qwen3_weights(self.text_encoder_path, abstract_q_vars["params"], qwen3_config)

    tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_path)
    scheduler = FlaxFlowMatchScheduler(num_train_timesteps=1000, shift=3.0)

    # 4. Instantiate Pipeline
    print(" -> Instantiating FlaxFlux2KleinImageEditPipeline...")
    mock_config = type("Config", (), {
        "max_sequence_length": 512,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "text_encoder_max_layer": 27,
    })()

    pipeline = FlaxFlux2KleinImageEditPipeline(
        transformer=transformer,
        vae=nnx_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        config=mock_config,
        vae_bn_mean=bn_mean,
        vae_bn_std=bn_std,
    )

    # 5. Load 4 Reference Images
    ref_images = []
    for i in range(4):
      img_path = os.path.join(self.golden_dir, "ref_images", f"ref_image_{i}.png")
      self.assertTrue(os.path.exists(img_path), f"Missing reference image: {img_path}")
      ref_images.append(Image.open(img_path))

    # 6. Run Pipeline Call
    prompt = "a vibrant artistic painting combining the dog, car, mountain, and fruit bowl in surreal neon lighting"
    print(f" -> Calling pipeline with prompt: '{prompt}' and {len(ref_images)} reference images...")
    
    images_out = pipeline(
        prompt=prompt,
        images=ref_images,
        height=512,
        width=512,
        num_inference_steps=4,
        text_encoder_params=q_params,
        prng_key=jax.random.PRNGKey(42),
        output_type="pil",
    )

    self.assertIsInstance(images_out, list)
    self.assertEqual(len(images_out), 1)
    out_img = images_out[0]
    self.assertEqual(out_img.size, (512, 512))

    save_path = os.path.join(self.golden_dir, "pipeline_edited_image.png")
    out_img.save(save_path)
    print(f"🎉 Pipeline execution succeeded! Output saved to: {save_path}")

    print("=" * 80)
    print("✅ Phase 6: Standalone FlaxFlux2KleinImageEditPipeline PASSED!")
    print("=" * 80)


if __name__ == "__main__":
  unittest.main()
