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
import torch
import jax
import jax.numpy as jnp
from flax import nnx
from diffusers import Flux2Transformer2DModel

from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
from maxdiffusion.models.flux.util import load_and_convert_flux_klein_nnx_weights


class TestFlux2KleinTransformerImageEditStep0Parity(unittest.TestCase):
  """Verifies Phase 4: Single-step Transformer forward pass with joint reference image conditioning."""

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
          if os.path.exists(self.transformer_path):
            break
    self.assertIsNotNone(self.model_dir, "Transformer directory not found!")

  def test_transformer_step0_forward_parity(self):
    """Tests NNXFlux2KleinTransformer2DModel forward pass at step 0 against PyTorch in float32."""
    print("\n" + "=" * 80)
    print("🧪 Running Phase 4: Step 0 Transformer Forward Pass Exact Parity Test...")
    print("=" * 80)

    # 1. Load PyTorch Diffusers Transformer
    print(" -> Loading PyTorch Transformer in float32...")
    pt_transformer = Flux2Transformer2DModel.from_pretrained(
        self.transformer_path, torch_dtype=torch.float32
    ).eval()

    # 2. Instantiate NNX Transformer Model (FLUX.2-Klein 4B config)
    print(" -> Instantiating Flax NNX Transformer in float32...")
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
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
    )

    # 3. Load Weights
    print(" -> Loading Transformer NNX weights from safetensors...")
    transformer_state = load_and_convert_flux_klein_nnx_weights(
        self.transformer_path,
        nnx.state(transformer, nnx.Param),
        num_double_layers=5,
        num_single_layers=20,
        dtype=jnp.float32,
    )
    nnx.update(transformer, transformer_state)

    # 4. Load Golden Inputs from Step 0
    prompt_embeds = np.load(os.path.join(self.golden_dir, "prompt_embeds.npy"))  # (1, 512, 7680)
    text_ids = np.load(os.path.join(self.golden_dir, "text_ids.npy"))  # (1, 512, 4)
    initial_noise = np.load(os.path.join(self.golden_dir, "initial_noise_latents.npy"))  # (1, 1024, 128)
    gen_ids = np.load(os.path.join(self.golden_dir, "gen_latent_ids.npy"))  # (1, 1024, 4)
    ref_latents_concat = np.load(os.path.join(self.golden_dir, "image_latents_concat.npy"))  # (1, 4096, 128)
    ref_latent_ids = np.load(os.path.join(self.golden_dir, "image_latent_ids.npy"))  # (1, 4096, 4)

    # 5. Joint Sequence Concatenation
    joint_image_latents = np.concatenate([initial_noise, ref_latents_concat], axis=1)  # (1, 5120, 128)
    joint_image_ids = np.concatenate([gen_ids, ref_latent_ids], axis=1)  # (1, 5120, 4)

    # 6. PyTorch Forward Pass
    with torch.no_grad():
      pt_out = pt_transformer(
          hidden_states=torch.tensor(joint_image_latents, dtype=torch.float32),
          encoder_hidden_states=torch.tensor(prompt_embeds, dtype=torch.float32),
          timestep=torch.tensor([1.0], dtype=torch.float32),
          img_ids=torch.tensor(joint_image_ids, dtype=torch.float32),
          txt_ids=torch.tensor(text_ids, dtype=torch.float32),
      ).sample.numpy()

    # 7. Flax NNX Forward Pass
    print(" -> Executing Flax NNX Transformer forward pass...")
    out = transformer(
        hidden_states=jnp.array(joint_image_latents, dtype=jnp.float32),
        encoder_hidden_states=jnp.array(prompt_embeds, dtype=jnp.float32),
        timestep=jnp.array([1.0], dtype=jnp.float32),
        img_ids=jnp.array(joint_image_ids, dtype=jnp.float32),
        txt_ids=jnp.array(text_ids, dtype=jnp.float32),
        guidance=None,
    ).sample  # (1, 5120, 128)
    jax_out_np = np.array(out)

    # 8. Slicing generated latents
    num_gen_tokens = initial_noise.shape[1]  # 1024
    jax_noise_pred = jax_out_np[:, :num_gen_tokens, :]
    pt_noise_pred = pt_out[:, :num_gen_tokens, :]

    # 9. Verify Numerical Precision
    pred_diff = np.abs(jax_noise_pred - pt_noise_pred)
    max_pred_diff = np.max(pred_diff)
    mean_pred_diff = np.mean(pred_diff)
    print(f" -> Sliced Step 0 Noise Prediction: Max Diff = {max_pred_diff:.6e}, Mean Diff = {mean_pred_diff:.6e}")
    self.assertLess(max_pred_diff, 5e-4, "Step 0 noise prediction exceeds float32 tolerance!")
    self.assertLess(mean_pred_diff, 1e-5, "Step 0 mean difference exceeds float32 tolerance!")

    print("=" * 80)
    print("✅ Phase 4: Step 0 Transformer Forward Pass Parity PASSED!")
    print("=" * 80)


if __name__ == "__main__":
  unittest.main()
