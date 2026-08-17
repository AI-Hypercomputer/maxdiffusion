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
import gc
import unittest
import pytest
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import AutoConfig, Qwen2TokenizerFast

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.flux.util import load_and_convert_flux_klein_nnx_weights
from maxdiffusion.models.qwen3_flax import FlaxQwen3Model, FlaxQwen3Config
from maxdiffusion.models.qwen3_utils import load_and_convert_qwen3_weights
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.pipelines.flux.flux2klein_pipeline import FlaxFlux2KleinPipeline

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT = "a vibrant artistic painting combining the dog, car, mountain, and fruit bowl in surreal neon lighting"


class TestFlux2KleinImageEditE2EParity(unittest.TestCase):
  """End-to-End Parity Test between PyTorch Diffusers CPU and MaxDiffusion TPU."""

  def setUp(self):
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_use_shardy_partitioner", True)

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

    self.output_dir = "/mnt/data/e2e_parity" if os.path.exists("/mnt/data") else "/tmp/e2e_parity"
    os.makedirs(self.output_dir, exist_ok=True)

    # Resolve reference images
    ref_dir = "/mnt/data/golden_image_edit_data/ref_images"
    self.ref_images = []
    if os.path.exists(ref_dir):
      for i in range(4):
        p = os.path.join(ref_dir, f"ref_image_{i}.png")
        if os.path.exists(p):
          self.ref_images.append(Image.open(p).convert("RGB"))

    if len(self.ref_images) < 4:
      # Generate synthetic test reference images if not present
      colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
      for i, c in enumerate(colors):
        arr = np.full((512, 512, 3), c, dtype=np.uint8)
        self.ref_images.append(Image.fromarray(arr))

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run on Github Actions (requires TPU and full weights)")
  def test_e2e_image_edit_parity_vs_diffusers(self):
    """Generates an image edit on PyTorch Diffusers CPU and MaxDiffusion TPU and asserts SSIM >= 0.75."""
    from diffusers import Flux2KleinPipeline as DiffusersFlux2KleinPipeline

    print("\n" + "=" * 80)
    print("🚀 [STEP 1/3] Running Reference PyTorch Diffusers CPU Pipeline...")
    print("=" * 80)

    diffusers_pipe = DiffusersFlux2KleinPipeline.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16)
    diffusers_pipe.to("cpu")

    # Generate initial noise latents deterministically on CPU (4D tensor for Diffusers prepare_latents)
    gen = torch.Generator(device="cpu").manual_seed(42)
    raw_latents_pt = torch.randn(
        (1, 128, 512 // 16, 512 // 16),
        generator=gen,
        dtype=torch.bfloat16,
        device="cpu",
    )

    with torch.no_grad():
      diffusers_out = diffusers_pipe(
          prompt=PROMPT,
          image=self.ref_images,
          height=512,
          width=512,
          num_inference_steps=4,
          latents=raw_latents_pt,
          guidance_scale=1.0,
      )

    diffusers_image = diffusers_out.images[0]
    diffusers_img_path = os.path.join(self.output_dir, "diffusers_cpu_output.png")
    diffusers_image.save(diffusers_img_path)
    print(f" -> Saved PyTorch Diffusers output to: {diffusers_img_path}")

    # Free PyTorch pipeline memory before TPU run
    del diffusers_pipe
    gc.collect()

    print("\n" + "=" * 80)
    print("🚀 [STEP 2/3] Running MaxDiffusion Unified FlaxFlux2KleinPipeline on TPU...")
    print("=" * 80)

    # 1. Device mesh setup
    active_devices = jax.devices()
    active_device_count = len(active_devices)

    pyconfig._config = None
    pyconfig.config = None
    config_path = os.path.join(THIS_DIR, "..", "configs", "base_flux2klein.yml")
    args = [
        None,
        config_path,
        "run_name=e2e_parity_test",
        f"output_dir={self.output_dir}",
        f"per_device_batch_size={1.0 / active_device_count}",
        "height=512",
        "width=512",
        "seed=42",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
        "text_encoder_attention=dot_product",
    ]
    pyconfig.initialize(args)
    config = pyconfig.config

    if active_device_count > 1:
      pyconfig._config.keys["ici_tensor_parallelism"] = active_device_count
      pyconfig._config.keys["ici_data_parallelism"] = 1
      pyconfig._config.keys["ici_fsdp_parallelism"] = 1
      pyconfig._config.keys["ici_context_parallelism"] = 1

    devices_array = create_device_mesh(config, devices=active_devices)
    mesh = Mesh(devices_array, config.mesh_axes)

    # 2. Load NNX Transformer
    print(" -> Loading NNX Transformer weights...")
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

    # 3. Load NNX VAE
    print(" -> Loading NNX VAE weights...")
    nnx_vae = NNXAutoencoderKLFlux2(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    bn_mean, bn_std = load_and_convert_flux2klein_nnx_vae_weights(self.vae_path, nnx_vae, dtype=jnp.bfloat16)

    # 4. Load Qwen3
    print(" -> Loading Qwen3 weights...")
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
    abstract_q_vars = text_encoder.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 512), dtype=jnp.int32), jnp.zeros((1, 512), dtype=jnp.int32)
    )
    q_params = load_and_convert_qwen3_weights(self.text_encoder_path, abstract_q_vars["params"], qwen3_config)

    tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_path)
    scheduler = FlaxFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        sigma_max=1.0,
        sigma_min=0.001,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
    )

    # 5. Place parameters on TPU HBM
    t_params = nnx.state(transformer, nnx.Param)
    v_params = nnx.state(nnx_vae, nnx.Param)

    t_params = jax.device_put(t_params)
    v_params = jax.device_put(v_params)
    q_params = jax.device_put(q_params)

    # 6. Instantiate Unified FlaxFlux2KleinPipeline
    pipeline = FlaxFlux2KleinPipeline(
        transformer=transformer,
        vae=nnx_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        config=config,
        mesh=mesh,
    )

    # 7. AOT Compile async
    pipeline.compile_aot_async(
        params=t_params,
        vae_params=v_params,
        qwen3_params=q_params,
        vae_bn_mean=bn_mean,
        vae_bn_std=bn_std,
        batch_size=1,
        height=512,
        width=512,
        images=self.ref_images,
    )

    # Convert PyTorch initial noise latents to JAX array (shape: 1, 32, 64, 64)
    initial_latents_jax = jnp.array(raw_latents_pt.detach().float().cpu().numpy())

    # 8. Run pipeline
    print(f" -> Running FlaxFlux2KleinPipeline with {len(self.ref_images)} reference images on TPU...")
    pipeline(
        prompt=PROMPT,
        params=t_params,
        vae_params=v_params,
        qwen3_params=q_params,
        vae_bn_mean=bn_mean,
        vae_bn_std=bn_std,
        transformer_shardings=None,
        vae_shardings=None,
        qwen3_shardings=None,
        height=512,
        width=512,
        num_inference_steps=4,
        batch_size=1,
        images=self.ref_images,
        use_latents=True,
        latents=initial_latents_jax,
        output_dir=self.output_dir,
        output_name="maxdiffusion_tpu_output.png",
    )

    maxdiff_img_path = os.path.join(self.output_dir, "maxdiffusion_tpu_output.png")
    self.assertTrue(os.path.exists(maxdiff_img_path), "MaxDiffusion output image was not saved!")
    maxdiff_image = Image.open(maxdiff_img_path).convert("RGB")

    print("\n" + "=" * 80)
    print("📊 [STEP 3/3] Evaluating End-to-End Parity (SSIM & PSNR)...")
    print("=" * 80)

    diffusers_arr = np.array(diffusers_image).astype(np.uint8)
    maxdiff_arr = np.array(maxdiff_image).astype(np.uint8)

    self.assertEqual(diffusers_arr.shape, maxdiff_arr.shape)

    ssim_val = ssim(diffusers_arr, maxdiff_arr, channel_axis=-1, data_range=255)
    mse = np.mean((diffusers_arr.astype(np.float64) - maxdiff_arr.astype(np.float64)) ** 2)
    psnr_val = 10.0 * np.log10(255.0**2 / (mse + 1e-10))

    print(f" -> SSIM (Diffusers CPU vs MaxDiffusion TPU): {ssim_val:.6f}")
    print(f" -> PSNR (Diffusers CPU vs MaxDiffusion TPU): {psnr_val:.2f} dB")
    print(f" -> MSE: {mse:.4f}")

    # Create side-by-side comparison image
    side_by_side = Image.new("RGB", (1024, 512))
    side_by_side.paste(diffusers_image, (0, 0))
    side_by_side.paste(maxdiff_image, (512, 0))
    comparison_path = os.path.join(self.output_dir, "e2e_parity_diffusers_vs_maxdiffusion.png")
    side_by_side.save(comparison_path)
    print(f" -> Saved side-by-side comparison to: {comparison_path}")

    self.assertGreaterEqual(ssim_val, 0.75, f"SSIM score {ssim_val:.4f} is below target threshold 0.75!")
    print("🎉 END-TO-END PARITY TEST PASSED! MaxDiffusion matches Diffusers reference!")


if __name__ == "__main__":
  unittest.main()
