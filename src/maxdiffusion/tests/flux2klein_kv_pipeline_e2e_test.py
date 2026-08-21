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

import gc
import json
import os
import unittest
import numpy as np
import pytest
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import flax.linen as nn
from jax.sharding import Mesh
from transformers import AutoConfig, Qwen2TokenizerFast

from maxdiffusion import max_utils, pyconfig
from maxdiffusion.max_utils import create_device_mesh, get_flash_block_sizes
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


def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
  arr1 = np.array(img1, dtype=np.float64)
  arr2 = np.array(img2, dtype=np.float64)
  mse = np.mean((arr1 - arr2) ** 2)
  if mse == 0:
    return float("inf")
  return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
  arr1 = np.array(img1.convert("RGB"))
  arr2 = np.array(img2.convert("RGB"))
  return float(ssim(arr1, arr2, channel_axis=-1))


def find_model_path():
  if "FLUX2_KLEIN_KV_MODEL_PATH" in os.environ:
    return os.environ["FLUX2_KLEIN_KV_MODEL_PATH"]
  hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
  candidates = [
      os.path.join(hf_home, "hub/models--black-forest-labs--FLUX.2-klein-9B-KV/snapshots"),
      os.path.join(hf_home, "hub/models--black-forest-labs--FLUX.2-klein-9b-kv/snapshots"),
      os.path.join(hf_home, "hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots"),
      "/mnt/hyperdisk_weights/hub/flux2klein-9b-kv",
      "/mnt/data/models/flux2klein-9b-kv",
  ]
  for c in candidates:
    if os.path.exists(c):
      if "snapshots" in c:
        snaps = os.listdir(c)
        if snaps:
          return os.path.join(c, snaps[0])
      else:
        return c
  return "black-forest-labs/FLUX.2-klein-9B-KV"


class TestFlux2KleinKVPipelineE2EBF16Parity(unittest.TestCase):
  """End-to-end parity test comparing Diffusers Flux2KleinKVPipeline vs MaxDiffusion FlaxFlux2KleinPipeline (use_kv=True) in bfloat16."""

  @classmethod
  def setUpClass(cls):
    cls.model_path = find_model_path()
    cls.work_dir = "/tmp/flux2klein_kv_e2e"
    os.makedirs(cls.work_dir, exist_ok=True)

    cls.height = 256
    cls.width = 256
    cls.num_inference_steps = 4
    cls.seed = int(os.getenv("FLUX2_KLEIN_E2E_SEED", "42"))

    # 1. Load 4 real reference images (256x256)
    ref_dir = os.path.join(THIS_DIR, "images", "flux2klein")
    cls.ref_images = []
    if os.path.exists(ref_dir):
      for i in range(4):
        p = os.path.join(ref_dir, f"ref_image_{i}.png")
        if os.path.exists(p):
          cls.ref_images.append(Image.open(p).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC))

    if len(cls.ref_images) < 4:
      colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
      for i, c in enumerate(colors):
        arr = np.full((256, 256, 3), c, dtype=np.uint8)
        cls.ref_images.append(Image.fromarray(arr))

    # 2. Generate shared starting noise latents (1, 32, 32, 32)
    rng = np.random.RandomState(cls.seed)
    latents_unpacked = rng.randn(1, 32, cls.height // 8, cls.width // 8).astype(np.float32)
    cls.latents_unpacked_jax = jnp.array(latents_unpacked)

    # Prepare packed latents for PyTorch: (1, 32, H/16, 2, W/16, 2) -> permute(0, 1, 3, 5, 2, 4) -> reshape(1, 128, H/16, W/16)
    latents_unpacked_pt = torch.from_numpy(latents_unpacked)
    latents_pt_packed = latents_unpacked_pt.view(1, 32, cls.height // 16, 2, cls.width // 16, 2)
    latents_pt_packed = latents_pt_packed.permute(0, 1, 3, 5, 2, 4)
    cls.latents_pt_packed = latents_pt_packed.reshape(1, 128, cls.height // 16, cls.width // 16)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires TPU and full FLUX.2-Klein 9B weights")
  def test_flux2klein_kv_pipeline_bf16_parity(self):
    """Executes PyTorch Diffusers Flux2KleinKVPipeline in BF16 and MaxDiffusion FlaxFlux2KleinPipeline in BF16 (use_kv=True) and compares outputs."""
    print("\n" + "=" * 80)
    print("🚀 FLUX.2-KLEIN-9B KV CACHE END-TO-END BF16 PARITY TEST")
    print("=" * 80)
    print(f"Model Path:          {self.model_path}")
    print(f"Prompt:              '{PROMPT}'")
    print(f"Number of Ref Images:{len(self.ref_images)} (256x256)")
    print(f"Target Resolution:   {self.width}x{self.height}")
    print(f"Inference Steps:     {self.num_inference_steps}")

    # =========================================================================
    # LEG 1: PyTorch Diffusers Flux2KleinKVPipeline in BF16
    # =========================================================================
    print("\n" + "-" * 80)
    print("🎬 LEG 1: Running PyTorch Diffusers Flux2KleinKVPipeline (bfloat16 on CPU)...")
    print("-" * 80)
    from diffusers import Flux2KleinKVPipeline

    pipe_pt = Flux2KleinKVPipeline.from_pretrained(
        self.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe_pt.to("cpu")

    with torch.no_grad():
      pt_out = pipe_pt(
          prompt=PROMPT,
          image=self.ref_images,
          latents=self.latents_pt_packed.to(torch.bfloat16),
          num_inference_steps=self.num_inference_steps,
          height=self.height,
          width=self.width,
          output_type="pil",
      ).images[0]

    pt_output_path = os.path.join(self.work_dir, "pt_bf16_output.png")
    pt_out.save(pt_output_path)
    print(f" -> Saved PyTorch Diffusers BF16 output: {pt_output_path}")

    del pipe_pt
    gc.collect()

    # =========================================================================
    # LEG 2: MaxDiffusion FlaxFlux2KleinPipeline in BF16 (use_kv=True)
    # =========================================================================
    print("\n" + "-" * 80)
    print("🎬 LEG 2: Running MaxDiffusion FlaxFlux2KleinPipeline with use_kv=True (bfloat16 on TPU)...")
    print("-" * 80)

    # 1. Device mesh setup
    active_devices = jax.devices()
    active_device_count = len(active_devices)

    pyconfig._config = None
    pyconfig.config = None
    config_path = os.path.join(THIS_DIR, "..", "configs", "base_flux2klein_9B.yml")
    args = [
        None,
        config_path,
        "run_name=e2e_kv_parity_test",
        f"output_dir={self.work_dir}",
        f"per_device_batch_size={1.0 / active_device_count}",
        f"height={self.height}",
        f"width={self.width}",
        f"num_inference_steps={self.num_inference_steps}",
        f"seed={self.seed}",
        "use_kv=True",
        "weights_dtype=bfloat16",
        "activations_dtype=bfloat16",
        "precision=DEFAULT",
        "attention=tokamax_flash",
        'flash_block_sizes={"block_q": 512, "block_kv": 512, "block_kv_compute": 512}',
        "text_encoder_attention=dot_product",
    ]
    pyconfig.initialize(args, unittest=True)
    config = pyconfig.config

    if active_device_count > 1:
      pyconfig._config.keys["ici_tensor_parallelism"] = active_device_count
      pyconfig._config.keys["ici_data_parallelism"] = 1
      pyconfig._config.keys["ici_fsdp_parallelism"] = 1
      pyconfig._config.keys["ici_context_parallelism"] = 1

    pyconfig._config.keys["flash_block_sizes"] = {
        "block_q": 512,
        "block_kv": 512,
        "block_kv_compute": 512,
    }

    devices_array = create_device_mesh(config, devices=active_devices)
    mesh = Mesh(devices_array, config.mesh_axes)

    # 2. Text Encoder & Tokenizer
    text_encoder_path = os.path.join(self.model_path, "text_encoder")
    tokenizer_path = os.path.join(self.model_path, "tokenizer")
    pt_config = AutoConfig.from_pretrained(text_encoder_path)
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
        attention_kernel="dot_product",
        mesh=mesh,
        max_layer_to_run=getattr(config, "text_encoder_max_layer", 27),
        is_causal=True,
    )
    text_encoder = FlaxQwen3Model(qwen3_config)
    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)

    # 3. NNX Transformer
    transformer_path = os.path.join(self.model_path, "transformer")
    transformer_config_json = os.path.join(transformer_path, "config.json")
    transformer_pt_cfg = {}
    if os.path.exists(transformer_config_json):
      with open(transformer_config_json, "r") as f:
        transformer_pt_cfg = json.load(f)

    num_double_layers = transformer_pt_cfg.get("num_layers", 8)
    depth = transformer_pt_cfg.get("num_single_layers", 24)
    num_attention_heads = transformer_pt_cfg.get("num_attention_heads", 32)

    transformer = NNXFlux2KleinTransformer2DModel(
        rngs=nnx.Rngs(0),
        in_channels=128,
        num_layers=num_double_layers,
        num_single_layers=depth,
        attention_head_dim=128,
        num_attention_heads=num_attention_heads,
        joint_attention_dim=3 * pt_config.hidden_size,
        pooled_projection_dim=768,
        guidance_embeds=transformer_pt_cfg.get("guidance_embeds", False),
        axes_dim=(32, 32, 32, 32),
        theta=2000.0,
        mlp_ratio=3.0,
        attention_kernel=config.attention,
        flash_min_seq_length=512,
        flash_block_sizes=get_flash_block_sizes(config),
        mesh=mesh,
        dtype=jnp.bfloat16,
        weights_dtype=jnp.bfloat16,
        scale_shift_order="scale_shift",
        use_base2_exp=True,
    )

    # 4. NNX VAE
    vae_path = os.path.join(self.model_path, "vae", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(vae_path):
      vae_path = os.path.join(self.model_path, "vae")
    vae = NNXAutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    # 5. Extract mesh shardings for all models
    abstract_transformer_state = nnx.state(transformer, nnx.Param)
    abstract_vae_state = nnx.state(vae, nnx.Param)

    def qwen3_init_fn():
      return text_encoder.init(
          jax.random.PRNGKey(0), jnp.zeros((1, 512), dtype=jnp.int32), jnp.zeros((1, 512), dtype=jnp.int32)
      )

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      logical_transformer_specs = nnx.get_partition_spec(abstract_transformer_state)
      logical_vae_specs = nnx.get_partition_spec(abstract_vae_state)
      abstract_qwen3_vars = jax.eval_shape(qwen3_init_fn)
      logical_qwen3_specs = nn.get_partition_spec(abstract_qwen3_vars)

      transformer_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
      vae_shardings = nn.logical_to_mesh_sharding(logical_vae_specs, mesh, config.logical_axis_rules)
      qwen3_shardings = flax.core.freeze(
          nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)["params"]
      )

    # 6. Load weights on Host CPU and shard across TPU HBM
    cpu_device = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu_device):
      t_params = load_and_convert_flux_klein_nnx_weights(
          transformer_path,
          abstract_transformer_state,
          num_double_layers=num_double_layers,
          num_single_layers=depth,
          dtype=jnp.bfloat16,
      )
      vae_bn_mean, vae_bn_std = load_and_convert_flux2klein_nnx_vae_weights(vae_path, vae, dtype=jnp.bfloat16)
      v_params = nnx.state(vae, nnx.Param)

      def unbox_fn(x):
        import flax.linen.spmd as flax_spmd

        return x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x

      qwen3_params_template = jax.tree_util.tree_map(
          unbox_fn, abstract_qwen3_vars["params"], is_leaf=lambda k: hasattr(k, "unbox")
      )
      qwen3_params_template = flax.core.unfreeze(qwen3_params_template)
      q_params = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params_template, qwen3_config)
      q_params = flax.core.freeze(q_params)

    # Shard onto TPU HBM
    print(" -> Sharding parameters across TPU HBM...")
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      t_params = jax.tree_util.tree_map(max_utils.device_put_replicated, t_params, transformer_shardings)
      v_params = jax.tree_util.tree_map(max_utils.device_put_replicated, v_params, vae_shardings)
      nnx.update(vae, v_params)
      q_params = jax.tree_util.tree_map(max_utils.device_put_replicated, q_params, qwen3_shardings)

    # 7. Scheduler
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

    # 8. Pipeline instantiation & execution
    pipeline = FlaxFlux2KleinPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        config=config,
        mesh=mesh,
    )

    pipeline.compile_aot_async(
        params=t_params,
        vae_params=v_params,
        qwen3_params=q_params,
        vae_bn_mean=vae_bn_mean,
        vae_bn_std=vae_bn_std,
        batch_size=1,
        height=self.height,
        width=self.width,
        images=self.ref_images,
        use_kv=True,
    )

    jax_output_name = "jax_bf16_output.png"
    pipeline(
        prompt=PROMPT,
        params=t_params,
        vae_params=v_params,
        qwen3_params=q_params,
        vae_bn_mean=vae_bn_mean,
        vae_bn_std=vae_bn_std,
        transformer_shardings=transformer_shardings,
        vae_shardings=vae_shardings,
        qwen3_shardings=qwen3_shardings,
        height=self.height,
        width=self.width,
        num_inference_steps=self.num_inference_steps,
        batch_size=1,
        images=self.ref_images,
        use_latents=True,
        latents=self.latents_unpacked_jax,
        use_kv=True,
        output_dir=self.work_dir,
        output_name=jax_output_name,
    )

    jax_output_path = os.path.join(self.work_dir, jax_output_name)
    self.assertTrue(os.path.exists(jax_output_path), f"JAX output image not found at {jax_output_path}")
    print(f" -> Found MaxDiffusion JAX BF16 output: {jax_output_path}")

    # =========================================================================
    # LEG 3: Compute Parity Metrics (SSIM & PSNR)
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 CROSS-FRAMEWORK BF16 PARITY EVALUATION REPORT")
    print("=" * 80)

    img_pt = Image.open(pt_output_path).convert("RGB")
    img_jax = Image.open(jax_output_path).convert("RGB")

    score_ssim = compute_ssim(img_jax, img_pt)
    score_psnr = compute_psnr(img_jax, img_pt)

    print(f" -> Structural Similarity (SSIM): {score_ssim:.6f}")
    print(f" -> Peak Signal-to-Noise Ratio (PSNR): {score_psnr:.2f} dB")
    print("=" * 80)

    self.assertGreaterEqual(
        score_ssim, 0.70, f"End-to-End BF16 SSIM {score_ssim:.6f} is below the required acceptance threshold of 0.70"
    )
    print("✅ End-to-End BF16 Parity Test PASSED successfully!\n")


if __name__ == "__main__":
  unittest.main()
