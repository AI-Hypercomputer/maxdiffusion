# Copyright 2026 Google LLC
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

import os
import sys

# Set HF_HOME cache path early
if not os.environ.get("HF_HOME"):
  if os.path.exists("/mnt/data/hf_cache"):
    os.environ["HF_HOME"] = "/mnt/data/hf_cache"

from maxdiffusion import pyconfig

# Initialize config first to prevent early XLA initialization
config_path = "src/maxdiffusion/configs/base_flux2klein.yml"
print(f"Initializing pyconfig with: {config_path}")
pyconfig.initialize([
    None,
    config_path,
    "run_name=test_4b_e2e_parity",
    "output_dir=/tmp/",
    "weights_dtype=bfloat16",
    "activations_dtype=bfloat16",
    "ici_data_parallelism=1",
    "ici_fsdp_parallelism=1",
    "ici_tensor_parallelism=4",
    "ici_context_parallelism=1",
])
config = pyconfig.config

# Now import the rest of the libraries safely
import time
import gc
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from maxdiffusion.max_utils import create_device_mesh, get_precision
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning

from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, compute_empirical_mu
from maxdiffusion.models.vae_flax import FlaxAutoencoderKL
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model, load_and_convert_qwen3_weights
from maxdiffusion.models.flux.util import (
    load_and_convert_flux_klein_weights as load_and_convert_weights,
    load_and_convert_vae_weights,
    cast_dict_to_bfloat16_inplace,
    prepare_text_ids,
    prepare_latent_image_ids,
    pack_latents,
    unpack_latents,
)


def run_parity():
  global config

  # 2. Setup device mesh
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  print(f"Device mesh created: {mesh}")

  # Locate cached model files
  model_id = config.pretrained_model_name_or_path
  cache_dir = f"/mnt/data/hf_cache/hub/models--{model_id.replace('/', '--')}/snapshots"
  if not os.path.exists(cache_dir):
    raise FileNotFoundError(f"Hugging Face cache directory not found: {cache_dir}")
  snapshots = os.listdir(cache_dir)
  snapshot_dir = os.path.join(cache_dir, snapshots[0])
  print(f"Loading weights from snapshot directory: {snapshot_dir}")

  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
  transformer_path = os.path.join(snapshot_dir, "transformer")
  vae_safetensors_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
  tokenizer_path = os.path.join(snapshot_dir, "tokenizer")

  # Inputs
  prompt = "A dog eating pasta"
  width = 512
  height = 512
  num_inference_steps = 4
  batch_size = 1
  seed = 42

  # Generate identical starting noise on CPU
  print(f"Generating shared starting noise on CPU (seed={seed})...")
  generator = torch.Generator(device="cpu").manual_seed(seed)
  latents_unpacked_pt = torch.randn(batch_size, 32, height // 8, width // 8, generator=generator, dtype=torch.float32)
  latents_numpy = latents_unpacked_pt.numpy()

  # Pack/patchify noise for PyTorch pipeline input
  latents_pt_packed = latents_unpacked_pt.view(batch_size, 32, height // 16, 2, width // 16, 2)
  latents_pt_packed = latents_pt_packed.permute(0, 1, 3, 5, 2, 4)
  latents_pt_packed = latents_pt_packed.reshape(batch_size, 128, height // 16, width // 16)

  # Save directory
  os.makedirs("src/maxdiffusion/tests/flux2klein/images", exist_ok=True)
  pt_fp32_path = "src/maxdiffusion/tests/flux2klein/images/stunt_4b_pytorch_fp32.png"
  pt_bf16_path = "src/maxdiffusion/tests/flux2klein/images/stunt_4b_pytorch_bf16.png"
  jax_bf16_path = "src/maxdiffusion/tests/flux2klein/images/stunt_4b_jax_bf16.png"

  # -------------------------------------------------------------------------
  # LEG 1: PyTorch CPU Float32 (Golden Reference)
  # -------------------------------------------------------------------------
  print("\n" + "=" * 80)
  print("🚀 LEG 1: RUNNING PYTORCH CPU PIPELINE (FP32)...")
  print("=" * 80)
  from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

  pipe_fp32 = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.float32, local_files_only=True)
  pipe_fp32.to("cpu")

  with torch.no_grad():
    pt_image_fp32 = pipe_fp32(
        prompt=prompt,
        width=width,
        height=height,
        latents=latents_pt_packed,
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images[0]
  pt_image_fp32.save(pt_fp32_path)
  print(f"Saved Golden PyTorch FP32 image to: {pt_fp32_path}")

  del pipe_fp32
  gc.collect()

  # -------------------------------------------------------------------------
  # LEG 2: PyTorch CPU Bfloat16 (Precision Baseline)
  # -------------------------------------------------------------------------
  print("\n" + "=" * 80)
  print("🚀 LEG 2: RUNNING PYTORCH CPU PIPELINE (BF16)...")
  print("=" * 80)
  pipe_bf16 = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.bfloat16, local_files_only=True)
  pipe_bf16.to("cpu")

  with torch.no_grad():
    pt_image_bf16 = pipe_bf16(
        prompt=prompt,
        width=width,
        height=height,
        latents=latents_pt_packed.to(torch.bfloat16),
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images[0]
  pt_image_bf16.save(pt_bf16_path)
  print(f"Saved PyTorch BF16 image to: {pt_bf16_path}")

  del pipe_bf16
  gc.collect()

  # -------------------------------------------------------------------------
  # LEG 3: JAX TPU Bfloat16 (Our Implementation)
  # -------------------------------------------------------------------------
  print("\n" + "=" * 80)
  print("🚀 LEG 3: RUNNING JAX TPU PIPELINE (BF16)...")
  print("=" * 80)

  # Load configs
  from transformers import AutoConfig

  pt_config = AutoConfig.from_pretrained(text_encoder_path, local_files_only=True)

  # Initialize JAX models
  transformer_bf16 = FluxTransformer2DModel(
      in_channels=128,
      num_layers=config.num_double_layers,
      num_single_layers=config.depth,
      attention_head_dim=128,
      num_attention_heads=config.num_attention_heads,
      joint_attention_dim=3 * pt_config.hidden_size,
      pooled_projection_dim=768,
      mlp_ratio=3.0,
      qkv_bias=False,
      joint_attention_bias=False,
      x_embedder_bias=False,
      proj_out_bias=False,
      use_global_modulation=True,
      use_swiglu=True,
      axes_dims_rope=(32, 32, 32, 32),
      theta=2000,
      mesh=mesh,
      dtype=jnp.bfloat16,
      weights_dtype=jnp.bfloat16,
      scale_shift_order=getattr(config, "scale_shift_order", "shift_scale"),
      precision=get_precision(config),
  )

  vae_bf16 = FlaxAutoencoderKL(
      in_channels=3,
      out_channels=3,
      down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
      up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
      block_out_channels=(128, 256, 512, 512),
      layers_per_block=2,
      act_fn="silu",
      latent_channels=32,
      norm_num_groups=32,
      sample_size=512,
      use_quant_conv=True,
      use_post_quant_conv=True,
      dtype=jnp.bfloat16,
  )

  qwen3_config_bf16 = FlaxQwen3Config(
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
  )
  qwen3_model_bf16 = FlaxQwen3Model(qwen3_config_bf16)

  # Initialize parameters and load weights on host CPU
  print("Initializing JAX parameters on CPU...")
  cpu_device = jax.devices("cpu")[0]
  seq_len_txt = 512
  seq_len_img = (height // 16) * (width // 16)  # 1024

  with jax.default_device(cpu_device):
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      key = jax.random.PRNGKey(0)
      key, vae_key, qwen_key = jax.random.split(key, 3)

      # Init Flux
      img_dummy = jnp.zeros((batch_size, seq_len_img, 128))
      img_ids_dummy = jnp.zeros((batch_size, seq_len_img, 4))
      txt_dummy = jnp.zeros((batch_size, seq_len_txt, 3 * pt_config.hidden_size))
      txt_ids_dummy = jnp.zeros((batch_size, seq_len_txt, 4))
      vec_dummy = jnp.zeros((batch_size, 768))
      t_vec_dummy = jnp.zeros((batch_size,))
      guidance_vec_dummy = jnp.zeros((batch_size,))

      variables_bf16 = transformer_bf16.init(
          key,
          hidden_states=img_dummy,
          img_ids=img_ids_dummy,
          encoder_hidden_states=txt_dummy,
          txt_ids=txt_ids_dummy,
          pooled_projections=vec_dummy,
          timestep=t_vec_dummy,
          guidance=guidance_vec_dummy,
      )
      params_bf16 = variables_bf16["params"]

      # Init VAE
      dummy_img = jnp.zeros((batch_size, 3, height, width))
      vae_variables_bf16 = vae_bf16.init(vae_key, dummy_img)
      vae_params_bf16 = vae_variables_bf16["params"]

      # Init Qwen3
      dummy_ids = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
      dummy_mask = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
      qwen3_variables_bf16 = qwen3_model_bf16.init(qwen_key, dummy_ids, dummy_mask)
      qwen3_params_bf16 = qwen3_variables_bf16["params"]

      # Mesh shardings before unboxing
      import flax.linen as nn

      logical_specs = nn.get_partition_spec(variables_bf16)
      transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_specs, mesh, config.logical_axis_rules)
      transformer_shardings_bf16 = flax.core.freeze(transformer_mesh_shardings["params"])

      vae_logical_specs = nn.get_partition_spec(vae_variables_bf16)
      vae_mesh_shardings = nn.logical_to_mesh_sharding(vae_logical_specs, mesh, config.logical_axis_rules)
      vae_shardings_bf16 = flax.core.freeze(vae_mesh_shardings["params"])

      qwen_logical_specs = nn.get_partition_spec(qwen3_variables_bf16)
      qwen_mesh_shardings = nn.logical_to_mesh_sharding(qwen_logical_specs, mesh, config.logical_axis_rules)
      qwen3_shardings_bf16 = flax.core.freeze(qwen_mesh_shardings["params"])

      # Unbox
      import flax.linen.spmd as flax_spmd

      params_bf16 = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          params_bf16,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      params_bf16 = flax.core.unfreeze(params_bf16)

      vae_params_bf16 = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          vae_params_bf16,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      vae_params_bf16 = flax.core.unfreeze(vae_params_bf16)

      qwen3_params_bf16 = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          qwen3_params_bf16,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      qwen3_params_bf16 = flax.core.unfreeze(qwen3_params_bf16)

      # Load safetensors weights
      print(f"Loading transformer safetensors from: {transformer_path}")
      params_bf16 = load_and_convert_weights(transformer_path, params_bf16, config.num_double_layers, config.depth)
      vae_params_bf16, vae_bn_mean_bf16, vae_bn_std_bf16 = load_and_convert_vae_weights(
          vae_safetensors_path, vae_params_bf16
      )

      print(f"Loading text_encoder safetensors from: {text_encoder_path}")
      qwen3_params_bf16 = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params_bf16, qwen3_config_bf16)

      # Cast to bfloat16 in-place
      cast_dict_to_bfloat16_inplace(params_bf16)
      cast_dict_to_bfloat16_inplace(vae_params_bf16)
      cast_dict_to_bfloat16_inplace(qwen3_params_bf16)
      vae_bn_mean_bf16 = vae_bn_mean_bf16.astype(jnp.bfloat16)
      vae_bn_std_bf16 = vae_bn_std_bf16.astype(jnp.bfloat16)

      params_bf16 = flax.core.freeze(params_bf16)
      vae_params_bf16 = flax.core.freeze(vae_params_bf16)
      qwen3_params_bf16 = flax.core.freeze(qwen3_params_bf16)

  # Setup JAX scheduler
  # Empirical mu calculation is defined locally above
  mu = compute_empirical_mu(seq_len_img, num_inference_steps)
  jax_scheduler = FlaxFlowMatchScheduler(
      num_train_timesteps=1000,
      shift=mu,
      sigma_max=1.0,
      sigma_min=0.001,
      inverse_timesteps=False,
      extra_one_step=False,
      reverse_sigmas=False,
      use_dynamic_shifting=True,
      time_shift_type="exponential",
  )
  scheduler_state = jax_scheduler.create_state()
  scheduler_state = jax_scheduler.set_timesteps_ltx2(
      state=scheduler_state, num_inference_steps=num_inference_steps, shift=mu, sigmas=None
  )

  # Position grids
  txt_ids_val = prepare_text_ids(batch_size, seq_len_txt)
  img_ids_val = prepare_latent_image_ids(batch_size, height // 16, width // 16)

  # JIT Compile step functions
  @jax.jit
  def jitted_qwen3_forward(q_params, ids, mask):
    return qwen3_model_bf16.apply({"params": q_params}, input_ids=ids, attention_mask=mask)

  @jax.jit
  def jitted_transformer_step(t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timestep, guidance):
    return transformer_bf16.apply(
        {"params": t_params},
        hidden_states=latents,
        img_ids=img_ids,
        encoder_hidden_states=prompt_embeds,
        txt_ids=txt_ids,
        pooled_projections=vec,
        timestep=timestep,
        guidance=guidance,
    )

  @jax.jit
  def jitted_vae_decode(v_params, latents_unpatched):
    return vae_bf16.apply({"params": v_params}, latents=latents_unpatched, method=vae_bf16.decode)

  # 1. Text Embedding
  print("  Encoding prompt (JAX Qwen3 BF16)...")
  qwen3_params_tpu = jax.device_put(qwen3_params_bf16, qwen3_shardings_bf16)

  from transformers import Qwen2TokenizerFast

  tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
  messages = [{"role": "user", "content": prompt}]
  templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
  inputs = tokenizer(templated_text, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len_txt)
  prompt_ids = jnp.array(inputs["input_ids"])
  prompt_mask = jnp.array(inputs["attention_mask"])

  hidden_states, all_hidden_states = jitted_qwen3_forward(qwen3_params_tpu, prompt_ids, prompt_mask)
  h_9 = all_hidden_states[9]
  h_18 = all_hidden_states[18]
  h_27 = all_hidden_states[27]
  out = jnp.stack([h_9, h_18, h_27], axis=1)
  prompt_embeds_jax_bf16 = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len_txt, 3 * pt_config.hidden_size))
  prompt_embeds_jax_bf16.block_until_ready()

  del qwen3_params_tpu
  gc.collect()

  # 2. Denoising Loop
  print("  Running JAX Flux denoising loop (BF16)...")
  params_tpu = jax.device_put(params_bf16, transformer_shardings_bf16)

  latents_packed_jax = pack_latents(jnp.array(latents_numpy)).astype(jnp.bfloat16)
  guidance_vec_val = jnp.array([4.0] * batch_size)
  vec_val = jnp.zeros((batch_size, 768))

  latents = latents_packed_jax
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    for step_idx in range(num_inference_steps):
      sigmas = scheduler_state.sigmas
      sigma = sigmas[step_idx]
      step_t = jnp.array([sigma * 1000.0])

      model_output = jitted_transformer_step(
          params_tpu,
          latents,
          img_ids_val,
          prompt_embeds_jax_bf16,
          txt_ids_val,
          vec_val,
          step_t,
          guidance_vec_val,
      )
      step_output = jax_scheduler.step(
          state=scheduler_state, model_output=model_output.sample, timestep=step_t[0], sample=latents
      )
      latents = step_output.prev_sample
      scheduler_state = step_output.state

  latents.block_until_ready()
  del params_tpu
  gc.collect()

  # 3. VAE decoding
  print("  VAE Decoding (JAX BF16)...")
  vae_params_tpu = jax.device_put(vae_params_bf16, vae_shardings_bf16)

  vae_bn_mean_seq = vae_bn_mean_bf16.reshape(1, 1, 128)
  vae_bn_std_seq = vae_bn_std_bf16.reshape(1, 1, 128)
  latents_bn = latents * vae_bn_std_seq + vae_bn_mean_seq
  final_latents_unpatched = unpack_latents(latents_bn, batch_size, 32, height, width)

  with mesh:
    jax_image_out_bf16 = jitted_vae_decode(vae_params_tpu, final_latents_unpatched)
    jax_image_out_bf16.sample.block_until_ready()

  jax_image_bf16 = jax_image_out_bf16.sample / 2.0 + 0.5
  jax_image_bf16 = jnp.clip(jax_image_bf16, 0.0, 1.0)
  jax_image_bf16 = jnp.transpose(jax_image_bf16, (0, 2, 3, 1))  # NHWC
  image_jax_bf16 = np.array(jax_image_bf16[0])

  # Save JAX image
  # Scale to 255.0 to write using PIL
  Image.fromarray((image_jax_bf16 * 255.0).astype(np.uint8)).save(jax_bf16_path)
  print(f"Saved JAX BF16 image to: {jax_bf16_path}")

  del vae_params_tpu
  gc.collect()

  # -------------------------------------------------------------------------
  # COMPARISONS & METRICS REPORT
  # -------------------------------------------------------------------------
  print("\n" + "=" * 80)
  print("📊 4B MODEL END-TO-END PARITY REPORT")
  print("=" * 80)

  # Load images back from disk to ensure PIL visual parity alignment
  pt_fp32_np = np.array(Image.open(pt_fp32_path)).astype(np.float32)
  pt_bf16_np = np.array(Image.open(pt_bf16_path)).astype(np.float32)
  jax_bf16_np = np.array(Image.open(jax_bf16_path)).astype(np.float32)

  # 1. PyTorch CPU BF16 vs PyTorch CPU FP32 (Baseline Precision Loss)
  ssim_pt_bf16 = ssim(pt_bf16_np, pt_fp32_np, channel_axis=-1, data_range=255.0)
  rmse_pt_bf16 = np.sqrt(np.mean((pt_bf16_np - pt_fp32_np) ** 2))
  l2_pt_bf16 = np.sqrt(np.sum((pt_bf16_np - pt_fp32_np) ** 2))
  max_err_pt_bf16 = np.max(np.abs(pt_bf16_np - pt_fp32_np))

  # 2. JAX TPU BF16 vs PyTorch CPU FP32 (Our Parity)
  ssim_jax_bf16 = ssim(jax_bf16_np, pt_fp32_np, channel_axis=-1, data_range=255.0)
  rmse_jax_bf16 = np.sqrt(np.mean((jax_bf16_np - pt_fp32_np) ** 2))
  l2_jax_bf16 = np.sqrt(np.sum((jax_bf16_np - pt_fp32_np) ** 2))
  max_err_jax_bf16 = np.max(np.abs(jax_bf16_np - pt_fp32_np))

  # 3. JAX TPU BF16 vs PyTorch CPU BF16 (Direct Parity)
  ssim_direct = ssim(jax_bf16_np, pt_bf16_np, channel_axis=-1, data_range=255.0)
  rmse_direct = np.sqrt(np.mean((jax_bf16_np - pt_bf16_np) ** 2))
  l2_direct = np.sqrt(np.sum((jax_bf16_np - pt_bf16_np) ** 2))
  max_err_direct = np.max(np.abs(jax_bf16_np - pt_bf16_np))

  print("\n--- Leg 2 (PyTorch CPU BF16) vs Leg 1 (PyTorch CPU FP32) Baseline ---")
  print(f"  SSIM:             {ssim_pt_bf16:.6f}")
  print(f"  RMSE:             {rmse_pt_bf16:.4f} / 255")
  print(f"  L2 Distance:      {l2_pt_bf16:.4f}")
  print(f"  Max Absolute Err: {max_err_pt_bf16:.4f} / 255")

  print("\n--- Leg 3 (JAX TPU BF16) vs Leg 1 (PyTorch CPU FP32) Parity ---")
  print(f"  SSIM:             {ssim_jax_bf16:.6f}  (Target: > 0.88)")
  print(f"  RMSE:             {rmse_jax_bf16:.4f} / 255")
  print(f"  L2 Distance:      {l2_jax_bf16:.4f}")
  print(f"  Max Absolute Err: {max_err_jax_bf16:.4f} / 255")

  print("\n--- Leg 3 (JAX TPU BF16) vs Leg 2 (PyTorch CPU BF16) Direct Alignment ---")
  print(f"  SSIM:             {ssim_direct:.6f}")
  print(f"  RMSE:             {rmse_direct:.4f} / 255")
  print(f"  L2 Distance:      {l2_direct:.4f}")
  print(f"  Max Absolute Err: {max_err_direct:.4f} / 255")
  print("=" * 80 + "\n")

  # Log results to markdown file in output dir
  md_report_path = "src/maxdiffusion/tests/flux2klein/flux4b_streamlined_e2e_report.md"
  with open(md_report_path, "w") as f:
    f.write(
        f"""# 📈 Flux.2-klein-4B Streamlined E2E Parity Report
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
Resolution: {width}x{height}
Prompt: '{prompt}'

## 📊 Parity Metrics Summary

| Comparison | SSIM | RMSE | L2 Distance | Max Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch CPU BF16 vs Gold FP32 (Baseline)** | {ssim_pt_bf16:.6f} | {rmse_pt_bf16:.4f} | {l2_pt_bf16:.4f} | {max_err_pt_bf16:.4f} |
| **JAX TPU BF16 vs Gold FP32 (Parity check)** | {ssim_jax_bf16:.6f} | {rmse_jax_bf16:.4f} | {l2_jax_bf16:.4f} | {max_err_jax_bf16:.4f} |
| **JAX TPU BF16 vs PyTorch CPU BF16 (Direct)** | {ssim_direct:.6f} | {rmse_direct:.4f} | {l2_direct:.4f} | {max_err_direct:.4f} |

## 🖼️ Image Artifact paths
*   **Golden PyTorch FP32**: [stunt_4b_pytorch_fp32.png](file://{os.path.abspath(pt_fp32_path)})
*   **PyTorch CPU BF16**: [stunt_4b_pytorch_bf16.png](file://{os.path.abspath(pt_bf16_path)})
*   **JAX TPU BF16**: [stunt_4b_jax_bf16.png](file://{os.path.abspath(jax_bf16_path)})
"""
    )
  print(f"Saved detailed markdown report to: {md_report_path}")


if __name__ == "__main__":
  if os.getenv("GITHUB_ACTIONS") == "true":
    print("Skipping E2E parity test on GitHub Actions (requires TPU HBM and model weights).")
    sys.exit(0)
  run_parity()
