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
import os
import time
import sys
from typing import List

from PIL import Image, UnidentifiedImageError
from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion import pyconfig
from maxdiffusion import max_logging
from maxdiffusion import max_utils
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.train_utils import transformer_engine_context

from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model
from maxdiffusion.models.qwen3_utils import load_and_convert_qwen3_weights
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.pipelines.flux.flux2klein_pipeline import FlaxFlux2KleinPipeline


def partition_prompts(prompt_str: str, batch_size: int) -> List[str]:
  """Splits a prompt string by '||' and replicates/truncates to fill the batch_size."""
  raw_prompts = [p.strip() for p in prompt_str.split("||") if p.strip()]
  if not raw_prompts:
    raw_prompts = ["A detailed vector illustration of a robotic hummingbird"]

  num_prompts = len(raw_prompts)
  if num_prompts == 1:
    return raw_prompts * batch_size
  elif num_prompts <= batch_size:
    reps = batch_size // num_prompts
    active = []
    for p in raw_prompts:
      active.extend([p] * reps)
    if len(active) < batch_size:
      active.extend([raw_prompts[-1]] * (batch_size - len(active)))
    return active
  else:
    max_logging.log(
        f"⚠️ Warning: Found {num_prompts} prompts, but batch_size is {batch_size}. Truncating to the first {batch_size}."
    )
    return raw_prompts[:batch_size]


def encode_prompt(prompt: str, snapshot_dir: str = None, repo_id: str = "black-forest-labs/FLUX.2-klein-4B"):
  """Encodes a prompt string into Qwen3 text embeddings using PyTorch text encoder on CPU."""
  import os
  import torch
  import gc
  from transformers import AutoTokenizer, AutoModelForCausalLM
  from huggingface_hub import snapshot_download

  if snapshot_dir is None:
    snapshot_dir = snapshot_download(repo_id=repo_id)

  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
  tokenizer_path = os.path.join(snapshot_dir, "tokenizer")

  if not os.path.exists(os.path.join(text_encoder_path, "config.json")) or not os.path.exists(tokenizer_path):
    try:
      fb_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
      if not os.path.exists(os.path.join(text_encoder_path, "config.json")):
        text_encoder_path = os.path.join(fb_dir, "text_encoder")
      if not os.path.exists(tokenizer_path):
        tokenizer_path = (
            os.path.join(fb_dir, "tokenizer")
            if os.path.exists(os.path.join(fb_dir, "tokenizer"))
            else os.path.join(fb_dir, "text_encoder")
        )
    except Exception:
      if not os.path.exists(tokenizer_path):
        tokenizer_path = text_encoder_path

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  text_encoder = AutoModelForCausalLM.from_pretrained(text_encoder_path, torch_dtype=torch.float32)
  text_encoder.eval()

  messages = [{"role": "user", "content": prompt}]
  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
  inputs = tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
  with torch.no_grad():
    outputs = text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
    out = torch.stack([outputs.hidden_states[k] for k in (9, 18, 27)], dim=1)
    b, c, s, h = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(b, s, c * h)

  del text_encoder
  gc.collect()
  return prompt_embeds.cpu().numpy()


def main(argv):
  # Enable shardy partitioner for TPU execution
  jax.config.update("jax_use_shardy_partitioner", True)

  # 1. Load configurations
  config_path = "src/maxdiffusion/configs/base_flux2klein.yml"
  custom_overrides = []
  if len(argv) > 1:
    if argv[1].endswith(".yml") or argv[1].endswith(".yaml"):
      config_path = argv[1]
      if len(argv) > 2:
        custom_overrides = argv[2:]
    else:
      custom_overrides = argv[1:]

  max_logging.log(f"Initializing pyconfig with config: {config_path}")
  default_args = [
      None,
      config_path,
      "run_name=flux2klein_generation",
      "output_dir=output/",
  ]
  default_args.extend(custom_overrides)

  is_interactive = any(arg and "interactive=True" in arg.replace(" ", "") for arg in default_args)
  if is_interactive:
    max_logging.log("ℹ️ Interactive mode detected: overriding use_latents=False for dynamic inputs.")
    default_args.append("use_latents=False")

  pyconfig.initialize(default_args)

  # Import modules after jax.distributed.initialize() has run via pyconfig.initialize()

  config = pyconfig.config
  os.makedirs(config.output_dir, exist_ok=True)

  num_devices_to_use = getattr(config, "num_devices", None)
  if num_devices_to_use is not None and num_devices_to_use > 0:
    active_devices = jax.devices()[:num_devices_to_use]
  else:
    active_devices = jax.devices()
  active_device_count = len(active_devices)

  if hasattr(config, "per_device_batch_size") and config.per_device_batch_size > 0:
    calculated_batch_size = int(config.per_device_batch_size * active_device_count)
    assert calculated_batch_size >= 1, (
        f"Calculated global batch_size is {calculated_batch_size}, which is invalid (must be >= 1). "
        f"per_device_batch_size={config.per_device_batch_size} multiplied by active_device_count={active_device_count} "
        f"evaluated to {config.per_device_batch_size * active_device_count}, which truncates to 0. "
        f"Please increase per_device_batch_size or specify an explicit batch_size in your configuration."
    )
    if calculated_batch_size != config.batch_size:
      max_logging.log(
          f"ℹ️ Updating batch_size from {config.batch_size} to {calculated_batch_size} "
          f"based on per_device_batch_size={config.per_device_batch_size} and active_device_count={active_device_count}."
      )
      pyconfig._config.keys["batch_size"] = calculated_batch_size

  # 2. Setup device mesh
  custom_parallelism_set = any(
      any(arg.startswith(f"{k}=") for arg in sys.argv)
      for k in [
          "ici_data_parallelism",
          "ici_fsdp_parallelism",
          "ici_context_parallelism",
          "ici_tensor_parallelism",
      ]
  )

  if not custom_parallelism_set and active_device_count > 1:
    max_logging.log(
        f"ℹ️ Defaulting to Tensor Parallelism: ici_tensor_parallelism={active_device_count} on {active_device_count} TPU devices."
    )
    pyconfig._config.keys["ici_tensor_parallelism"] = active_device_count
    pyconfig._config.keys["ici_data_parallelism"] = 1
    pyconfig._config.keys["ici_fsdp_parallelism"] = 1
    pyconfig._config.keys["ici_context_parallelism"] = 1

  max_logging.log("Setting up JAX device mesh...")
  devices_array = create_device_mesh(config, devices=active_devices)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Check compatibility of batch dimension sharding
  data_size = mesh.shape.get("data", 1)
  fsdp_size = mesh.shape.get("fsdp", 1)
  if config.batch_size % (data_size * fsdp_size) != 0:
    max_logging.log(
        f"⚠️ Warning: batch_size ({config.batch_size}) is not divisible by FSDP*Data mesh size ({fsdp_size * data_size})."
    )
    max_logging.log(
        "  Automatically falling back to sharding batch dimension across 'data' axis only to prevent JAX SPMD errors."
    )
    new_rules = []
    for rule in config.logical_axis_rules:
      if rule[0] in ("activation_batch", "conv_batch"):
        new_rules.append([rule[0], "data"])
      else:
        new_rules.append(rule)
    pyconfig._config.keys["logical_axis_rules"] = tuple(new_rules)

  # 3. Resolve weights repository snapshots
  repo_id = getattr(config, "pretrained_model_name_or_path", None)
  if not repo_id:
    raise ValueError("pretrained_model_name_or_path must be specified in configuration YAML or CLI.")

  use_kv = getattr(config, "use_kv", False)
  if use_kv:
    if repo_id in ("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-4b"):
      max_logging.log("⚠️ Warning: KV cache not supported for 4B model, ignoring use_kv=True.")
      pyconfig._config.keys["use_kv"] = False
    elif repo_id in ("black-forest-labs/FLUX.2-klein-9B", "black-forest-labs/FLUX.2-klein-9b"):
      repo_id = "black-forest-labs/FLUX.2-klein-9b-kv"
      pyconfig._config.keys["pretrained_model_name_or_path"] = repo_id
      max_logging.log(f"ℹ️ use_kv=True: switched pretrained_model_name_or_path to KV model variant: {repo_id}")

  max_logging.log(f"Target model detected: {repo_id}")

  if os.path.exists(repo_id):
    snapshot_dir = repo_id
    max_logging.log(f"Using local model directory: {snapshot_dir}")
  else:
    from huggingface_hub import snapshot_download

    rev = getattr(config, "revision", None)
    if not rev or rev == "refs/pr/95":
      rev = "main"
    try:
      snapshot_dir = snapshot_download(repo_id=repo_id, revision=rev, local_files_only=True)
    except Exception:
      snapshot_dir = snapshot_download(repo_id=repo_id, revision=rev)

  max_logging.log(f"Host {jax.process_index()} using HF snapshot directory: {snapshot_dir}")
  safetensors_path = os.path.join(snapshot_dir, "transformer")
  vae_safetensors_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")

  # 4. Load Qwen3 Config & Setup model layout
  from transformers import AutoConfig
  from maxdiffusion.max_utils import get_flash_block_sizes
  from flax import nnx
  from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
  from maxdiffusion.models.flux.util import load_and_convert_flux_klein_nnx_weights

  pt_config = AutoConfig.from_pretrained(text_encoder_path)

  te_bs = get_flash_block_sizes(
      type(
          "Config",
          (),
          {
              "flash_block_sizes": getattr(config, "text_encoder_flash_block_sizes", {}) or {},
              "attention": getattr(config, "text_encoder_attention", "flash"),
          },
      )()
  )

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
      dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
      attention_kernel=getattr(config, "text_encoder_attention", "flash"),
      flash_block_sizes=te_bs,
      mesh=mesh,
      ulysses_shards=getattr(config, "ulysses_shards", -1),
      ulysses_attention_chunks=getattr(config, "ulysses_attention_chunks", 1),
      max_layer_to_run=getattr(config, "text_encoder_max_layer", 27),
      is_causal=getattr(config, "text_encoder_is_causal", True),
  )
  qwen3_model = FlaxQwen3Model(qwen3_config)

  # Load Transformer config for layer counts if present
  transformer_pt_cfg = {}
  transformer_config_json = os.path.join(safetensors_path, "config.json")
  if os.path.exists(transformer_config_json):
    try:
      import json

      with open(transformer_config_json, "r") as f:
        transformer_pt_cfg = json.load(f)
    except Exception:
      pass

  num_double_layers = getattr(config, "num_double_layers", None) or transformer_pt_cfg.get("num_layers", 5)
  depth = getattr(config, "depth", None) or transformer_pt_cfg.get("num_single_layers", 20)
  num_attention_heads = getattr(config, "num_attention_heads", None) or transformer_pt_cfg.get("num_attention_heads", 24)

  # 5. Instantiate JAX NNXFlux2KleinTransformer2DModel
  transformer = NNXFlux2KleinTransformer2DModel(
      rngs=nnx.Rngs(0),
      in_channels=128,
      num_layers=num_double_layers,
      num_single_layers=depth,
      attention_head_dim=128,
      num_attention_heads=num_attention_heads,
      joint_attention_dim=3 * pt_config.hidden_size,
      pooled_projection_dim=768,
      guidance_embeds=True,
      axes_dim=(32, 32, 32, 32),
      theta=2000.0,
      mlp_ratio=3.0,
      attention_kernel=config.attention,
      flash_min_seq_length=512,
      flash_block_sizes=get_flash_block_sizes(config),
      mesh=mesh,
      dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
      weights_dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
      scale_shift_order=getattr(config, "scale_shift_order", "scale_shift"),
      ulysses_shards=getattr(config, "ulysses_shards", -1),
      ulysses_attention_chunks=getattr(config, "ulysses_attention_chunks", 1),
      use_base2_exp=getattr(config, "use_base2_exp", True),
  )

  # 6. Instantiate JAX NNX VAE
  vae = NNXAutoencoderKLFlux2(
      in_channels=3,
      out_channels=3,
      latent_channels=32,
      block_out_channels=(128, 256, 512, 512),
      layers_per_block=2,
      norm_num_groups=32,
      dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
      param_dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
  )

  # 7. Evaluate shapes & extract mesh shardings
  max_logging.log("Evaluating model shapes and shardings...")
  seq_len_txt = config.max_sequence_length
  dummy_ids = jnp.zeros((config.batch_size, seq_len_txt), dtype=jnp.int32)
  dummy_mask = jnp.zeros((config.batch_size, seq_len_txt), dtype=jnp.int32)

  key = jax.random.PRNGKey(0)
  qwen_key = jax.random.fold_in(key, 1)

  abstract_state = nnx.state(transformer, nnx.Param)
  abstract_vae_state = nnx.state(vae, nnx.Param)

  def qwen3_init_fn():
    return qwen3_model.init(qwen_key, dummy_ids, dummy_mask)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    logical_transformer_specs = nnx.get_partition_spec(abstract_state)
    logical_vae_specs = nnx.get_partition_spec(abstract_vae_state)
    abstract_qwen3_vars = jax.eval_shape(qwen3_init_fn)
    logical_qwen3_specs = nn.get_partition_spec(abstract_qwen3_vars)

    transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
    vae_mesh_shardings = nn.logical_to_mesh_sharding(logical_vae_specs, mesh, config.logical_axis_rules)
    qwen3_mesh_shardings = nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)

  vae_shardings = vae_mesh_shardings
  qwen3_shardings = flax.core.freeze(qwen3_mesh_shardings["params"])
  transformer_shardings = transformer_mesh_shardings

  # 8. Load weights on Host CPU
  max_logging.log("Loading parameters on Host CPU...")
  t_load_start = time.time()
  cpu_device = jax.local_devices(backend="cpu")[0]
  with jax.default_device(cpu_device):
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      import flax.linen.spmd as flax_spmd

      def unbox_fn(x):
        return x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x

      t_sub0 = time.time()
      qwen3_params = jax.tree_util.tree_map(
          unbox_fn, abstract_qwen3_vars["params"], is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned)
      )
      qwen3_params = flax.core.unfreeze(qwen3_params)

      max_logging.log(f" -> [SUB-TIMING 1/3] PyTree unboxing template setup: {time.time() - t_sub0:.2f}s")
      t_sub1 = time.time()

      weight_dtype = jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32

      params = load_and_convert_flux_klein_nnx_weights(
          safetensors_path, abstract_state, num_double_layers, depth, dtype=weight_dtype
      )
      vae_bn_mean, vae_bn_std = load_and_convert_flux2klein_nnx_vae_weights(vae_safetensors_path, vae, dtype=weight_dtype)
      vae_params = nnx.state(vae, nnx.Param)
      qwen3_params = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params, qwen3_config)
      max_logging.log(
          f" -> [SUB-TIMING 2/3] Safetensors loading & key mapping (in target dtype): {time.time() - t_sub1:.4f}s"
      )

      qwen3_params = flax.core.freeze(qwen3_params)

      max_logging.log("\n" + "=" * 80)
      max_logging.log("🚀 Pinning all parameters to TPU HBM permanently...")
      max_logging.log("=" * 80 + "\n")
      t_sub3 = time.time()
      max_logging.log("Putting params on TPU HBM...")
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        params = jax.tree_util.tree_map(max_utils.device_put_replicated, params, transformer_shardings)
        max_logging.log("Putting vae_params on TPU HBM...")
        vae_params = jax.tree_util.tree_map(max_utils.device_put_replicated, vae_params, vae_shardings)
        nnx.update(vae, vae_params)
        max_logging.log("Putting qwen3_params on TPU HBM...")
        qwen3_params = jax.tree_util.tree_map(max_utils.device_put_replicated, qwen3_params, qwen3_shardings)
      max_logging.log(f" -> [SUB-TIMING 3/3] TPU HBM device_put placement: {time.time() - t_sub3:.4f}s")
      max_logging.log("All parameters placed on TPU HBM successfully!")
      gc.collect()
      jax.effects_barrier()

  load_time = time.time() - t_load_start
  max_logging.log(f" -> [TIMING] Total Model Loading & Device Placement: {load_time:.4f} seconds ⏱️\n")

  # 9. Setup FlowMatch Scheduler
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

  # 10. Instantiate and invoke FlaxFlux2KleinPipeline
  max_logging.log("Instantiating JAX FlaxFlux2KleinPipeline...")
  pipeline = FlaxFlux2KleinPipeline(
      transformer=transformer,
      vae=vae,
      text_encoder=qwen3_model,
      tokenizer=None,
      scheduler=scheduler,
      config=config,
      mesh=mesh,
  )

  prompt_str = getattr(config, "prompt", None)
  if not prompt_str:
    raise ValueError("Prompt must be specified in the configuration YAML or passed via CLI prompt='...'")
  active_prompts = partition_prompts(prompt_str, config.batch_size)

  # Parse reference image paths for multi-image editing if provided
  images = None
  image_paths = getattr(config, "image_paths", None)
  if image_paths is not None:
    if isinstance(image_paths, str) and image_paths.strip():
      import ast

      try:
        image_paths = ast.literal_eval(image_paths)
      except Exception:
        image_paths = [p.strip() for p in image_paths.split(",") if p.strip()]
    if isinstance(image_paths, (list, tuple)) and len(image_paths) > 0:
      max_logging.log(f" -> Loading {len(image_paths)} reference image(s) for multi-image editing...")
      images = []
      for p in image_paths:
        try:
          if not os.path.exists(p):
            raise FileNotFoundError(f"Reference image file not found: {p}")
          with Image.open(p) as img_raw:
            ref_w, ref_h = img_raw.size
            if ref_w * ref_h > 1024 * 1024:
              scale = (1024 * 1024 / (ref_w * ref_h)) ** 0.5
              ref_w = int(ref_w * scale)
              ref_h = int(ref_h * scale)
            ref_w = max(16, (ref_w // 16) * 16)
            ref_h = max(16, (ref_h // 16) * 16)
            if (ref_w, ref_h) != img_raw.size:
              img = img_raw.convert("RGB").resize((ref_w, ref_h), Image.Resampling.BICUBIC)
            else:
              img = img_raw.convert("RGB")
            images.append(img)
        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
          max_logging.log(f"❌ Error loading reference image '{p}': {e}")
          raise ValueError(f"Failed to load reference image '{p}': {e}") from e
        except Exception as e:
          max_logging.log(f"❌ Unexpected error loading reference image '{p}': {e}")
          raise ValueError(f"Failed to load reference image '{p}': {e}") from e

  if getattr(config, "interactive", False):
    max_logging.log("\n" + "=" * 80)
    max_logging.log("   BATCHED INTERACTIVE GENERATION MODE ENABLED 🎮")
    max_logging.log("The model has been fully loaded and compiled on the TPU.")
    max_logging.log(f"Batch size: {config.batch_size} parallel images.")
    max_logging.log("Enter prompts separated by '||' (e.g. A cute cat || A red car)")
    max_logging.log("Type 'exit' to quit.")
    max_logging.log("=" * 80)

    image_idx = 1
    while True:
      try:
        user_input = input("\nEnter prompt(s): ")
      except (KeyboardInterrupt, EOFError):
        break
      if user_input.strip().lower() in ("exit", "quit"):
        break
      if not user_input.strip():
        continue

      prompts = partition_prompts(user_input, config.batch_size)
      output_file = f"generated_{image_idx:03d}.png"

      pipeline(
          prompt=prompts,
          params=params,
          vae_params=vae_params,
          qwen3_params=qwen3_params,
          vae_bn_mean=vae_bn_mean,
          vae_bn_std=vae_bn_std,
          transformer_shardings=transformer_shardings,
          vae_shardings=vae_shardings,
          qwen3_shardings=qwen3_shardings,
          height=config.height,
          width=config.width,
          num_inference_steps=config.num_inference_steps,
          batch_size=config.batch_size,
          images=images,
          use_latents=False,
          output_dir=config.output_dir,
          output_name=output_file,
      )
      image_idx += 1
  else:
    # Run one-shot generation
    latents_to_use = None
    use_latents_flag = False
    if getattr(config, "latents_path", ""):
      max_logging.log(f"Loading custom starting noise latents from: {config.latents_path}...")
      latents_to_use = np.load(config.latents_path)
      use_latents_flag = True
      max_logging.log(f" -> Custom latents shape: {latents_to_use.shape} | sum: {latents_to_use.sum():.6f}")

    max_logging.log("\n" + "=" * 80)
    max_logging.log("🚀 Pre-compiling XLA graphs concurrently (AOT Compilation)...")
    max_logging.log("=" * 80)
    aot_time = pipeline.compile_aot_async(
        params=params,
        vae_params=vae_params,
        qwen3_params=qwen3_params,
        vae_bn_mean=vae_bn_mean,
        vae_bn_std=vae_bn_std,
        batch_size=config.batch_size,
        height=config.height,
        width=config.width,
        images=images,
        use_kv=getattr(config, "use_kv", False),
    )

    max_logging.log("\n" + "=" * 80)
    max_logging.log("🚀 Running initial dry run (Warmup Pass) to verify compiled graph execution...")
    max_logging.log("=" * 80)
    _, warmup_trace = pipeline(
        prompt=active_prompts,
        params=params,
        vae_params=vae_params,
        qwen3_params=qwen3_params,
        vae_bn_mean=vae_bn_mean,
        vae_bn_std=vae_bn_std,
        transformer_shardings=transformer_shardings,
        vae_shardings=vae_shardings,
        qwen3_shardings=qwen3_shardings,
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        batch_size=config.batch_size,
        images=images,
        use_latents=use_latents_flag,
        latents=latents_to_use,
        use_kv=getattr(config, "use_kv", False),
        output_dir=config.output_dir,
        output_name="flux2klein_warmup.png",
        warmup=True,
    )
    warmup_time = (
        warmup_trace.get("vae_encode", 0.0)
        + warmup_trace.get("prompt_encoding", 0.0)
        + warmup_trace.get("denoise_loop", 0.0)
        + warmup_trace.get("vae_decode", 0.0)
    )

    num_reps = int(getattr(config, "num_reps", 1))
    max_logging.log("\n" + "=" * 80)
    max_logging.log(f"⏱️ Running timed pass at full TPU speed (num_reps={num_reps})...")
    max_logging.log("=" * 80)

    main_traces = []
    main_times = []

    for rep in range(num_reps):
      rep_str = f" [Rep {rep+1}/{num_reps}]" if num_reps > 1 else ""
      if rep > 0:
        max_logging.log(f"⏱️ Running timed pass{rep_str}...")

      if max_utils.profiler_enabled(config) and rep == 0:
        max_logging.log(f"🚀 XProf / JAX Profiler active! Capturing trace into: {config.tensorboard_dir}")
        with max_utils.Profiler(config, session_name="flux2klein_inference"):
          _, trace_i = pipeline(
              prompt=active_prompts,
              params=params,
              vae_params=vae_params,
              qwen3_params=qwen3_params,
              vae_bn_mean=vae_bn_mean,
              vae_bn_std=vae_bn_std,
              transformer_shardings=transformer_shardings,
              vae_shardings=vae_shardings,
              qwen3_shardings=qwen3_shardings,
              height=config.height,
              width=config.width,
              num_inference_steps=config.num_inference_steps,
              batch_size=config.batch_size,
              images=images,
              use_latents=use_latents_flag,
              latents=latents_to_use,
              use_kv=getattr(config, "use_kv", False),
              output_dir=config.output_dir,
              output_name=f"rep_{rep+1}_{config.output_name}" if num_reps > 1 else config.output_name,
          )
      else:
        _, trace_i = pipeline(
            prompt=active_prompts,
            params=params,
            vae_params=vae_params,
            qwen3_params=qwen3_params,
            vae_bn_mean=vae_bn_mean,
            vae_bn_std=vae_bn_std,
            transformer_shardings=transformer_shardings,
            vae_shardings=vae_shardings,
            qwen3_shardings=qwen3_shardings,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            batch_size=config.batch_size,
            images=images,
            use_latents=use_latents_flag,
            latents=latents_to_use,
            use_kv=getattr(config, "use_kv", False),
            output_dir=config.output_dir,
            output_name=f"rep_{rep+1}_{config.output_name}" if num_reps > 1 else config.output_name,
        )

      tot_time_i = trace_i.get(
          "e2e_pipeline_total",
          trace_i.get("vae_encode", 0.0)
          + trace_i.get("prompt_encoding", 0.0)
          + trace_i.get("denoise_loop", 0.0)
          + trace_i.get("vae_decode", 0.0),
      )
      main_traces.append(trace_i)
      main_times.append(tot_time_i)
      if num_reps > 1:
        vae_enc_str = f" | VAE_Enc={trace_i.get('vae_encode', 0.0):.4f}s" if trace_i.get("vae_encode", 0.0) > 0 else ""
        max_logging.log(
            f"   -> Rep {rep+1}/{num_reps} Completed: Total={tot_time_i:.4f}s{vae_enc_str} | Qwen3={trace_i.get('qwen3_encoding', 0.0):.4f}s | Denoise={trace_i.get('denoise_loop', 0.0):.4f}s | VAE_Dec={trace_i.get('vae_decode', 0.0):.4f}s"
        )

    avg_main_time = sum(main_times) / num_reps
    avg_vae_encode = sum(tr.get("vae_encode", 0.0) for tr in main_traces) / num_reps
    avg_vae_to_qwen3 = sum(tr.get("vae_encode_to_qwen3", 0.0) for tr in main_traces) / num_reps
    avg_start_to_qwen3 = sum(tr.get("start_to_qwen3", 0.0) for tr in main_traces) / num_reps
    avg_prompt_enc = sum(tr.get("qwen3_encoding", tr.get("prompt_encoding", 0.0)) for tr in main_traces) / num_reps
    avg_qwen3_to_denoise = sum(tr.get("qwen3_to_denoise", 0.0) for tr in main_traces) / num_reps
    avg_denoise = sum(tr.get("denoise_loop", 0.0) for tr in main_traces) / num_reps
    avg_denoise_to_vae = sum(tr.get("denoise_to_vae", 0.0) for tr in main_traces) / num_reps
    avg_vae_decode = sum(tr.get("vae_decode", 0.0) for tr in main_traces) / num_reps
    avg_image_saving = sum(tr.get("image_saving", 0.0) for tr in main_traces) / num_reps

    total_cold_start = load_time + aot_time + warmup_time

    max_logging.log("\n" + "=" * 80)
    max_logging.log("📊 FLUX.2-KLEIN COMPLETE LATENCY & TIMING BREAKDOWN")
    max_logging.log("=" * 80)
    max_logging.log(f"1) Model Loading & Placement Time:              {load_time:.4f} seconds ⏱️")
    max_logging.log(f"2) Concurrent AOT XLA Compilation Time:         {aot_time:.4f} seconds ⚡")
    max_logging.log(f"3) Warmup Pass Execution Time:                   {warmup_time:.4f} seconds ⏱️")
    if warmup_trace.get("vae_encode", 0.0) > 0:
      max_logging.log(f"   - VAE Encoding:    {warmup_trace.get('vae_encode', 0.0):.4f}s")
    max_logging.log(f"   - Qwen3 Encoding:  {warmup_trace.get('prompt_encoding', 0.0):.4f}s")
    max_logging.log(f"   - Flux Denoising:  {warmup_trace.get('denoise_loop', 0.0):.4f}s")
    max_logging.log(f"   - VAE Decoding:    {warmup_trace.get('vae_decode', 0.0):.4f}s")
    max_logging.log(f"👉 TOTAL COLD-START TIME (Loading + AOT + Warmup): {total_cold_start:.4f} seconds 🎯")
    rep_label = f" (Average across {num_reps} reps)" if num_reps > 1 else ""
    max_logging.log(f"4) Main Warmed-Up Pass (Pure Inference Latency){rep_label}: {avg_main_time:.4f} seconds ⏱️")
    step_num = 1
    if avg_vae_encode > 0:
      max_logging.log(f"   - {step_num}. VAE Image Encoding:       {avg_vae_encode*1000:.2f} ms ({avg_vae_encode:.4f}s)")
      step_num += 1
      max_logging.log(f"   - {step_num}. VAE -> Qwen3:             {avg_vae_to_qwen3*1000:.2f} ms ({avg_vae_to_qwen3:.4f}s)")
      step_num += 1
    else:
      max_logging.log(
          f"   - {step_num}. Start -> Qwen3:          {avg_start_to_qwen3*1000:.2f} ms ({avg_start_to_qwen3:.4f}s)"
      )
      step_num += 1
    max_logging.log(f"   - {step_num}. Qwen3 Encoding:         {avg_prompt_enc*1000:.2f} ms ({avg_prompt_enc:.4f}s)")
    step_num += 1
    max_logging.log(
        f"   - {step_num}. Qwen3 -> Denoising:      {avg_qwen3_to_denoise*1000:.2f} ms ({avg_qwen3_to_denoise:.4f}s)"
    )
    step_num += 1
    max_logging.log(f"   - {step_num}. Flux Denoising Loop:    {avg_denoise*1000:.2f} ms ({avg_denoise:.4f}s)")
    step_num += 1
    max_logging.log(f"   - {step_num}. Denoising -> VAE:       {avg_denoise_to_vae*1000:.2f} ms ({avg_denoise_to_vae:.4f}s)")
    step_num += 1
    max_logging.log(f"   - {step_num}. VAE Decoding:           {avg_vae_decode*1000:.2f} ms ({avg_vae_decode:.4f}s)")
    step_num += 1
    max_logging.log(f"   - {step_num}. Image Saving:           {avg_image_saving*1000:.2f} ms ({avg_image_saving:.4f}s)")
    max_logging.log(f"   - 👉 TOTAL E2E PIPELINE:     {avg_main_time*1000:.2f} ms ({avg_main_time:.4f}s)")
    max_logging.log("=" * 80)

    max_logging.log("\n=======================================================")
    max_logging.log(f"SUCCESS! Batched generation complete for {config.batch_size} images! 🎨🎉")
    max_logging.log("=======================================================\n")


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
