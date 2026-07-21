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

from maxdiffusion.models.flux.transformers.transformer_flux_flax import Flux2KleinTransformer2DModel
from maxdiffusion.models.vae_flax import FlaxAutoencoderKL
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model, load_and_convert_qwen3_weights
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler


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
        tokenizer_path = os.path.join(fb_dir, "tokenizer") if os.path.exists(os.path.join(fb_dir, "tokenizer")) else os.path.join(fb_dir, "text_encoder")
    except Exception as e:
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
  from maxdiffusion.models.flux.util import (
      load_and_convert_flux_klein_weights,
      load_and_convert_vae_weights,
      cast_dict_to_bfloat16_inplace,
  )
  from maxdiffusion.pipelines.flux.flux2klein_pipeline import FlaxFlux2KleinPipeline

  config = pyconfig.config
  os.makedirs(config.output_dir, exist_ok=True)

  # 2. Setup device mesh
  if config.batch_size == 1 and config.ici_tensor_parallelism == 1 and jax.device_count() > 1:
    max_logging.log(
        f"ℹ️ Auto-configuring Tensor Parallelism: ici_tensor_parallelism={jax.device_count()}, ici_fsdp_parallelism=1 for batch_size=1 on {jax.device_count()} TPU devices."
    )
    pyconfig._config.keys["ici_tensor_parallelism"] = jax.device_count()
    pyconfig._config.keys["ici_fsdp_parallelism"] = 1

  max_logging.log("Setting up JAX device mesh...")
  devices_array = create_device_mesh(config)
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
    depth_val = getattr(config, "depth", None)
    repo_id = "black-forest-labs/FLUX.2-klein-9B" if depth_val == 24 else "black-forest-labs/FLUX.2-klein-4B"
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
      snapshot_dir = snapshot_download(repo_id=repo_id, local_files_only=True)

  max_logging.log(f"Host {jax.process_index()} using HF snapshot directory: {snapshot_dir}")
  safetensors_path = os.path.join(snapshot_dir, "transformer")
  vae_safetensors_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")

  # 4. Load Qwen3 Config & Setup model layout
  from transformers import AutoConfig

  try:
    pt_config = AutoConfig.from_pretrained(text_encoder_path, local_files_only=True)
  except Exception as e:
    depth_val = getattr(config, "depth", 24)
    hf_repo = "black-forest-labs/FLUX.2-klein-9B" if depth_val in (24, -1) else "black-forest-labs/FLUX.2-klein-4B"
    max_logging.log(f"ℹ️ Config not found in {text_encoder_path}. Resolving from HF cache: {hf_repo}")
    pt_config = AutoConfig.from_pretrained(hf_repo, subfolder="text_encoder", local_files_only=True)

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
  )
  qwen3_model = FlaxQwen3Model(qwen3_config)

  # Load Transformer HF config.json directly for model architecture parameters
  import json

  transformer_config_json = os.path.join(safetensors_path, "config.json")
  transformer_pt_cfg = {}
  loaded_cfg = False
  if os.path.exists(transformer_config_json):
    try:
      with open(transformer_config_json, "r") as f:
        transformer_pt_cfg = json.load(f)
        loaded_cfg = True
    except Exception as e:
      max_logging.log(f"ℹ️ Could not parse {transformer_config_json}: {e}. Falling back to HF cache...")

  if not loaded_cfg:
    depth_val = getattr(config, "depth", 24)
    hf_repo = "black-forest-labs/FLUX.2-klein-9B" if depth_val in (24, -1) else "black-forest-labs/FLUX.2-klein-4B"
    try:
      from huggingface_hub import hf_hub_download
      cfg_file = hf_hub_download(repo_id=hf_repo, filename="transformer/config.json", local_files_only=True)
      with open(cfg_file, "r") as f:
        transformer_pt_cfg = json.load(f)
    except Exception as e:
      max_logging.log(f"⚠️ Warning resolving transformer config fallback: {e}")

  num_double_layers = getattr(config, "num_double_layers", -1)
  if num_double_layers is None or num_double_layers <= 0:
    num_double_layers = transformer_pt_cfg.get("num_layers", 5)

  depth = getattr(config, "depth", -1)
  if depth is None or depth <= 0:
    depth = transformer_pt_cfg.get("num_single_layers", 20)

  num_attention_heads = getattr(config, "num_attention_heads", -1)
  if num_attention_heads is None or num_attention_heads <= 0:
    num_attention_heads = transformer_pt_cfg.get("num_attention_heads", 24)

  # 5. Instantiate JAX Flux2KleinTransformer2DModel
  transformer = Flux2KleinTransformer2DModel(
      in_channels=128,
      num_layers=num_double_layers,
      num_single_layers=depth,
      attention_head_dim=128,
      num_attention_heads=num_attention_heads,
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
      dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
      weights_dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
      attention_kernel=config.attention,
      scale_shift_order=getattr(config, "scale_shift_order", "shift_scale"),
  )

  # 6. Instantiate JAX VAE
  vae = FlaxAutoencoderKL(
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
      dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
  )

  # 7. Evaluate shapes & extract mesh shardings
  max_logging.log("Evaluating model shapes and shardings...")
  h_packed = config.height // 16
  w_packed = config.width // 16
  seq_len_img = h_packed * w_packed
  seq_len_txt = config.max_sequence_length

  img_dummy = jnp.zeros((config.batch_size, seq_len_img, 128))
  img_ids_dummy = jnp.zeros((config.batch_size, seq_len_img, 4))
  txt_dummy = jnp.zeros((config.batch_size, seq_len_txt, 3 * pt_config.hidden_size))
  txt_ids_dummy = jnp.zeros((config.batch_size, seq_len_txt, 4))
  vec_dummy = jnp.zeros((config.batch_size, 768))
  t_vec_dummy = jnp.zeros((config.batch_size,))
  guidance_vec_dummy = jnp.zeros((config.batch_size,))
  dummy_img = jnp.zeros((config.batch_size, 3, 512, 512))
  dummy_ids = jnp.zeros((config.batch_size, seq_len_txt), dtype=jnp.int32)
  dummy_mask = jnp.zeros((config.batch_size, seq_len_txt), dtype=jnp.int32)

  key = jax.random.PRNGKey(0)
  key, vae_key, qwen_key = jax.random.split(key, 3)

  def transformer_init_fn():
    return transformer.init(
        key,
        hidden_states=img_dummy,
        img_ids=img_ids_dummy,
        encoder_hidden_states=txt_dummy,
        txt_ids=txt_ids_dummy,
        pooled_projections=vec_dummy,
        timestep=t_vec_dummy,
        guidance=guidance_vec_dummy,
    )

  def vae_init_fn():
    return vae.init(vae_key, dummy_img)

  def qwen3_init_fn():
    return qwen3_model.init(qwen_key, dummy_ids, dummy_mask)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_transformer_vars = jax.eval_shape(transformer_init_fn)
    abstract_vae_vars = jax.eval_shape(vae_init_fn)
    abstract_qwen3_vars = jax.eval_shape(qwen3_init_fn)

    logical_transformer_specs = nn.get_partition_spec(abstract_transformer_vars)
    logical_vae_specs = nn.get_partition_spec(abstract_vae_vars)
    logical_qwen3_specs = nn.get_partition_spec(abstract_qwen3_vars)

    transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
    vae_mesh_shardings = nn.logical_to_mesh_sharding(logical_vae_specs, mesh, config.logical_axis_rules)
    qwen3_mesh_shardings = nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)

  transformer_shardings = flax.core.freeze(transformer_mesh_shardings["params"])
  vae_shardings = flax.core.freeze(vae_mesh_shardings["params"])
  qwen3_shardings = flax.core.freeze(qwen3_mesh_shardings["params"])

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
      params = jax.tree_util.tree_map(
          unbox_fn, abstract_transformer_vars["params"], is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned)
      )
      params = flax.core.unfreeze(params)

      vae_params = jax.tree_util.tree_map(
          unbox_fn, abstract_vae_vars["params"], is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned)
      )
      vae_params = flax.core.unfreeze(vae_params)

      qwen3_params = jax.tree_util.tree_map(
          unbox_fn, abstract_qwen3_vars["params"], is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned)
      )
      qwen3_params = flax.core.unfreeze(qwen3_params)

      max_logging.log(f" -> [SUB-TIMING 1/3] PyTree unboxing template setup: {time.time() - t_sub0:.2f}s")
      t_sub1 = time.time()

      weight_dtype = jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32

      params = load_and_convert_flux_klein_weights(safetensors_path, params, num_double_layers, depth, dtype=weight_dtype)
      vae_params, vae_bn_mean, vae_bn_std = load_and_convert_vae_weights(vae_safetensors_path, vae_params, dtype=weight_dtype)
      qwen3_params = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params, qwen3_config, dtype=weight_dtype)
      max_logging.log(f" -> [SUB-TIMING 2/3] Safetensors loading & key mapping (in target dtype): {time.time() - t_sub1:.2f}s")

      params = flax.core.freeze(params)
      vae_params = flax.core.freeze(vae_params)
      qwen3_params = flax.core.freeze(qwen3_params)

      max_logging.log("\n" + "=" * 80)
      max_logging.log("🚀 Pinning all parameters to TPU HBM permanently...")
      max_logging.log("=" * 80 + "\n")
      t_sub3 = time.time()
      max_logging.log("Putting params on TPU HBM...")
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        try:
          params = jax.tree_util.tree_map(max_utils.device_put_replicated, params, transformer_shardings)
        except Exception as err:
          max_logging.log("\n❌ jax.device_put(params, transformer_shardings) FAILED!")
          flat_p = flax.traverse_util.flatten_dict(params)
          flat_s = flax.traverse_util.flatten_dict(transformer_shardings)
          k_p = set(flat_p.keys())
          k_s = set(flat_s.keys())
          max_logging.log(f"Keys in sharding spec but missing in params: {k_s - k_p}")
          max_logging.log(f"Keys in params but missing in sharding spec: {k_p - k_s}")
          sys.stdout.flush()
          raise err
        max_logging.log("Putting vae_params on TPU HBM...")
        vae_params = jax.tree_util.tree_map(max_utils.device_put_replicated, vae_params, vae_shardings)
        max_logging.log("Putting qwen3_params on TPU HBM...")
        qwen3_params = jax.tree_util.tree_map(max_utils.device_put_replicated, qwen3_params, qwen3_shardings)
      max_logging.log(f" -> [SUB-TIMING 3/3] TPU HBM device_put placement: {time.time() - t_sub3:.2f}s")
      max_logging.log("All parameters placed on TPU HBM successfully!")
      gc.collect()
      jax.effects_barrier()

  load_time = time.time() - t_load_start
  max_logging.log(f" -> [TIMING] Total Model Loading & Device Placement: {load_time:.2f} seconds ⏱️\n")

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

  active_prompts = partition_prompts(config.prompt, config.batch_size)

  if getattr(config, "interactive", False):
    print("\n" + "=" * 80)
    print("   BATCHED INTERACTIVE GENERATION MODE ENABLED 🎮")
    print("The model has been fully loaded and compiled on the TPU.")
    print(f"Batch size: {config.batch_size} parallel images.")
    print("Enter prompts separated by '||' (e.g. A cute cat || A red car)")
    print("Type 'exit' to quit.")
    print("=" * 80)

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
        use_latents=use_latents_flag,
        latents=latents_to_use,
        output_dir=config.output_dir,
        output_name="flux2klein_warmup.png",
    )
    warmup_time = (
        warmup_trace.get("prompt_encoding", 0.0)
        + warmup_trace.get("denoise_loop", 0.0)
        + warmup_trace.get("vae_decode", 0.0)
    )

    max_logging.log("\n" + "=" * 80)
    max_logging.log("⏱️ Running timed pass at full TPU speed...")
    max_logging.log("=" * 80)
    _, main_trace = pipeline(
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
        use_latents=use_latents_flag,
        latents=latents_to_use,
        output_dir=config.output_dir,
        output_name=config.output_name,
    )
    main_time = (
        main_trace.get("prompt_encoding", 0.0) + main_trace.get("denoise_loop", 0.0) + main_trace.get("vae_decode", 0.0)
    )

    total_cold_start = load_time + aot_time + warmup_time

    max_logging.log("\n" + "=" * 80)
    max_logging.log("📊 FLUX.2-KLEIN COMPLETE LATENCY & TIMING BREAKDOWN")
    max_logging.log("=" * 80)
    max_logging.log(f"1) Model Loading & Placement Time:              {load_time:.2f} seconds ⏱️")
    max_logging.log(f"2) Concurrent AOT XLA Compilation Time:         {aot_time:.2f} seconds ⚡")
    max_logging.log(f"3) Warmup Pass Execution Time:                   {warmup_time:.2f} seconds ⏱️")
    max_logging.log(f"   - Qwen3 Encoding:  {warmup_trace.get('prompt_encoding', 0.0):.2f}s")
    max_logging.log(f"   - Flux Denoising:  {warmup_trace.get('denoise_loop', 0.0):.2f}s")
    max_logging.log(f"   - VAE Decoding:    {warmup_trace.get('vae_decode', 0.0):.2f}s")
    max_logging.log(f"👉 TOTAL COLD-START TIME (Loading + AOT + Warmup): {total_cold_start:.2f} seconds 🎯")
    max_logging.log(f"4) Main Warmed-Up Pass (Pure Inference Latency): {main_time:.2f} seconds ⏱️")
    max_logging.log(f"   - Qwen3 Encoding:  {main_trace.get('prompt_encoding', 0.0):.2f}s")
    max_logging.log(f"   - Flux Denoising:  {main_trace.get('denoise_loop', 0.0):.2f}s")
    max_logging.log(f"   - VAE Decoding:    {main_trace.get('vae_decode', 0.0):.2f}s")
    max_logging.log("=" * 80)

    max_logging.log("\n=======================================================")
    max_logging.log(f"SUCCESS! Batched generation complete for {config.batch_size} images! 🎨🎉")
    max_logging.log("=======================================================\n")


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
