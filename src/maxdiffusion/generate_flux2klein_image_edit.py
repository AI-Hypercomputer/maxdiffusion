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
from typing import List, Union

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from transformers import AutoConfig, Qwen2TokenizerFast

from maxdiffusion import pyconfig
from maxdiffusion import max_logging
from maxdiffusion import max_utils
from maxdiffusion.max_utils import create_device_mesh

from maxdiffusion.models.flux.transformers.transformer_flux_flax import NNXFlux2KleinTransformer2DModel
from maxdiffusion.models.flux.vae.autoencoder_kl_flux2_nnx import (
    NNXAutoencoderKLFlux2,
    load_and_convert_flux2klein_nnx_vae_weights,
)
from maxdiffusion.models.flux.util import load_and_convert_flux_klein_nnx_weights
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.pipelines.flux.flux2klein_image_edit_pipeline import FlaxFlux2KleinImageEditPipeline


def encode_prompt_cpu(prompt: str, snapshot_dir: str):
  """Encodes prompt using PyTorch Qwen3 on CPU and frees memory immediately."""
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM

  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
  tokenizer_path = os.path.join(snapshot_dir, "tokenizer") if os.path.exists(os.path.join(snapshot_dir, "tokenizer")) else text_encoder_path

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


def run_pipeline(config):
  """Instantiates models and executes multi-reference image editing pipeline."""
  os.makedirs(config.output_dir, exist_ok=True)

  num_devices_to_use = getattr(config, "num_devices", None)
  if num_devices_to_use is not None and num_devices_to_use > 0:
    active_devices = jax.devices()[:num_devices_to_use]
  else:
    active_devices = jax.devices()
  active_device_count = len(active_devices)

  devices_array = max_utils.create_device_mesh(config, devices=active_devices)
  mesh = Mesh(devices_array, config.mesh_axes)

  # 1. Resolve Checkpoint Snapshot Paths
  from huggingface_hub import snapshot_download

  repo_id = config.pretrained_model_name_or_path
  max_logging.log(f"Resolving checkpoint for {repo_id}...")
  snapshot_dir = None
  hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
  cache_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}", "snapshots")
  if os.path.exists(cache_dir) and os.listdir(cache_dir):
    snapshot_dir = os.path.join(cache_dir, os.listdir(cache_dir)[0])
  else:
    snapshot_dir = snapshot_download(repo_id=repo_id)

  safetensors_path = os.path.join(snapshot_dir, "transformer")
  vae_safetensors_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
  tokenizer_path = os.path.join(snapshot_dir, "tokenizer") if os.path.exists(os.path.join(snapshot_dir, "tokenizer")) else os.path.join(snapshot_dir, "text_encoder")

  # 2. Encode Prompt on CPU
  prompt = getattr(config, "prompt", "a surreal masterpiece blending all reference images")
  max_logging.log(f"Encoding prompt: '{prompt}'...")
  prompt_embeds = encode_prompt_cpu(prompt, snapshot_dir)
  max_logging.log(f"Prompt encoded successfully! Shape: {prompt_embeds.shape}")

  # 3. Inspect Architecture Config from Checkpoint
  import json
  transformer_config_path = os.path.join(safetensors_path, "config.json")
  if os.path.exists(transformer_config_path):
    with open(transformer_config_path, "r") as f:
      t_cfg = json.load(f)
    num_heads = t_cfg.get("num_attention_heads", 24)
    head_dim = t_cfg.get("attention_head_dim", 128)
    joint_dim = t_cfg.get("joint_attention_dim", 7680)
    num_layers = t_cfg.get("num_layers", 5)
    num_single_layers = t_cfg.get("num_single_layers", 20)
    mlp_ratio = t_cfg.get("mlp_ratio", 3.0)
    axes_dims_rope = tuple(t_cfg.get("axes_dims_rope", [32, 32, 32, 32]))
  else:
    num_heads = 24
    head_dim = 128
    joint_dim = 7680
    num_layers = 5
    num_single_layers = 20
    mlp_ratio = 3.0
    axes_dims_rope = (32, 32, 32, 32)

  weight_dtype = jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32

  # 4. Instantiate NNX Transformer
  max_logging.log(f"Instantiating NNXFlux2KleinTransformer2DModel (layers={num_layers}+{num_single_layers}, heads={num_heads}, dim={num_heads*head_dim})...")
  rngs = nnx.Rngs(0)
  transformer = NNXFlux2KleinTransformer2DModel(
      rngs=rngs,
      patch_size=1,
      in_channels=128,
      num_layers=num_layers,
      num_single_layers=num_single_layers,
      attention_head_dim=head_dim,
      num_attention_heads=num_heads,
      joint_attention_dim=joint_dim,
      pooled_projection_dim=None,
      guidance_embeds=False,
      axes_dim=axes_dims_rope,
      mlp_ratio=mlp_ratio,
      scale_shift_order=getattr(config, "scale_shift_order", "scale_shift"),
      dtype=weight_dtype,
      weights_dtype=weight_dtype,
  )
  t_state = load_and_convert_flux_klein_nnx_weights(
      safetensors_path,
      nnx.state(transformer, nnx.Param),
      num_double_layers=num_layers,
      num_single_layers=num_single_layers,
      dtype=weight_dtype,
  )
  nnx.update(transformer, t_state)

  # 5. Instantiate NNX VAE
  max_logging.log("Instantiating NNXAutoencoderKLFlux2...")
  nnx_vae = NNXAutoencoderKLFlux2(dtype=weight_dtype, param_dtype=weight_dtype)
  bn_mean, bn_std = load_and_convert_flux2klein_nnx_vae_weights(
      vae_safetensors_path, nnx_vae, dtype=weight_dtype
  )

  tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
  scheduler = FlaxFlowMatchScheduler(num_train_timesteps=1000, shift=3.0)

  # 6. Instantiate Image Edit Pipeline
  pipeline = FlaxFlux2KleinImageEditPipeline(
      transformer=transformer,
      vae=nnx_vae,
      text_encoder=None,
      tokenizer=tokenizer,
      scheduler=scheduler,
      config=config,
      mesh=mesh,
      vae_bn_mean=bn_mean,
      vae_bn_std=bn_std,
  )

  # 7. Load Reference Images
  image_paths = getattr(config, "image_paths", [])
  if isinstance(image_paths, str):
    image_paths = [p.strip() for p in image_paths.split(",") if p.strip()]

  images = []
  for p in image_paths:
    if os.path.exists(p):
      images.append(Image.open(p))
    else:
      max_logging.log(f"⚠️ Warning: Reference image path not found: {p}")

  if not images:
    raise ValueError(f"No valid reference images found in `image_paths`: {image_paths}")

  max_logging.log(f"Loaded {len(images)} conditioning reference images.")

  # 8. Run Pipeline Execution
  max_logging.log(f"Starting image edit inference with prompt: '{prompt}'...")

  t_start = time.perf_counter()
  images_out = pipeline(
      prompt=prompt,
      images=images,
      height=getattr(config, "height", 512),
      width=getattr(config, "width", 512),
      num_inference_steps=getattr(config, "num_inference_steps", 4),
      prompt_embeds=prompt_embeds,
      prng_key=jax.random.PRNGKey(getattr(config, "seed", 0)),
      output_type="pil",
  )
  t_end = time.perf_counter()
  max_logging.log(f"Image edit completed in {t_end - t_start:.2f}s! 🚀")

  out_filename = getattr(config, "output_name", "flux2klein_edited_image.png")
  out_path = os.path.join(config.output_dir, out_filename)
  images_out[0].save(out_path)
  max_logging.log(f"🎉 Saved output image to: {out_path}")
  return out_path


def main(argv):
  jax.config.update("jax_use_shardy_partitioner", True)
  pyconfig.initialize(argv)
  run_pipeline(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
