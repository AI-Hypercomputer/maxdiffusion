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

import json
import torch
import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)


LTX_2_0_VIDEO_VAE_RENAME_DICT = {
    # Encoder
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    # Decoder
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
    # Common
    # For all 3D ResNets
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

LTX_2_3_VIDEO_VAE_RENAME_DICT = {
    **LTX_2_0_VIDEO_VAE_RENAME_DICT,
    # Decoder extra blocks
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
}




def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)


def rename_for_ltx2_transformer(key):
  """
  Renames Diffusers LTX-2 keys to MaxDiffusion Flax LTX-2 keys.
  """
  key = key.replace("patchify_proj", "proj_in")
  key = key.replace("audio_patchify_proj", "audio_proj_in")
  key = key.replace("norm_final", "norm_out")
  if "adaLN_modulation_1" in key:
    key = key.replace("adaLN_modulation_1", "scale_shift_table")

  if "caption_modulator_1" in key:
    key = key.replace("caption_modulator_1", "video_a2v_cross_attn_scale_shift_table")
  if "audio_caption_modulator_1" in key:
    key = key.replace("audio_caption_modulator_1", "audio_a2v_cross_attn_scale_shift_table")
  if "audio_norm_final" in key:
    key = key.replace("audio_norm_final", "audio_norm_out")
  if ("audio_ff" in key or "ff" in key) and "proj" in key:
    key = key.replace(".proj", "")
  if "to_out_0" in key:
    key = key.replace("to_out_0", "to_out")

  # Add missing mappings
  key = key.replace("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift")
  key = key.replace("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate")
  key = key.replace("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift")
  key = key.replace("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate")
  key = key.replace("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table")
  key = key.replace("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table")

  # LTX-2.3 specific mappings
  # Handle substrings before they are replaced by shorter patterns below
  key = key.replace("audio_prompt_adaln_single", "audio_prompt_adaln")
  key = key.replace("prompt_adaln_single", "prompt_adaln")
  key = key.replace("audio_prompt_scale_shift_table", "audio_scale_shift_table")
  key = key.replace("prompt_scale_shift_table", "scale_shift_table")

  if "prompt_adaln" in key:
    key = key.replace("prompt_adaln", "caption_projection")
  if "audio_prompt_adaln" in key:
    key = key.replace("audio_prompt_adaln", "audio_caption_projection")
  if "video_text_proj_in" in key:
    key = key.replace("video_text_proj_in", "feature_extractor.video_linear")
  if "audio_text_proj_in" in key:
    key = key.replace("audio_text_proj_in", "feature_extractor.audio_linear")

  key = key.replace("k_norm", "norm_k")
  key = key.replace("q_norm", "norm_q")
  key = key.replace("adaln_single", "time_embed")
  return key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=48):
  block_index = None

  # Handle transformer_blocks_N (underscore) produced by rename_key
  if len(pt_tuple_key) > 0 and "transformer_blocks_" in pt_tuple_key[0]:
    import re

    m = re.match(r"transformer_blocks_(\d+)", pt_tuple_key[0])
    if m:
      block_index = int(m.group(1))
      if scan_layers:
        # Map transformer_blocks_N -> transformer_blocks
        pt_tuple_key = ("transformer_blocks",) + pt_tuple_key[1:]
      else:
        # Map transformer_blocks_N -> transformer_blocks, index
        pt_tuple_key = ("transformer_blocks", str(block_index)) + pt_tuple_key[1:]

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)
  
  # Transpose back caption projections for LTX-2.3 as they are already in JAX format or shouldn't be transposed
  if "caption_projection" in flax_key or "audio_caption_projection" in flax_key:
    if "kernel" in flax_key and flax_tensor.ndim == 2:
      flax_tensor = flax_tensor.T

  flax_key_str = [str(k) for k in flax_key]

  if "scale_shift_table" in flax_key_str:
    if flax_key_str[-1] in ["kernel", "weight"]:
      flax_key_str.pop()

  flax_key = tuple(flax_key_str)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers and block_index is not None:
    if "transformer_blocks" in flax_key:
      if flax_key in flax_state_dict:
        new_tensor = flax_state_dict[flax_key]
      else:
        new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)

      new_tensor = new_tensor.at[block_index].set(flax_tensor)
      flax_tensor = new_tensor

  return flax_key, flax_tensor


def load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=None):
  """
  Loads weights from a sharded safetensors checkpoint or a single file.
  """
  tensors = {}
  
  if filename is not None:
    ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
    if filename.endswith(".safetensors"):
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
    else:
      loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
      for k, v in loaded_state_dict.items():
        tensors[k] = torch2jax(v)
    return tensors

  index_file = "diffusion_pytorch_model.safetensors.index.json"
  try:
    index_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=index_file)
    with open(index_path, "r") as f:
      index_data = json.load(f)
    weight_map = index_data["weight_map"]
    shards = set(weight_map.values())

    for shard_file in shards:
      shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=shard_file)
      with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
  except EntryNotFoundError:
    # Fallback to single file
    filename = "diffusion_pytorch_model.safetensors"
    try:
      ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
    except EntryNotFoundError:
      filename = "diffusion_pytorch_model.bin"
      ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)

    if filename.endswith(".safetensors"):
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
    else:
      loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
      for k, v in loaded_state_dict.items():
        tensors[k] = torch2jax(v)

  return tensors


def load_transformer_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 48,
    scan_layers: bool = True,
    subfolder: str = "transformer",
    filename: str = None,
):
  device = jax.local_devices(backend=device)[0]
  max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")

  with jax.default_device(device):
    # Support sharded loading
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=filename)

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)

    random_flax_state_dict = {}
    for key in flattened_dict:
      random_flax_state_dict[tuple(str(item) for item in key)] = flattened_dict[key]

    for pt_key, tensor in tensors.items():
      if filename == "ltx-2.3-22b-dev.safetensors":
        if not pt_key.startswith("model.diffusion_model."):
          continue
        pt_key = pt_key.replace("model.diffusion_model.", "")
        if pt_key.startswith("audio_embeddings_connector") or pt_key.startswith("video_embeddings_connector"):
          continue

      renamed_pt_key = rename_key(pt_key)
      renamed_pt_key = rename_for_ltx2_transformer(renamed_pt_key)

      pt_tuple_key = tuple(renamed_pt_key.split("."))

      flax_key, flax_tensor = get_key_and_value(
          pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
      )

      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def load_vae_weights(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vae", filename: str = None
):
  device = jax.local_devices(backend=device)[0]

  max_logging.log(f"Load and port {pretrained_model_name_or_path} VAE on {device}")

  with jax.default_device(device):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=filename)

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_eval = flatten_dict(eval_shapes)

    random_flax_state_dict = {}
    for key in flattened_eval:
      random_flax_state_dict[tuple(str(item) for item in key)] = flattened_eval[key]

    needs_vae_prefix = any(key[0] == "vae" for key in random_flax_state_dict)

    for pt_key, tensor in tensors.items():
      # Filter keys for combined checkpoint to avoid noise and memory overhead
      if filename == "ltx-2.3-22b-dev.safetensors":
        if not pt_key.startswith("vae."):
          continue

      # latents_mean and latents_std are nnx.Params and will be loaded correctly.
      new_key = pt_key
      if filename == "ltx-2.3-22b-dev.safetensors":
        for replace_key, rename_to in LTX_2_3_VIDEO_VAE_RENAME_DICT.items():
          new_key = new_key.replace(replace_key, rename_to)

      renamed_pt_key = rename_key(new_key)
      renamed_pt_key = renamed_pt_key.replace("nin_shortcut", "conv_shortcut")

      pt_tuple_key = tuple(renamed_pt_key.split("."))
      # Remove 'vae' prefix to match model structure which expects 'encoder'/'decoder' directly
      if pt_tuple_key[0] == "vae":
        pt_tuple_key = pt_tuple_key[1:]

      pt_list = []
      resnet_index = None

      for i, part in enumerate(pt_tuple_key):
        if "_" in part and part.split("_")[-1].isdigit():
          name = "_".join(part.split("_")[:-1])
          idx = int(part.split("_")[-1])

          if name == "resnets" or name == "block":
            pt_list.append("resnets")
            resnet_index = idx
          elif name == "upsamplers":
            pt_list.append("upsampler")
          elif name in ["down_blocks", "up_blocks", "downsamplers"]:
            pt_list.append(name)
            pt_list.append(str(idx))
          else:
            pt_list.append(part)
        elif part == "upsampler":
          pt_list.append("upsampler")
        elif part in ["conv1", "conv2", "conv", "conv_in", "conv_out", "conv_shortcut"]:
          pt_list.append(part)
          if (
              part != "conv"
              and (i + 1 == len(pt_tuple_key) or pt_tuple_key[i + 1] != "conv")
              and (len(pt_list) < 2 or pt_list[-2] != "conv")
          ):
            pt_list.append("conv")
        else:
          pt_list.append(part)

      pt_tuple_key = tuple(pt_list)

      flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
      flax_key = _tuple_str_to_int(flax_key)


      if resnet_index is not None:
        str_flax_key = tuple([str(x) for x in flax_key])
        if str_flax_key in random_flax_state_dict:
          if flax_key not in flax_state_dict:
            target_shape = random_flax_state_dict[str_flax_key].shape
            flax_state_dict[flax_key] = jnp.zeros(target_shape, dtype=flax_tensor.dtype)
          flax_state_dict[flax_key] = flax_state_dict[flax_key].at[resnet_index].set(flax_tensor)
        else:
          flax_state_dict[flax_key] = flax_tensor
      else:
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
    filtered_eval_shapes = {
        k: v for k, v in flattened_eval.items() if not any("dropout" in str(x) or "rngs" in str(x) for x in k)
    }

    validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def rename_for_ltx2_vocoder(key):
  key = key.replace("ups.", "upsamplers.")
  key = key.replace("resblocks.", "resblocks_")
  key = key.replace("conv_post", "conv_out")
  key = key.replace("conv_pre", "conv_in")
  key = key.replace("act_post", "act_out")
  
  # LTX-2.3 specific mappings for Vocoder
  if "downsample" in key and "lowpass" not in key:
    key = key.replace("downsample", "downsample.lowpass")
    
  return key


def load_vocoder_weights(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vocoder", filename: str = None
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=filename)

  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  for pt_key, tensor in tensors.items():
    if filename and not pt_key.startswith("vocoder."):
      continue
    if filename and pt_key.startswith("vocoder."):
      pt_key = pt_key[len("vocoder."):]
    key = rename_for_ltx2_vocoder(pt_key)
    if filename == "ltx-2.3-22b-dev.safetensors":
      key = key.replace("resblocks_", "resnets.")
    parts = key.split(".")

    if parts[-1] == "weight":
      parts[-1] = "kernel"

    flax_key = _tuple_str_to_int(parts)

    # Skip filter keys as they are derived in NNX model
    if "filter" in flax_key:
      continue

    if flax_key[-1] == "kernel":
      if "upsamplers" in flax_key:
        tensor = tensor.transpose(2, 0, 1)[::-1, :, :]
      else:
        tensor = tensor.transpose(2, 1, 0)
    
    if "mel_stft" in flax_key and ("forward_basis" in flax_key or "inverse_basis" in flax_key):
      tensor = tensor.transpose(2, 1, 0)

    flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)

  validate_flax_state_dict(eval_shapes, flax_state_dict)
  return unflatten_dict(flax_state_dict)


def rename_for_ltx2_connector(key):
  key = key.replace("video_connector", "video_embeddings_connector")
  key = key.replace("audio_connector", "audio_embeddings_connector")
  key = key.replace("text_proj_in", "feature_extractor.linear")

  if "transformer_blocks" in key:
    key = key.replace("transformer_blocks", "stacked_blocks")
    key = key.replace("ff.net.0.proj", "ff.net_0")
    key = key.replace("ff.net.2", "ff.net_2")
    key = key.replace("to_out.0", "to_out")

  if key.endswith(".weight"):
    if "norm_q" in key or "norm_k" in key:
      key = key.replace(".weight", ".scale")
    else:
      key = key.replace(".weight", ".kernel")

  return key


def load_connector_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "connectors",
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device)
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  grouped_weights = {"video_embeddings_connector": {}, "audio_embeddings_connector": {}}

  for pt_key, tensor in tensors.items():
    key = rename_for_ltx2_connector(pt_key)

    if key.endswith(".kernel"):
      if tensor.ndim == 2:
        tensor = tensor.transpose(1, 0)

    if "stacked_blocks" in key:
      parts = key.split(".")
      if "stacked_blocks" in parts:
        sb_index = parts.index("stacked_blocks")
        if sb_index + 1 < len(parts):
          layer_idx = int(parts[sb_index + 1])
          connector = parts[0]

          param_parts = parts[: sb_index + 1] + parts[sb_index + 2 :]
          param_name = tuple(param_parts)

          if connector in grouped_weights:
            if param_name not in grouped_weights[connector]:
              grouped_weights[connector][param_name] = {}
            grouped_weights[connector][param_name][layer_idx] = tensor
            continue

    key_tuple = tuple(key.split("."))
    final_key_tuple = _tuple_str_to_int(key_tuple)

    flax_state_dict[final_key_tuple] = jax.device_put(tensor, device=cpu)

  for connector, params in grouped_weights.items():
    for param_name, layers in params.items():
      sorted_layers = sorted(layers.keys())
      stacked_tensor = jnp.stack([layers[i] for i in sorted_layers], axis=0)

      flax_state_dict[_tuple_str_to_int(param_name)] = jax.device_put(stacked_tensor, device=cpu)

  del tensors
  jax.clear_caches()
  validate_flax_state_dict(eval_shapes, flax_state_dict)
  return unflatten_dict(flax_state_dict)


def rename_for_ltx2_audio_vae(key):
  if key.endswith(".weight"):
    key = key.replace(".weight", ".kernel")

  key = key.replace("mid.block_1", "mid_block1")
  key = key.replace("mid.block_2", "mid_block2")
  key = key.replace("mid.attn_1", "mid_attn")

  key = key.replace("up.", "up_stages.")
  key = key.replace("down.", "down_stages.")

  key = key.replace("block.", "blocks.")

  key = key.replace("nin_shortcut", "conv_shortcut_layer")

  if "upsample.conv.kernel" in key:
    key = key.replace("upsample.conv.kernel", "upsample.conv.conv.kernel")
  if "upsample.conv.bias" in key:
    key = key.replace("upsample.conv.bias", "upsample.conv.conv.bias")

  key = key.replace("per_channel_statistics.mean-of-means", "latents_mean")
  key = key.replace("per_channel_statistics.std-of-means", "latents_std")

  return key


def load_audio_vae_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "audio_vae",
    filename: str = None,
):
  tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=filename)
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  flattened_eval = flatten_dict(eval_shapes)

  for pt_key, tensor in tensors.items():
    if filename and not pt_key.startswith("audio_vae."):
      continue
    if filename and pt_key.startswith("audio_vae."):
      pt_key = pt_key[len("audio_vae."):]
    key = rename_for_ltx2_audio_vae(pt_key)

    if key.endswith(".kernel") and tensor.ndim == 4:
      tensor = tensor.transpose(2, 3, 1, 0)

    flax_key = _tuple_str_to_int(key.split("."))

    if "up_stages" in flax_key:
      up_stages_idx = flax_key.index("up_stages")
      if up_stages_idx + 1 < len(flax_key) and isinstance(flax_key[up_stages_idx + 1], int):
        flax_key_list = list(flax_key)
        flax_key_list[up_stages_idx + 1] = 2 - flax_key[up_stages_idx + 1]
        flax_key = tuple(flax_key_list)

    flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)
  filtered_eval_shapes = {
      k: v for k, v in flattened_eval.items() if not any("dropout" in str(x) or "rngs" in str(x) for x in k)
  }

  validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
  return unflatten_dict(flax_state_dict)



