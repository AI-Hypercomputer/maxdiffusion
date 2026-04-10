import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import unflatten_dict, flatten_dict
from maxdiffusion import max_logging
from ..modeling_flax_pytorch_utils import validate_flax_state_dict, rename_key
from .ltx2_utils import load_sharded_checkpoint
from .ltx2_utils import (
    _tuple_str_to_int,
    rename_for_ltx2_transformer,
    get_key_and_value,
    rename_for_ltx2_audio_vae,
    rename_for_ltx2_vocoder,
)
def load_ltx2_3_checkpoint(pretrained_model_name_or_path: str, subfolder: str, device: str, filename: str):
  """Loads weights from a single safetensors file for LTX-2.3."""
  from huggingface_hub import hf_hub_download
  from safetensors import safe_open
  from ..modeling_flax_pytorch_utils import torch2jax

  ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
  tensors = {}
  with safe_open(ckpt_path, framework="pt") as f:
    for k in f.keys():
      tensors[k] = torch2jax(f.get_tensor(k))
  return tensors


def rename_for_ltx2_3_transformer(key):
  """
  Renames Diffusers LTX-2.3 keys to MaxDiffusion Flax LTX-2.3 keys.
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

def rename_for_ltx2_3_vocoder(key):
  """Renames Diffusers LTX-2.3 Vocoder keys to MaxDiffusion Flax keys."""
  key = key.replace("ups.", "upsamplers.")
  key = key.replace("resblocks.", "resblocks_")
  key = key.replace("conv_post", "conv_out")
  key = key.replace("conv_pre", "conv_in")
  key = key.replace("act_post", "act_out")
  
  # LTX-2.3 specific mappings for Vocoder
  if "downsample" in key and "lowpass" not in key:
    key = key.replace("downsample", "downsample.lowpass")
    
  return key


LTX_2_3_CONNECTORS_KEYS_RENAME_DICT = {
    "model.diffusion_model.": "",
    "connectors.": "",
    "transformer_1d_blocks": "stacked_blocks",
    "text_embedding_projection.audio_aggregate_embed.weight": "audio_text_proj_in.kernel",
    "text_embedding_projection.audio_aggregate_embed.bias": "audio_text_proj_in.bias",
    "text_embedding_projection.video_aggregate_embed.weight": "video_text_proj_in.kernel",
    "text_embedding_projection.video_aggregate_embed.bias": "video_text_proj_in.bias",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "norm_q.weight": "norm_q.scale",
    "norm_k.weight": "norm_k.scale",
    "to_q.weight": "to_q.kernel",
    "to_k.weight": "to_k.kernel",
    "to_v.weight": "to_v.kernel",
    "to_out.0.weight": "to_out.kernel",
    "to_out.0.bias": "to_out.bias",
    "ff.net.0.proj.weight": "ff.net_0.kernel",
    "ff.net.0.proj.bias": "ff.net_0.bias",
    "ff.net.2.weight": "ff.net_2.kernel",
    "ff.net.2.bias": "ff.net_2.bias",
    "to_gate_logits.weight": "to_gate_logits.kernel",
    "audio_linear.weight": "audio_text_proj_in.kernel",
    "audio_linear.bias": "audio_text_proj_in.bias",
    "video_linear.weight": "video_text_proj_in.kernel",
    "video_linear.bias": "video_text_proj_in.bias",
}

LTX_2_3_ONLY_RENAME_DICT = {
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
}

def load_and_segregate_ltx2_3_weights(pretrained_model_name_or_path: str, filename: str = "ltx-2.3-22b-dev.safetensors"):
  """Loads the full LTX-2.3 file once and splits it into component-specific dictionaries."""
  tensors = load_ltx2_3_checkpoint(pretrained_model_name_or_path, "", "cpu", filename=filename)
  
  segregated = {
      "transformer": {},
      "vae": {},
      "audio_vae": {},
      "connectors": {},
      "vocoder": {},
  }
  
  for pt_key, tensor in tensors.items():
      if pt_key.startswith("model.diffusion_model."):
          segregated["transformer"][pt_key.replace("model.diffusion_model.", "")] = tensor
      elif pt_key.startswith("audio_vae."):
          segregated["audio_vae"][pt_key.replace("audio_vae.", "")] = tensor
      elif pt_key.startswith("vae."):
          segregated["vae"][pt_key] = tensor
      elif pt_key.startswith("vocoder."):
          segregated["vocoder"][pt_key.replace("vocoder.", "")] = tensor
      elif any(x in pt_key for x in ["connectors.", "video_embeddings_connector", "audio_embeddings_connector", "text_embedding_projection"]):
          segregated["connectors"][pt_key] = tensor
          
  return segregated


def load_transformer_weights_2_3(
    eval_shapes: dict,
    device: str,
    tensors: dict,
    num_layers: int = 48,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  max_logging.log(f"Load and port LTX-2.3 transformer on {device}")

  with jax.default_device(device):
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)

    random_flax_state_dict = {}
    for key in flattened_dict:
      random_flax_state_dict[tuple(str(item) for item in key)] = flattened_dict[key]

    for pt_key, tensor in tensors.items():
      # Keys are already filtered and stripped of "model.diffusion_model." by load_and_segregate
      if pt_key.startswith("audio_embeddings_connector") or pt_key.startswith("video_embeddings_connector"):
        continue

      renamed_pt_key = rename_key(pt_key)
      renamed_pt_key = rename_for_ltx2_3_transformer(renamed_pt_key)

      pt_tuple_key = tuple(renamed_pt_key.split("."))

      flax_key, flax_tensor = get_key_and_value(
          pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
      )

      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    jax.clear_caches()
    return flax_state_dict


def load_audio_vae_weights_2_3(
    eval_shapes: dict,
    device: str,
    tensors: dict,
):
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  flattened_eval = flatten_dict(eval_shapes)

  for pt_key, tensor in tensors.items():
    # Keys are already filtered and stripped of "audio_vae." by load_and_segregate
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


def load_vae_weights_2_3(
    eval_shapes: dict,
    device: str,
    tensors: dict,
):
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]
  flattened_eval = flatten_dict(eval_shapes)

  random_flax_state_dict = {}
  for key in flattened_eval:
    random_flax_state_dict[tuple(str(item) for item in key)] = flattened_eval[key]

  for pt_key, tensor in tensors.items():
    # Remove 'vae.' prefix if present in safetensors but not in model
    if pt_key.startswith("vae."):
      pt_key = pt_key[len("vae."):]
      
    if pt_key == "per_channel_statistics.mean-of-means":
      pt_key = "latents_mean"
    elif pt_key == "per_channel_statistics.std-of-means":
      pt_key = "latents_std"
      
    renamed_pt_key = pt_key.replace("nin_shortcut", "conv_shortcut")
    renamed_pt_key = rename_key(renamed_pt_key)

    pt_tuple_key = tuple(renamed_pt_key.split("."))

    decoder_mapping = {
        "up_blocks_0": "mid_block",
        "up_blocks_1": "up_blocks_0.upsamplers_0",
        "up_blocks_2": "up_blocks_0",
        "up_blocks_3": "up_blocks_1.upsamplers_0",
        "up_blocks_4": "up_blocks_1",
        "up_blocks_5": "up_blocks_2.upsamplers_0",
        "up_blocks_6": "up_blocks_2",
        "up_blocks_7": "up_blocks_3.upsamplers_0",
        "up_blocks_8": "up_blocks_3",
    }

    encoder_mapping = {
        "down_blocks_0": "down_blocks_0",
        "down_blocks_1": "down_blocks_0.downsamplers_0",
        "down_blocks_2": "down_blocks_1",
        "down_blocks_3": "down_blocks_1.downsamplers_0",
        "down_blocks_4": "down_blocks_2",
        "down_blocks_5": "down_blocks_2.downsamplers_0",
        "down_blocks_6": "down_blocks_3",
        "down_blocks_7": "down_blocks_3.downsamplers_0",
        "down_blocks_8": "mid_block",
    }

    mapped_pt_list = []
    for part in pt_tuple_key:
      if part in decoder_mapping:
        mapped_pt_list.extend(decoder_mapping[part].split("."))
      elif part in encoder_mapping:
        mapped_pt_list.extend(encoder_mapping[part].split("."))
      else:
        mapped_pt_list.append(part)
    
    pt_tuple_key = tuple(mapped_pt_list)

    pt_list = []
    resnet_index = None

    for i, part in enumerate(pt_tuple_key):
      if "_" in part and part.split("_")[-1].isdigit():
        name = "_".join(part.split("_")[:-1])
        idx = int(part.split("_")[-1])

        if name == "resnets" or name == "block" or name == "res_blocks":
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

    from .ltx2_utils import rename_key_and_reshape_tensor
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
  return unflatten_dict(flax_state_dict)


def load_vocoder_weights_2_3(
    eval_shapes: dict,
    device: str,
    tensors: dict,
):
  flax_state_dict = {}
  cpu = jax.local_devices(backend="cpu")[0]

  for pt_key, tensor in tensors.items():
    # Keys are already filtered and stripped of "vocoder." by load_and_segregate
    key = rename_for_ltx2_3_vocoder(pt_key)
    
    # Always apply LTX-2.3 specific replacement
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


def load_connectors_weights_2_3(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "",
    filename: str = None,
    is_ltx2_3: bool = False,
    tensors: dict = None,
):
  device = jax.local_devices(backend=device)[0]

  with jax.default_device(device):
    if tensors is None:
      tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=filename)
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_eval = flatten_dict(eval_shapes)

    accumulated_stacked = {}

    for pt_key, tensor in tensors.items():
      if not any(x in pt_key for x in ["connectors.", "video_embeddings_connector", "audio_embeddings_connector", "text_embedding_projection"]):
        continue

      flax_key_str = pt_key
      for replace_key, rename_to in LTX_2_3_CONNECTORS_KEYS_RENAME_DICT.items():
        flax_key_str = flax_key_str.replace(replace_key, rename_to)

      if is_ltx2_3:
        for replace_key, rename_to in LTX_2_3_ONLY_RENAME_DICT.items():
          flax_key_str = flax_key_str.replace(replace_key, rename_to)

      segments = flax_key_str.split(".")
      
      # Only extract digit if it immediately follows 'stacked_blocks'
      layer_idx = None
      base_segments = []
      i = 0
      while i < len(segments):
        seg = segments[i]
        if seg == "stacked_blocks" and i + 1 < len(segments) and segments[i+1].isdigit():
          base_segments.append(seg)
          layer_idx = int(segments[i+1])
          i += 2
        else:
          base_segments.append(seg)
          i += 1
          
      if layer_idx is not None:
        base_key = _tuple_str_to_int(base_segments)
        if base_key not in accumulated_stacked:
          accumulated_stacked[base_key] = {}
        
        # Transpose FF and gate kernels to match Flax layout (in, out)
        if ("ff" in base_segments or "to_gate_logits" in base_segments) and base_segments[-1] == "kernel":
          tensor = jnp.transpose(tensor, (1, 0))
          
        accumulated_stacked[base_key][layer_idx] = tensor
      else:
        # Transpose projection kernels in feature extractor or new LTX-2.3 projections
        if any(x in segments for x in ["feature_extractor", "audio_text_proj_in", "video_text_proj_in"]) and segments[-1] == "kernel":
          tensor = jnp.transpose(tensor, (1, 0))
          
        flax_key = _tuple_str_to_int(segments)
        flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)

    # Now stack the accumulated ones
    for base_key, layers in accumulated_stacked.items():
      num_layers = max(layers.keys()) + 1
      if len(layers) != num_layers:
        raise ValueError(f"Missing layers for {base_key}, got {layers.keys()}")
        
      sorted_tensors = [layers[i] for i in range(num_layers)]
      stacked_tensor = jnp.stack(sorted_tensors, axis=0)
      flax_state_dict[base_key] = jax.device_put(stacked_tensor, device=cpu)

    filtered_eval_shapes = {
        k: v for k, v in flattened_eval.items() if not any("dropout" in str(x) or "rngs" in str(x) for x in k)
    }
    validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
    return unflatten_dict(flax_state_dict)
