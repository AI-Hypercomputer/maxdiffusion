import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from flax.traverse_util import flatten_dict, unflatten_dict
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)


def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except:
      out_list.append(item)
  return tuple(out_list)


def load_wan_vae(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True):
  device = jax.devices(device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_path = hf_hub_download(
          pretrained_model_name_or_path, subfolder="vae", filename="diffusion_pytorch_model.safetensors"
      )
    max_logging.log(f"Load and port Wan 2.1 VAE on {device}")

    if ckpt_path is not None:
      tensors = {}
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)
        # Order matters
        renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
        renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
        renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

        renamed_pt_key = renamed_pt_key.replace("conv_in.bias", "conv_in.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv_in.weight", "conv_in.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("conv_out.bias", "conv_out.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv_out.weight", "conv_out.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
        renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
        renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
        renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
        renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv1.weight", "conv1.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv2.weight", "conv2.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("time_conv.bias", "time_conv.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("time_conv.weight", "time_conv.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
        renamed_pt_key = renamed_pt_key.replace("conv_shortcut", "conv_shortcut.conv")
        if "decoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
          renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
        if "encoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
        flax_key = _tuple_str_to_int(flax_key)
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
    else:
      raise FileNotFoundError(f"Path {ckpt_path} was not found")

    return flax_state_dict
