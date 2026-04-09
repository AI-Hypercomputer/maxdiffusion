import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import unflatten_dict, flatten_dict
from maxdiffusion import max_logging
from ..modeling_flax_pytorch_utils import validate_flax_state_dict
from .ltx2_utils import load_sharded_checkpoint
from .ltx2_utils import (
    _tuple_str_to_int,
    LTX_2_0_VIDEO_VAE_RENAME_DICT,
)



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
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
}

def load_connectors_weights(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    subfolder: str = "",
    filename: str = None,
):
  device = jax.local_devices(backend=device)[0]

  with jax.default_device(device):
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
        # Transpose projection kernels in feature extractor
        if "feature_extractor" in segments and segments[-1] == "kernel":
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
