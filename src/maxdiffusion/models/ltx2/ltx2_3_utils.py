import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import unflatten_dict, flatten_dict
from maxdiffusion import max_logging
from ..modeling_flax_pytorch_utils import (
    load_sharded_checkpoint,
    validate_flax_state_dict,
)
from maxdiffusion.models.ltx2.ltx2_utils import (
    _tuple_str_to_int,
    LTX_2_0_VIDEO_VAE_RENAME_DICT,
)

LTX_2_3_VIDEO_VAE_RENAME_DICT = {
    **LTX_2_0_VIDEO_VAE_RENAME_DICT,
    # Decoder extra blocks
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
}

LTX_2_3_CONNECTORS_KEYS_RENAME_DICT = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.audio_aggregate_embed": "audio_text_proj_in",
    "text_embedding_projection.video_aggregate_embed": "video_text_proj_in",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
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
  max_logging.log(f"Load and port {pretrained_model_name_or_path} Connectors on {device}")

  with jax.default_device(device):
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=filename)
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_eval = flatten_dict(eval_shapes)

    for pt_key, tensor in tensors.items():
      if not any(x in pt_key for x in ["connectors.", "video_embeddings_connector", "audio_embeddings_connector"]):
        continue

      flax_key_str = pt_key
      for replace_key, rename_to in LTX_2_3_CONNECTORS_KEYS_RENAME_DICT.items():
        flax_key_str = flax_key_str.replace(replace_key, rename_to)

      flax_key = _tuple_str_to_int(flax_key_str.split("."))
      flax_state_dict[flax_key] = jax.device_put(tensor, device=cpu)

    filtered_eval_shapes = {
        k: v for k, v in flattened_eval.items() if not any("dropout" in str(x) or "rngs" in str(x) for x in k)
    }
    validate_flax_state_dict(unflatten_dict(filtered_eval_shapes), flax_state_dict)
    return unflatten_dict(flax_state_dict)
