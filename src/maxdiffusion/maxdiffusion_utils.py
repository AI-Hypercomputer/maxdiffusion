"""
 Copyright 2024 Google LLC

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

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax

def load_sdxllightning_unet(config, pipeline, params):
  """Load lightning """
  if not config.lightning_from_pt:
    raise ValueError("Only loading lightning models from Pytorch is currently supported.")
  unet_lightning_state_dict = load_file(hf_hub_download(config.lightning_repo, config.lightning_ckpt), device="cpu")
  flax_unet_dict = convert_pytorch_state_dict_to_flax(unet_lightning_state_dict, pipeline.unet)
  params["unet"] = flax_unet_dict
  return pipeline, params
