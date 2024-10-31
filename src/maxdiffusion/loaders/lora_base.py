# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..models.modeling_utils import load_state_dict
from ..utils import _get_model_file

import safetensors


class LoRABaseMixin:
  """Utility class for handing LoRAs"""

  _lora_lodable_modules = []
  num_fused_loras = 0

  def load_lora_weights(self, **kwargs):
    raise NotImplementedError("`load_lora_weights()` is not implemented.")

  @classmethod
  def _fetch_state_dict(
      cls,
      pretrained_model_name_or_path_or_dict,
      weight_name,
      use_safetensors,
      local_files_only,
      cache_dir,
      force_download,
      resume_download,
      proxies,
      use_auth_token,
      revision,
      subfolder,
      user_agent,
      allow_pickle,
  ):
    from .lora_pipeline import LORA_WEIGHT_NAME_SAFE

    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
      # Let's first try to load .safetensors weights
      if (use_safetensors and weight_name is None) or (weight_name is not None and weight_name.endswith(".safetensors")):
        try:
          # Here we're relaxing the loading check to enable more Inference API
          # friendliness where sometimes, it's not at all possible to automatically
          # determine `weight_name`.
          if weight_name is None:
            weight_name = cls._best_guess_weight_name(
                pretrained_model_name_or_path_or_dict,
                file_extension=".safetensors",
                local_files_only=local_files_only,
            )
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = safetensors.torch.load_file(model_file, device="cpu")
        except (IOError, safetensors.SafetensorError) as e:
          if not allow_pickle:
            raise e
          # try loading non-safetensors weights
          model_file = None
          pass

        if model_file is None:
          if weight_name is None:
            weight_name = cls._best_guess_weight_name(
                pretrained_model_name_or_path_or_dict, file_extension=".bin", local_files_only=local_files_only
            )
          model_file = _get_model_file(
              pretrained_model_name_or_path_or_dict,
              weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
              cache_dir=cache_dir,
              force_download=force_download,
              resume_download=resume_download,
              proxies=proxies,
              local_files_only=local_files_only,
              use_auth_token=use_auth_token,
              revision=revision,
              subfolder=subfolder,
              user_agent=user_agent,
          )
          state_dict = load_state_dict(model_file)
      else:
        state_dict = pretrained_model_name_or_path_or_dict

    return state_dict
