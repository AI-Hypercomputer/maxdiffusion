# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multiple sized model with varied quantization and sparsity.

Quantization is enabled since starting
Sparsity is one shot pruning, updated and applied at 100K steps.
"""

import copy

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs import base_config
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.tmlr import full_model_bfloat16
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.tmlr import small_model_bfloat16
import ml_collections


def get_config():
  """Sweep for multiple sized model with varied quantization and sparsity."""
  sweep_config = ml_collections.ConfigDict()
  base_configs = [
      full_model_bfloat16.get_config(
          quant_target=base_config.QuantTarget.WEIGHTS_AND_AUTO_ACTS
      ),
      small_model_bfloat16.get_config(
          quant_target=base_config.QuantTarget.WEIGHTS_AND_AUTO_ACTS
      ),
  ]

  configs = []
  for model_config in base_configs:
    for prec in [8, 4]:
      config = copy.deepcopy(model_config)

      config.quant_type = 'aqt'

      # mlp + attention weights + acts  quantization
      config.dense.weight_prec = prec
      config.dense.quant_act.prec = prec

      config.num_train_steps = 100000

      config.metadata.hyper_str = (
          f'{config.metadata.hyper_str}_prec({prec}_10000steps)'
      )
      configs.append(config)

  sweep_config.configs = configs
  return sweep_config
