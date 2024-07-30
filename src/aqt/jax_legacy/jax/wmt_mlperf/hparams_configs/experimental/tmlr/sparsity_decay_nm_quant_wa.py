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

gxm third_party/py/aqt/jax_legacy/jax/wmt_mlperf/google/xm_launch.py
--xm_resource_alloc=group:peace/babelfish-device
--hparams_config_dir=third_party/py/aqt/jax_legacy/jax/wmt_mlperf/hparams_configs/experimental/tmlr
--hparams_config_filename=sparsity_decay_nm_quant_wa.py --cell=yo
--sparsity_start_step=10000 --sparsity_update_freq=1000
--name="decay_sparsity_quant_sweep_2"
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
    for prec in [8, 4, 2]:
      for prune_rate in [(2, 4), (1, 4)]:
        config = copy.deepcopy(model_config)

        config.quant_type = 'aqt'

        # mlp + attention weights + acts  quantization
        config.dense.weight_prec = prec
        config.dense.quant_act.prec = prec

        config.num_train_steps = 100000

        # mlp weights sparsity
        config.mlp_block.dense_1.weight_sparsity.type = 'STRUCTURED_NM'
        config.mlp_block.dense_1.weight_sparsity.prune_rate = prune_rate
        config.mlp_block.dense_1.weight_sparsity.structure_decay = True

        config.mlp_block.dense_2.weight_sparsity.type = 'STRUCTURED_NM'
        config.mlp_block.dense_2.weight_sparsity.prune_rate = prune_rate
        config.mlp_block.dense_2.weight_sparsity.structure_decay = True

        # attn_weights sparsity
        config.attention.dense_kqv.weight_sparsity.type = 'STRUCTURED_NM'
        config.attention.dense_kqv.weight_sparsity.prune_rate = prune_rate
        config.attention.dense_kqv.weight_sparsity.structure_decay = True

        config.attention.dense_out.weight_sparsity.type = 'STRUCTURED_NM'
        config.attention.dense_out.weight_sparsity.prune_rate = prune_rate
        config.attention.dense_out.weight_sparsity.structure_decay = True

        config.metadata.hyper_str = f'{config.metadata.hyper_str}_decay_NM({prune_rate})_prec({prec}_100000steps)'
        configs.append(config)

  sweep_config.configs = configs
  return sweep_config
