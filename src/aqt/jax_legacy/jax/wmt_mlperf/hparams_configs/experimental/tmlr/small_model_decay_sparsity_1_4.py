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

"""use: --sparsity_start_step=1000.

--sparsity_update_freq=1000

Small-sized unquantized model.

A 'small size' halves each model size parameter of the full model:
* 3 layers in the encoder and decoder
* 512 emd_dim
* 8 heads
* 512 qkv_dim
* 2048 mlp_dim
"""

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs import base_config
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.tmlr import small_model_bfloat16


def get_config(quant_target=base_config.QuantTarget.NONE):
  """Returns configuration for a small transformer model."""
  config = small_model_bfloat16.get_config(quant_target=quant_target)
  config.metadata.hyper_str = 'small_structured_weights_decay_1_4'
  config.mlp_block.dense_1.weight_sparsity.prune_rate = (1, 4)
  config.mlp_block.dense_1.weight_sparsity.type = 'STRUCTURED_NM'
  config.mlp_block.dense_1.weight_sparsity.structure_decay = True

  config.mlp_block.dense_2.weight_sparsity.prune_rate = (1, 4)
  config.mlp_block.dense_2.weight_sparsity.type = 'STRUCTURED_NM'
  config.mlp_block.dense_2.weight_sparsity.structure_decay = True

  # attn_weights sparsity
  config.attention.dense_kqv.weight_sparsity.type = 'STRUCTURED_NM'
  config.attention.dense_kqv.weight_sparsity.prune_rate = (1, 4)
  config.attention.dense_kqv.weight_sparsity.structure_decay = True

  config.attention.dense_out.weight_sparsity.type = 'STRUCTURED_NM'
  config.attention.dense_out.weight_sparsity.prune_rate = (1, 4)
  config.attention.dense_out.weight_sparsity.structure_decay = True

  return config
