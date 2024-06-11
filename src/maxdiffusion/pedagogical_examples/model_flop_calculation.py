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

from absl import app
from functools import reduce
from typing import (
  Any,
  Callable,
  Dict,
  Iterable,
  List,
  Mapping,
  Optional,
  Sequence,
  Set,
  Tuple,
  Union,
)

import jax
import numpy as np
from jax.sharding import Mesh
import flax.linen as nn
import flax.linen.module as module_lib
from flax.linen.summary import _process_inputs
from flax.typing import (
  PRNGKey,
  RNGSequences,
)
from maxdiffusion import (
  FlaxStableDiffusionPipeline
)
from maxdiffusion.models.attention_flax import AttentionOp
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_dtype,
  get_flash_block_sizes,
  calculate_training_tflops
)
from maxdiffusion.maxdiffusion_utils import get_dummy_inputs

# Taking inspiration from flax's https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/summary.html#tabulate
# to retrieve layer parameters and calculate 
def calculate_model_flops(
    module: module_lib.Module,
    rngs: Union[PRNGKey, RNGSequences],
    train=False,
    **kwargs,
):
  with module_lib._tabulate_context():
    _ = jax.eval_shape(module.init, rngs, **kwargs)
    calls = module_lib._context.call_info_stack[-1].calls
    calls.sort(key=lambda c: c.index)

  visited_paths: Set[Tuple[str, ...]] = set()
  total_flops = 0
  for c in calls:
    inputs = _process_inputs(c.args, c.kwargs)
    if c.path in visited_paths:
      continue
    else:
      # Multiply 2 * input shapes * features
      # For simple Dense layer input_shape = batch, in_features = (16, 10) and features = 20
      # Then 2 * 16 * 10 * 20.
      # In case of attention, an example if input shape batch,seq,hidden = (16, 4096, 320) and features = 320
      # Then 2 * 16 * 4096 * 320^2.
      if isinstance(c.module, nn.Dense):
        total_flops += 2 * (
          reduce(lambda x, y: x * y, inputs.shape)
          * c.module.features
        )
      # Here we capture qk einsum, scaling, softmax and attention_values * v
      # qk einsum : 2 * batch_size * seq_length^2 * heads * head_dim where (heads * head_dim) == hidden_dim
      # scaling : division of the attn scores matrix by sqrt of hidden dimension batch_size * seq_length^2)
      # softmax : rough estimate is batch_size * seq_length * hidden_dim * log(seq_length)
      # attention_values * v : 2 * batch_size * seq_length ^ 2 * hidden_dim
      elif isinstance(c.module, AttentionOp):
        qk_einsum = 2 * (reduce(lambda x, y: x * y, inputs[0].shape)) * inputs[0].shape[1]
        scaling = inputs[0].shape[0] * inputs[0].shape[1]**2
        softmax = reduce(lambda x, y: x * y, inputs[0].shape) * np.log(inputs[0].shape[1])
        # for readability. 
        att_v = qk_einsum
        total_flops = total_flops + qk_einsum + scaling + softmax + att_v
      elif isinstance(c.module, nn.Conv):
        total_flops += 2 * (
          reduce(lambda x, y: x * y, inputs.shape)
          * c.module.features
          * reduce(lambda x, y: x * y, c.module.kernel_size))
    visited_paths.add(c.path)
  
  total_flops = (total_flops * 3 if train else total_flops) / 10**12
  return total_flops

def run(config):
  rng = jax.random.PRNGKey(config.seed)

  # Creates mesh using number of devices available
  # and ici/dcn parallelism rules
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  weight_dtype = get_dtype(config)
  flash_block_sizes = get_flash_block_sizes(config)
  pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,revision=config.revision,
    dtype=weight_dtype,
    safety_checker=None,
    feature_extractor=None,
    from_pt=config.from_pt,
    split_head_dim=config.split_head_dim,
    norm_num_groups=config.norm_num_groups,
    attention_kernel=config.attention,
    flash_block_sizes=flash_block_sizes,
    mesh=mesh,
    )
  (latents, timesteps,
   encoder_hidden_states, added_cond_kwargs) = get_dummy_inputs(config, pipeline)
  total_flops = calculate_model_flops(pipeline.unet,
                        rng,
                        train=True,
                        sample=latents,
                        timesteps=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs)
  
  model_tflops = calculate_training_tflops(pipeline, params["unet"], config)

  print(total_flops)
  print(model_tflops)

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)

if __name__ == "__main__":
  app.run(main)