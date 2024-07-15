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

"""
Run ex:
python src/maxdiffusion/pedagogical_examples/model_flop_calculation.py src/maxdiffusion/configs/base15.yml
"""

from absl import app
from typing import (
  Sequence,
)

import jax
from jax.sharding import Mesh
from maxdiffusion import (
  FlaxStableDiffusionPipeline
)
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_flash_block_sizes,
)
from maxdiffusion.maxdiffusion_utils import calculate_unet_tflops

def run(config):
  rng = jax.random.PRNGKey(config.seed)

  # Creates mesh using number of devices available
  # and ici/dcn parallelism rules
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  flash_block_sizes = get_flash_block_sizes(config)
  pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,revision=config.revision,
    dtype=config.activations_dtype,
    safety_checker=None,
    feature_extractor=None,
    from_pt=config.from_pt,
    split_head_dim=config.split_head_dim,
    norm_num_groups=config.norm_num_groups,
    attention_kernel=config.attention,
    flash_block_sizes=flash_block_sizes,
    mesh=mesh,
    )

  total_flops = calculate_unet_tflops(config,
                        pipeline,
                        rng,
                        train=True)

  print("total training tflops: ", total_flops)

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)

if __name__ == "__main__":
  app.run(main)
