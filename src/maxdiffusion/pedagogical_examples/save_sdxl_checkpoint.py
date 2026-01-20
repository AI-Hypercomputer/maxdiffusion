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

"""Load and save a checkpoint. This is useful for uploading checkpoints to gcs
and later loading them from gcs directly.
After calling this script, use gsutil to upload the weights to a bucket:
gsutil -m cp -r sdxl-model-finetuned gs://<your-bucket>/sdxl_1.0_base/
"""

from typing import Sequence
from absl import app
import jax
from jax.sharding import Mesh

from maxdiffusion import (FlaxStableDiffusionXLPipeline, max_logging, pyconfig)

from maxdiffusion.max_utils import (create_device_mesh, get_flash_block_sizes)


def run(config):
  # Setup Mesh
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  flash_block_sizes = get_flash_block_sizes(config)

  pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
      config.pretrained_model_name_or_path,
      revision=config.revision,
      dtype=config.activations_dtype,
      split_head_dim=config.split_head_dim,
      norm_num_groups=config.norm_num_groups,
      attention_kernel=config.attention,
      flash_block_sizes=flash_block_sizes,
      mesh=mesh,
  )

  pipeline.save_pretrained(config.output_dir, params)


def main(argv: Sequence[str]) -> None:
  max_logging.log(f"Found {jax.device_count()} devices.")
  pyconfig.initialize(argv)
  config = pyconfig.config
  run(config)


if __name__ == "__main__":
  app.run(main)
