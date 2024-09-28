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

"""This script is used an example of how to restore params from a orbax train_state ckpt."""

import os
import functools
from absl import app
from typing import Sequence
from etils import epath

import jax
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from maxdiffusion import pyconfig, max_utils
from maxdiffusion.models import FlaxUNet2DConditionModel


def run(config):
  rng = jax.random.PRNGKey(config.seed)

  # Creates mesh using number of devices available
  # and ici/dcn parallelism rules
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Load the UNET from the checkpoint
  unet, params = FlaxUNet2DConditionModel.from_pretrained(
      config.pretrained_model_name_or_path,
      revision=config.revision,
      dtype=config.activations_dtype,
      subfolder="unet",
      split_head_dim=True,
  )

  weights_init_fn = functools.partial(unet.init_weights, rng=rng)
  # max_utils.get_abstract_state(unet, None, config, mesh, weights_init_fn, training=False)

  unboxed_abstract_state, _, _ = max_utils.get_abstract_state(unet, None, config, mesh, weights_init_fn, False)
  ckptr = ocp.PyTreeCheckpointer()

  # ckpt_path = config.checkpoint_dir
  ckpt_path = os.path.join(config.checkpoint_dir, "11", "unet_state")
  ckpt_path = epath.Path(ckpt_path)

  print(f"loading paramteres from : {ckpt_path}")

  restore_args = ocp.checkpoint_utils.construct_restore_args(unboxed_abstract_state.params)
  restored = ckptr.restore(
      ckpt_path, item={"params": unboxed_abstract_state.params}, transforms={}, restore_args={"params": restore_args}
  )
  return restored["params"]


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


# Run via:
# python src/maxdiffusion/pedagogical_examples/checkpoint_params_restore.py src/diffusers/configs/base_xl.yml
if __name__ == "__main__":
  app.run(main)
