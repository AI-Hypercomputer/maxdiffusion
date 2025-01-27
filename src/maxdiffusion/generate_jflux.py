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

import time
from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp
from absl import app
from maxdiffusion import (pyconfig, max_logging)
from PIL import Image
from flax.linen import partitioning as nn_partitioning

import maxdiffusion.checkpointing.jflux_checkpointer as jflux_checkpointer

from einops import rearrange
import os
import re
from glob import iglob


def run(config):
  device_type = jflux_checkpointer.get_device_type()
  max_logging.log(f"Using {device_type} device")

  output_dir = "output"
  seed = jax.random.PRNGKey(seed=102333 if config.seed is None else config.seed)
  max_logging.log(f"Generating with seed {config.seed}:\n{config.prompt}")

  checkpointer = jflux_checkpointer.JfluxCheckpointer(config)
  pipeline = checkpointer.load_checkpoint()
  state, _, _ = checkpointer.create_flux_state(pipeline.flux, pipeline.init_flux_weights, None, True)
  state = state.params

  with checkpointer.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state = jax.device_put(state, pipeline.data_sharding)

    img = pipeline.create_noise(len(jax.devices()), config.resolution, config.resolution, config.activations_dtype, seed)
    (txt, txt_ids, vec, img) = pipeline.prepare_inputs([config.prompt for _ in range(len(jax.devices()))], img)

    def do_inference():
      return pipeline(
          state,
          txt,
          txt_ids,
          vec,
          config.num_inference_steps,
          config.resolution,
          config.resolution,
          config.guidance_scale,
          img,
          shift=config.model_name != "flux-schnell",
      )

    t0 = time.perf_counter()
    x = do_inference()
    t1 = time.perf_counter()
    print(f"Compile time: {t1 - t0:.1f}s.")
    # real run
    max_logging.log("real inference")
    t0 = time.perf_counter()
    x = do_inference()
    t1 = time.perf_counter()

    output_name = os.path.join(output_dir, "maxdiff_img_{idx}.jpg")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      idx = 0
    else:
      fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"maxdiff_img_[0-9]+\.jpg$", fn)]
      if len(fns) > 0:
        idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
      else:
        idx = 0
    fn = output_name.format(idx=idx)
    max_logging.log(f"Done in {t1 - t0:.1f}s. Saving {fn}")
    # bring into PIL format and save
    x = x.clip(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    x = 127.5 * (x + 1.0)
    x_numpy = np.array(x.astype(jnp.uint8))
    img = Image.fromarray(x_numpy)

    img.save(fn, quality=95, subsampling=0)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
