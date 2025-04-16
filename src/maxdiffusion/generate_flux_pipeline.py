"""
 Copyright 2025 Google LLC

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

from typing import Sequence
from absl import app
from contextlib import ExitStack
import functools
import time
import numpy as np
from PIL import Image
import jax

from maxdiffusion import pyconfig, max_logging, max_utils

from maxdiffusion.checkpointing.checkpointing_utils import load_params_from_path
from maxdiffusion.max_utils import setup_initial_state


def run(config):
  from maxdiffusion.checkpointing.flux_checkpointer import FluxCheckpointer

  checkpoint_loader = FluxCheckpointer(config, "FLUX_CHECKPOINT")
  pipeline, params = checkpoint_loader.load_checkpoint()

  if not params:
    ## VAE
    weights_init_fn = functools.partial(pipeline.vae.init_weights, rng=checkpoint_loader.rng)
    unboxed_abstract_state, _, _ = max_utils.get_abstract_state(
        pipeline.vae, None, config, checkpoint_loader.mesh, weights_init_fn, False
    )
    # load unet params from orbax checkpoint
    vae_params = load_params_from_path(
        config, checkpoint_loader.checkpoint_manager, unboxed_abstract_state.params, "vae_state"
    )

    vae_state = {"params": vae_params}

    ## Flux
    weights_init_fn = functools.partial(
        pipeline.flux.init_weights, rngs=checkpoint_loader.rng, max_sequence_length=config.max_sequence_length
    )

    unboxed_abstract_state, _, _ = max_utils.get_abstract_state(
        pipeline.flux, None, config, checkpoint_loader.mesh, weights_init_fn, False
    )
    # load unet params from orbax checkpoint
    flux_params = load_params_from_path(
        config, checkpoint_loader.checkpoint_manager, unboxed_abstract_state.params, "flux_state"
    )
    flux_state = {"params": flux_params}
  else:
    weights_init_fn = functools.partial(
        pipeline.flux.init_weights,
        rngs=checkpoint_loader.rng,
        max_sequence_length=config.max_sequence_length,
        eval_only=False,
    )
    transformer_state, flux_state_shardings = setup_initial_state(
        model=pipeline.flux,
        tx=None,
        config=config,
        mesh=checkpoint_loader.mesh,
        weights_init_fn=weights_init_fn,
        model_params=None,
        training=False,
    )
    transformer_state = transformer_state.replace(params=params["flux_transformer_params"])
    transformer_state = jax.device_put(transformer_state, flux_state_shardings)

    weights_init_fn = functools.partial(pipeline.vae.init_weights, rng=checkpoint_loader.rng)
    vae_state, _ = setup_initial_state(
        model=pipeline.vae,
        tx=None,
        config=config,
        mesh=checkpoint_loader.mesh,
        weights_init_fn=weights_init_fn,
        model_params=params["flux_vae"],
        training=False,
    )

    vae_state = {"params": vae_state.params}
    flux_state = {"params": transformer_state.params}

  t0 = time.perf_counter()
  with ExitStack():
    imgs = pipeline(flux_params=flux_state, timesteps=50, vae_params=vae_state).block_until_ready()
  t1 = time.perf_counter()
  max_logging.log(f"Compile time: {t1 - t0:.1f}s.")

  t0 = time.perf_counter()
  with ExitStack():
    imgs = pipeline(flux_params=flux_state, timesteps=50, vae_params=vae_state).block_until_ready()
  imgs = jax.experimental.multihost_utils.process_allgather(imgs, tiled=True)
  t1 = time.perf_counter()
  max_logging.log(f"Inference time: {t1 - t0:.1f}s.")
  imgs = np.array(imgs)
  imgs = (imgs * 0.5 + 0.5).clip(0, 1)
  imgs = np.transpose(imgs, (0, 2, 3, 1))
  imgs = np.uint8(imgs * 255)
  for i, image in enumerate(imgs):
    Image.fromarray(image).save(f"flux_{i}.png")

  return imgs


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
