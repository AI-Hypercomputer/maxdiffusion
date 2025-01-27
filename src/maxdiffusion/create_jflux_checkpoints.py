from typing import Sequence

from absl import app
from maxdiffusion import pyconfig

import maxdiffusion.checkpointing.jflux_checkpointer as jflux_checkpointer


def run(config):
  checkpointer = jflux_checkpointer.JfluxCheckpointer(config)
  flux, flux_state = checkpointer.load_pretrained_model(config.pretrained_model_name_or_path)
  # INTERNAL: Failed to load HSACO: HIP_ERROR_NoBinaryForGpu when jitting
  state, _, _ = checkpointer.create_flux_state(flux, None, {checkpointer.flux_state_item_name: flux_state}, True, False)
  step = config.checkpoint_step
  if step is None or step < 0:
    step = 0
  checkpointer.save_checkpoint(step, None, {checkpointer.flux_state_item_name: state})
  checkpointer.checkpoint_manager.wait_until_finished()


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
