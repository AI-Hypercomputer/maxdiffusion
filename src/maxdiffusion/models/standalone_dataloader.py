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

import datetime
import logging
import os
from typing import Sequence
import numpy as np
import optax
import jax
from jax.sharding import Mesh
from absl import app

from maxdiffusion import (
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    max_logging,
    max_utils,
    pyconfig,
    mllog_utils,
    generate,
    eval,
)

from maxdiffusion.input_pipeline.input_pipeline_interface import (
  make_pokemon_train_iterator,
  make_laion400m_train_iterator,
  get_shaped_batch
)
from maxdiffusion.models.train import validate_train_config, get_first_step, load_next_batch

def data_load_setup(config):
  rng = jax.random.PRNGKey(config.seed)
  # Setup Mesh
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  total_train_batch_size = config.per_device_batch_size * jax.device_count()
  if config.checkpoint_every % total_train_batch_size != 0:
    max_logging.log(f"Checkpoint at {config.checkpoint_every} samples is not evenly divisible by"
                    f" global batch size of {total_train_batch_size}. Checkpointing might not"
                    " work correctly.")
  weight_dtype = max_utils.get_dtype(config)
  flash_block_sizes = max_utils.get_flash_block_sizes(config)
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
  params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)  
  learning_rate_scheduler = max_utils.create_learning_rate_schedule(config)
  
  adamw = optax.adamw(
      learning_rate=learning_rate_scheduler,
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      weight_decay=config.adam_weight_decay,
  )

  (unet_state,
  unet_state_mesh_shardings,
  vae_state, vae_state_mesh_shardings) = max_utils.get_states(mesh,
                                                              adamw, rng, config,
                                                              pipeline, params["unet"],
                                                              params["vae"], training=True)


  if config.dataset_name == "lambdalabs/pokemon-blip-captions":
    data_iterator = make_pokemon_train_iterator(
      config,
      mesh,
      total_train_batch_size,
      pipeline,
      params,
      rng
    )
  else:
    data_iterator = make_laion400m_train_iterator(
      config, mesh, total_train_batch_size
    )
  return data_iterator, unet_state

def data_load_loop(config):
  data_iterator, state = data_load_setup(config)
  example_batch = None
  start = datetime.datetime.now()
  start_step = get_first_step(state)
  example_batch = load_next_batch(data_iterator, example_batch, config)
  jax.block_until_ready(example_batch)
  first_end = datetime.datetime.now()
  time_to_load_first_batch = first_end-start
  if jax.process_index() == 0:
    max_logging.log(f"STANDALONE DATALOADER : First step completed in {time_to_load_first_batch.seconds} seconds, on host 0") 

  for step in np.arange(start_step+1, config.steps):
    example_batch = load_next_batch(data_iterator, example_batch, config)
    # new_time = datetime.datetime.now()
    # if jax.process_index() == 0:
    #   max_logging.log(f"STANDALONE DATALOADER : load batch {step+1} in {(new_time-last_step_completion).seconds} seconds")
    # last_step_completion = new_time

  jax.block_until_ready(example_batch) # wait until the last batch is read
  end = datetime.datetime.now()
  if jax.process_index() == 0:
    max_logging.log(f"STANDALONE DATALOADER : {config.steps} batches loaded in {(end-start).seconds} seconds, on host 0")
  return state     

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  config = pyconfig.config
  mllog_utils.train_init_start(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  # cc.initialize_cache(os.path.expanduser("~/jax_cache"))
  validate_train_config(config)
  data_load_loop(config)
  #train(config)
if __name__ == "__main__":
  app.run(main)
