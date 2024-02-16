# ruff: noqa
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

# pylint: disable=bare-except, consider-using-generator
""" Common Max Utils needed by multiple modules"""
import functools
import json
import os
import subprocess

import numpy as np

import flax
import jax
import jax.numpy as jnp
import optax
from maxdiffusion import checkpointing, max_logging
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from flax import struct
from typing import Callable, Any
from flax import core

FrozenDict = core.frozen_dict.FrozenDict

class InferenceState(struct.PyTreeNode):
  # pylint: disable=g-bare-generic
  apply_fn: Callable = struct.field(pytree_node=False)
  params: FrozenDict[str, Any] | None = struct.field(pytree_node=True)

def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jax.tree_util.tree_reduce(
      lambda x, y: x + jax.numpy.sum(y ** 2), x, initializer=0.0
  ) ** 0.5

def activate_profiler(config):
  if jax.process_index() == 0 and config.enable_profiler:
    jax.profiler.start_trace(config.tensorboard_dir)

def deactivate_profiler(config):
  if jax.process_index() == 0 and config.enable_profiler:
    jax.profiler.stop_trace()

def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {}
  for val in metrics['scalar']:
    metrics_dict[val] = float(metrics['scalar'][val])
  metrics_dict['step'] = float(step)
  metrics_dict['run_name'] = run_name
  return metrics_dict

def write_metrics_locally(metrics, step, config, file):
  """Writes metrics locally for testing"""
  if step == 0:
    file.truncate(0)

  metrics_dict = _prepare_metrics_for_json(metrics, step, config.run_name)
  file.write(str(json.dumps(metrics_dict))+'\n')

  if step == config.max_train_steps - 1:
    file.close()

def write_metrics_for_gcs(metrics, step, config, running_metrics):
  """Writes metrics to gcs"""
  metrics_dict_step = _prepare_metrics_for_json(metrics, step, config.run_name)
  running_metrics.append(metrics_dict_step)
  if (step + 1) % config.log_period == 0 or step == config.max_train_steps - 1:
    start_step = (step // config.log_period) * config.log_period
    metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
    with open(metrics_filename, 'w', encoding="utf8") as metrics_for_gcs:
      for metrics_step in running_metrics:
        metrics_for_gcs.write(str(json.dumps(metrics_step))+'\n')

    metrics_for_gcs.close()
    gcs_filename=os.path.join(config.metrics_dir, metrics_filename)
    command = ["gsutil", "mv", metrics_filename, gcs_filename]
    max_logging.log(f"Moving file {metrics_filename} to GCS...")
    subprocess.run(command, check=True, capture_output=True)
    max_logging.log(f"File {metrics_filename} moved successfully!")
    running_metrics = [] # reset running_metrics to empty list
  return running_metrics

def initialize_jax_distributed_system():
  """ The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
      indirection in MaxText to avoid breaking the call sites unnecessarily.

      Currently jax.distributed.initialize() fully works as expected!
  """
  max_logging.log("Attempting to initialize the jax distributed system...")
  jax.distributed.initialize()
  max_logging.log("Jax distributed system initialized!")

def device_put_replicated(x, sharding):
  return jax.make_array_from_callback(x.shape, sharding, lambda index: x[index])

def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product/np.product(parallelism_vals)*-1

    assert determined_val >= 1 and determined_val.is_integer, f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == 'DCN' else "devices per slice"

  assert np.product(parallelism_vals) == target_product, f"Number of {target_type} {target_product} does not match\
    the product of the {parallelism_type} parallelism {np.product(parallelism_vals)}"

  return parallelism_vals

def create_device_mesh(config, devices=None, logging=True):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas """
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1
  num_devices_per_slice = num_devices//num_slices
  max_logging.log(f"Devices: {devices} (num_devices: {num_devices})")
  assert len(devices) > 1, "You must have at least two devices"


  ici_parallelism = [config.ici_data_parallelism, config.ici_fsdp_parallelism, config.ici_tensor_parallelism]

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, 'ICI')
  mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)

  if logging:
    max_logging.log(f"Decided on mesh: {mesh}")

  return mesh

def unbox_logicallypartioned_trainstate(
    boxed_train_state: train_state.TrainState):
  """ Unboxes the flax.LogicallyPartitioned pieces in a train state.

    Args:
      boxed_train_state: a train state that includes LogicallyPartitioned
        leaves.
    Returns:
      a TrainState where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(lambda x: x.unbox() if \
        isinstance(x, flax.linen.spmd.LogicallyPartitioned) \
        else x, boxed_train_state, \
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned))

def init_train_state(model_params, model, tx, training=True):
  """
  We pass in "static" objects like model, tx, config, as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model_params, model, tx, training
  """
  if training:
    state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=model_params,
      tx=tx)
  else:
    state = InferenceState(apply_fn=model.apply, params=model_params)
  return state

def get_abstract_state(model, tx, config, mesh, model_params, training=True):
  """ Get a shaped abstraction of the state (including optimizer)"""
  abstract_state = jax.eval_shape(functools.partial(init_train_state, model=model, tx=tx, training=training), model_params=model_params)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  unboxed_abstract_state = unbox_logicallypartioned_trainstate(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return unboxed_abstract_state, state_mesh_annotations

def setup_initial_state(model, tx, config, mesh, model_params, unboxed_abstract_state, state_mesh_annotations, checkpoint_manager=None, training=True):
  """ We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    mesh: jax.devices() mesh
    model_params: model parameters
    unboxed_abstract_state: abstract state from get_abstract_state()
    state_mesh_annotations: state mesh annotations from get_abstract_state()
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    training: boolean True when initial state is used in training

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  # Initialization
  state = None
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    if checkpoint_manager:
      state, raw_params = checkpointing.load_state_if_possible(checkpoint_manager,
                                                  config.load_parameters_path,
                                                  config.load_from_other_directory,
                                                  config.load_from_other_directory_step,
                                                  unboxed_abstract_state,
                                                  mesh,
                                                  state_mesh_annotations)

    state_mesh_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  if not state:
    init_train_state_partial = functools.partial(init_train_state, model=model, tx=tx, training=training)

    sharding = PositionalSharding(mesh.devices).replicate()
    partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
    model_params = jax.tree_util.tree_map(partial_device_put_replicated, model_params)

    with jax.transfer_guard("disallow"):
      state = jax.jit(
          init_train_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings
      )(model_params=model_params)

  state = unbox_logicallypartioned_trainstate(state)

  state_mesh_shardings = jax.tree_map(
    lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  return state, state_mesh_shardings

def get_states(mesh, tx, rng, config, pipeline, unet_params, vae_params, training=True):
  unet_variables = jax.jit(pipeline.unet.init_weights)(rng)
  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(pipeline.unet, tx, config, mesh, unet_variables, training=training)
  del unet_variables
  unet_state, unet_state_mesh_shardings = setup_initial_state(
  pipeline.unet,
  tx,
  config,
  mesh,
  unet_params,
  unboxed_abstract_state,
  state_mesh_annotations,
  training=training)

  vae_variables = jax.jit(pipeline.vae.init_weights)(rng)
  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(pipeline.vae, tx, config, mesh, vae_variables, training=training)
  del vae_variables
  vae_state, vae_state_mesh_shardings = setup_initial_state(
      pipeline.vae,
      tx,
      config,
      mesh,
      vae_params,
      unboxed_abstract_state,
      state_mesh_annotations,
      training=training
  )

  return unet_state, unet_state_mesh_shardings, vae_state, vae_state_mesh_shardings

# Learning Rate Schedule
# -----------------------------------------------------------------------------

def create_learning_rate_schedule(config):
  """Creates a warmup and cosine decay learning rate schedule:
  We take inspiration from Llama2's learning rate (LR) schedule, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  Learning rate schedule has either two or three parts:
  1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
  2) Cosine from [learning_rate] to [learning_rate * cosine_learning_rate_final_fraction] until learning_rate_schedule_steps
  3) Constant learning rate of 0 from learning_rate_schedule_steps to steps.
  The zero learning rate section can be used to more accurately measure the fully trained model's performance.
  """
  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = (step) / len_steps
      a = 0.5 * (jnp.cos(jnp.pi*pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr
    return schedule

  lr = config.learning_rate
  cos_final_lr = lr * config.cosine_learning_rate_final_fraction

  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  cos_steps = config.learning_rate_schedule_steps - warmup_steps
  constant_zero_steps = config.max_train_steps - config.learning_rate_schedule_steps

  warmup_schedule = optax.linear_schedule(
      init_value=0.0,
      end_value=lr,
      transition_steps=warmup_steps
  )
  cos_schedule = make_cos_schedule(lr, cos_final_lr, cos_steps)
  constant_schedule = optax.constant_schedule(0.0)

  pieces = [warmup_schedule, cos_schedule]
  boundaries=[
   warmup_steps,
   warmup_steps + cos_steps,
   ]

  if constant_zero_steps > 0:
    pieces.append(constant_schedule)
    boundaries.append(warmup_steps + cos_steps + constant_zero_steps)

  return optax.join_schedules(pieces, boundaries)

def get_dtype(config):
  dtype_str = config.dtype
  retval = jnp.bfloat16
  if dtype_str == "float32":
    retval = jnp.float32
  if dtype_str == "float16":
    retval = jnp.float16
  return retval
