# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for JAX inference."""

from typing import Any, Callable
import functools

from flax import core
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
import flax
import jax.numpy as jnp
import jax
import numpy as np

from maxdiffusion import max_logging

FrozenDict = core.frozen_dict.FrozenDict


def loop_body(
    step,
    args,
    model,
    pipeline,
    added_cond_kwargs,
    prompt_embeds,
    guidance_scale,
):
  latents, scheduler_state, state = args
  latents_input = jnp.concatenate([latents] * 2)

  t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
  timestep = jnp.broadcast_to(t, latents_input.shape[0])

  latents_input = pipeline.scheduler.scale_model_input(
      scheduler_state, latents_input, t
  )
  noise_pred = model.apply(
      {"params": state.params},
      jnp.array(latents_input),
      jnp.array(timestep, dtype=jnp.int32),
      encoder_hidden_states=prompt_embeds,
      added_cond_kwargs=added_cond_kwargs,
  ).sample

  noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
  noise_pred = noise_pred_uncond + guidance_scale * (
      noise_prediction_text - noise_pred_uncond
  )

  latents, scheduler_state = pipeline.scheduler.step(
      scheduler_state, noise_pred, t, latents
  ).to_tuple()

  return latents, scheduler_state, state


def get_add_time_ids(
    original_size, crops_coords_top_left, target_size, bs, dtype
):
  add_time_ids = list(original_size + crops_coords_top_left + target_size)
  add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
  return add_time_ids


def get_embeddings(prompt_ids, pipeline, params):
  te_1_inputs = prompt_ids[:, 0, :]
  te_2_inputs = prompt_ids[:, 1, :]

  prompt_embeds = pipeline.text_encoder(
      te_1_inputs, params=params["text_encoder"], output_hidden_states=True
  )
  prompt_embeds = prompt_embeds["hidden_states"][-2]
  prompt_embeds_2_out = pipeline.text_encoder_2(
      te_2_inputs, params=params["text_encoder_2"], output_hidden_states=True
  )
  prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]
  text_embeds = prompt_embeds_2_out["text_embeds"]
  prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)
  return prompt_embeds, text_embeds


def tokenize(prompt, pipeline):
  inputs = []
  for _tokenizer in [pipeline.tokenizer, pipeline.tokenizer_2]:
    text_inputs = _tokenizer(
        prompt,
        padding="max_length",
        max_length=_tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    inputs.append(text_inputs.input_ids)
  inputs = jnp.stack(inputs, axis=1)
  return inputs


def tokenize_one(prompt, tokenizer):
  inputs = []
  # for _tokenizer in [pipeline.tokenizer, pipeline.tokenizer_2]:
  text_inputs = tokenizer(
      [prompt],
      padding="max_length",
      max_length=tokenizer.model_max_length,
      truncation=True,
      return_tensors="np",
  )
  return text_inputs.input_ids


def vae_decode(latents, state, pipeline):
  latents = 1 / pipeline.vae.config.scaling_factor * latents
  image = pipeline.vae.apply(
      {"params": state.params}, latents, method=pipeline.vae.decode
  ).sample
  image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
  return image


def create_device_mesh(config, devices=None, logging=True):
  """Creates a device mesh with each slice in its own data parallel group.

  If there is only one slice, uses two replicas
  """
  print("create_device_mesh started")
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1
  num_devices_per_slice = num_devices // num_slices
  max_logging.log(f"Devices: {devices} (num_devices: {num_devices})")
  assert len(devices) > 1, "You must have at least two devices"

  ici_parallelism = [
      config.ici_data_parallelism,
      config.ici_fsdp_parallelism,
      config.ici_tensor_parallelism,
  ]

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(
      ici_parallelism, num_devices_per_slice, "ICI"
  )
  mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)

  if logging:
    max_logging.log(f"Decided on mesh: {mesh}")

  return mesh


def get_dtype(config):
  dtype_str = config.dtype
  retval = jnp.bfloat16
  if dtype_str == "float32":
    retval = jnp.float32
  if dtype_str == "float16":
    retval = jnp.float16
  return retval


class InferenceState(struct.PyTreeNode):
  # pylint: disable=g-bare-generic
  apply_fn: Callable = struct.field(pytree_node=False)
  params: FrozenDict[str, Any] | None = struct.field(pytree_node=True)


def get_states(
    mesh, tx, rng, config, pipeline, unet_params, vae_params, training=True
):
  print("get_states started")
  unet_variables = jax.jit(pipeline.unet.init_weights)(rng)
  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(
      pipeline.unet, tx, config, mesh, unet_variables, training=training
  )
  del unet_variables
  unet_state, unet_state_mesh_shardings = setup_initial_state(
      pipeline.unet,
      tx,
      config,
      mesh,
      unet_params,
      unboxed_abstract_state,
      state_mesh_annotations,
      training=training,
  )

  vae_variables = jax.jit(pipeline.vae.init_weights)(rng)
  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(
      pipeline.vae, tx, config, mesh, vae_variables, training=training
  )
  del vae_variables
  vae_state, vae_state_mesh_shardings = setup_initial_state(
      pipeline.vae,
      tx,
      config,
      mesh,
      vae_params,
      unboxed_abstract_state,
      state_mesh_annotations,
      training=training,
  )

  return (
      unet_state,
      unet_state_mesh_shardings,
      vae_state,
      vae_state_mesh_shardings,
  )


def setup_initial_state(
    model,
    tx,
    config,
    mesh,
    model_params,
    unboxed_abstract_state,
    state_mesh_annotations,
    checkpoint_manager=None,
    training=True,
):
  """Initialize the model and optimizer state, and optionally load from a checkpoint as necessary.

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
  print("setup_initial_state done")

  # Initialization
  state = None
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations
    )
  if not state:
    init_train_state_partial = functools.partial(
        init_train_state, model=model, tx=tx, training=training
    )

    sharding = PositionalSharding(mesh.devices).replicate()
    partial_device_put_replicated = functools.partial(
        device_put_replicated, sharding=sharding
    )
    model_params = jax.tree_util.tree_map(
        partial_device_put_replicated, model_params
    )

    with jax.transfer_guard("disallow"):
      state = jax.jit(
          init_train_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings,
      )(model_params=model_params)

  state = unbox_logicallypartioned_trainstate(state)

  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations
  )
  return state, state_mesh_shardings


def init_train_state(model_params, model, tx, training=True):
  """We pass in "static" objects like model, tx, config, as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model_params, model, tx, training
  """
  print("init_train_state done")
  if training:
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=model_params, tx=tx
    )
  else:
    state = InferenceState(apply_fn=model.apply, params=model_params)
  return state


def unbox_logicallypartioned_trainstate(
    boxed_train_state: train_state.TrainState,
):
  """Unboxes the flax.LogicallyPartitioned pieces in a train state.

  Args:
    boxed_train_state: a train state that includes LogicallyPartitioned leaves.

  Returns:
    a TrainState where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox()
      if isinstance(x, flax.linen.spmd.LogicallyPartitioned)
      else x,
      boxed_train_state,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def device_put_replicated(x, sharding):
  return jax.make_array_from_callback(x.shape, sharding, lambda index: x[index])


def get_abstract_state(model, tx, config, mesh, model_params, training=True):
  """Get a shaped abstraction of the state (including optimizer)"""
  abstract_state = jax.eval_shape(
      functools.partial(
          init_train_state, model=model, tx=tx, training=training
      ),
      model_params=model_params,
  )
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  unboxed_abstract_state = unbox_logicallypartioned_trainstate(abstract_state)
  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return unboxed_abstract_state, state_mesh_annotations


def fill_unspecified_mesh_axes(
    parallelism_vals, target_product, parallelism_type
):
  """Evaluates unspecified DCN/ICI parallelism values."""
  print("fill_unspecified_mesh_axes done")
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, (
        f"Found unspecified values (-1) for more than one {parallelism_type}   "
        "   parallelism axis. At most one axis can be unspecified."
    )

    determined_val = target_product / np.product(parallelism_vals) * -1

    assert determined_val >= 1 and determined_val.is_integer, (
        "Unspecified value unable to be determined with the given"
        f" {parallelism_type} parallelism values"
    )

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"

  assert np.product(parallelism_vals) == target_product, (
      f"Number of {target_type} {target_product} does not match the product"
      f" of the {parallelism_type} parallelism {np.product(parallelism_vals)}"
  )

  return parallelism_vals

