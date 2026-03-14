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

import functools

from flax import nnx
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
import jaxopt
from maxdiffusion.checkpointing.wan_vace_checkpointer_2_1 import WanVaceCheckpointer2_1
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.trainers.base_wan_trainer import (
    BaseWanTrainer,
    _to_array,
)
import tensorflow as tf


class WanVaceTrainer(BaseWanTrainer):

  def _get_checkpointer(self):
    return WanVaceCheckpointer2_1(config=self.config)

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(
        mesh, P(*self.config.data_sharding)
    )
    data_sharding = {
        "latents": data_sharding,
        "encoder_hidden_states": data_sharding,
        "conditioning_latents": data_sharding,
    }
    return data_sharding

  def get_eval_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {
        "latents": data_sharding,
        "encoder_hidden_states": data_sharding,
        "timesteps": data_sharding,
        "conditioning_latents": data_sharding,
    }
    return data_sharding

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config

    # If using synthetic data
    if config.dataset_type == "synthetic":
      return make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          config.global_batch_size_to_load,
          pipeline=pipeline,  # Pass pipeline to extract dimensions
          is_training=is_training,
      )

    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Wan 2.1 training only supports config.dataset_type set to tfrecords and config.cache_latents_text_encoder_outputs set to True"
      )
    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        "conditioning_latents": tf.io.FixedLenFeature([], tf.string),
    }

    if not is_training:
      feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)

    def prepare_sample_train(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(
          features["encoder_hidden_states"], out_type=tf.float32
      )
      conditioning_latents = tf.io.parse_tensor(
          features["conditioning_latents"], out_type=tf.float32
      )
      return {
          "latents": latents,
          "encoder_hidden_states": encoder_hidden_states,
          "conditioning_latents": conditioning_latents,
      }

    def prepare_sample_eval(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(
          features["encoder_hidden_states"], out_type=tf.float32
      )
      conditioning_latents = tf.io.parse_tensor(
          features["conditioning_latents"], out_type=tf.float32
      )
      timesteps = features["timesteps"]
      return {
          "latents": latents,
          "encoder_hidden_states": encoder_hidden_states,
          "conditioning_latents": conditioning_latents,
          "timesteps": timesteps,
      }

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample_train if is_training else prepare_sample_eval,
        is_training=is_training,
    )
    return data_iterator

  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    return jax.jit(
        functools.partial(
            train_step, scheduler=pipeline.scheduler, config=self.config
        ),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    return jax.jit(
        functools.partial(
            eval_step, scheduler=pipeline.scheduler, config=self.config
        ),
        in_shardings=(state_shardings, eval_data_shardings, None, None),
        out_shardings=(None, None),
    )


def train_step(state, data, rng, scheduler_state, scheduler, config):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on, :]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(
        config.weights_dtype
    )
    control_hidden_states = data["conditioning_latents"].astype(
        config.weights_dtype
    )

    bsz = latents.shape[0]
    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    noise = jax.random.normal(
        key=new_rng, shape=latents.shape, dtype=latents.dtype
    )
    noisy_latents, training_target, training_weight = (
        scheduler.apply_flow_match(noise, latents, timesteps)
    )
    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=noisy_latents,
          timestep=timesteps,
          encoder_hidden_states=encoder_hidden_states,
          control_hidden_states=control_hidden_states,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
      )

    with jax.named_scope("loss"):
      model_pred = model_pred.astype(jnp.float32)
      training_target = training_target.astype(jnp.float32)
      loss = (training_target - model_pred) ** 2
      if not config.disable_training_weights:
        training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
        loss = loss * training_weight
      loss = jnp.mean(loss)

    return loss

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)

  max_abs_grad = jax.tree_util.tree_reduce(
      lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))),
      grads,
      initializer=-1.0,
  )

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
      },
      "scalars": {},
  }

  new_state = state.apply_gradients(grads=grads)
  return new_state, scheduler_state, metrics, new_rng


def eval_step(state, data, rng, scheduler_state, scheduler, config):
  """
  Computes the evaluation loss for a single batch without updating model weights.
  """

  # The loss function logic is identical to training. We are evaluating the model's
  # ability to perform its core training objective (e.g., denoising).
  def loss_fn(
      params,
      latents,
      encoder_hidden_states,
      timesteps,
      rng,
      conditioning_latents,
  ):
    # Reconstruct the model from its definition and parameters
    model = nnx.merge(state.graphdef, params, state.rest_of_state)

    noise = jax.random.normal(key=rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = (
        scheduler.apply_flow_match(noise, latents, timesteps)
    )
    # Get the model's prediction
    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        control_hidden_states=conditioning_latents,
        deterministic=True,
    )

    # Calculate the loss against the target
    model_pred = model_pred.astype(jnp.float32)
    training_target = training_target.astype(jnp.float32)

    loss = (training_target - model_pred) ** 2
    if not config.disable_training_weights:
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight

    # Calculate the mean loss per sample across all non-batch dimensions.
    loss = loss.reshape(loss.shape[0], -1).mean(axis=1)

    return loss

  # --- Key Difference from train_step ---
  # Directly compute the loss without calculating gradients.
  # The model's state.params are used but not updated.
  # TODO(coolkp): Explore optimizing the creation of PRNGs in a vmap or statically outside of the loop
  bs = len(data["latents"])
  single_batch_size = config.global_batch_size_to_train_on
  losses = jnp.zeros(bs)
  for i in range(0, bs, single_batch_size):
    start = i
    end = min(i + single_batch_size, bs)
    latents = data["latents"][start:end, :].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"][start:end, :].astype(config.weights_dtype)
    conditioning_latents = data["conditioning_latents"][start:end, :].astype(
        config.weights_dtype
    )
    timesteps = data["timesteps"][start:end].astype("int64")
    _, new_rng = jax.random.split(rng, num=2)
    loss = loss_fn(
        state.params,
        latents,
        encoder_hidden_states,
        timesteps,
        new_rng,
        conditioning_latents,
    )
    losses = losses.at[start:end].set(loss)

  # Structure the metrics for logging and aggregation
  metrics = {"scalar": {"learning/eval_loss": losses}}

  # Return the computed metrics and the new RNG key for the next eval step
  return metrics, new_rng
