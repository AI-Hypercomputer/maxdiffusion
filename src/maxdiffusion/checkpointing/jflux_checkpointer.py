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

from abc import ABC
import functools
import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from maxdiffusion import (max_utils)
from maxdiffusion.pipelines.jflux.pipeline_jflux import JfluxPipeline
from maxdiffusion.models.flux_utils import configs
from maxdiffusion.models.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.models.embeddings_flax import HFEmbedder
from maxdiffusion.models.flux_utils import load_ae
from flax.linen import partitioning as nn_partitioning

from maxdiffusion.checkpointing.checkpointing_utils import (
    create_orbax_checkpoint_manager,
)


def get_device_type():
  """Returns the type of JAX device being used.

  Returns:
    str: "gpu", "tpu", or "cpu"
  """
  try:
    device_kind = jax.devices()[0].device_kind
    if "tpu" in device_kind.lower():
      return "tpu"
    elif "amd" in device_kind.lower():
      return "rocm"
    elif "nvidia" in device_kind.lower():
      return "cuda"
    else:
      return "cpu"
  except IndexError:
    return "cpu"  # No devices found, likely using CPU


class JfluxCheckpointer(ABC):
  flux_state_item_name = "flux_state"
  config_item_name = "config"

  def __init__(self, config):
    self.config = config

    self.rng = jax.random.PRNGKey(self.config.seed)
    devices_array = max_utils.create_device_mesh(config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    self.total_train_batch_size = self.config.total_train_batch_size

    self.checkpoint_manager = create_orbax_checkpoint_manager(
        self.config.checkpoint_dir,
        enable_checkpointing=True,
        save_interval_steps=self.config.save_interval_steps,
        checkpoint_type="none",
        item_names={JfluxCheckpointer.flux_state_item_name, JfluxCheckpointer.config_item_name},
    )

  def _create_optimizer(self, config):
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(config)
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

  def create_flux_state(self, flux, init_flux_weights, params, is_training, use_jit=True):
    tx, learning_rate_scheduler = None, None
    if is_training:

      tx, learning_rate_scheduler = self._create_optimizer(self.config)

    if init_flux_weights is not None:
      weights_init_fn = functools.partial(init_flux_weights, rng=self.rng)
    else:
      weights_init_fn = None
    flux_state, state_mesh_shardings = max_utils.setup_initial_state(
        model=flux,
        tx=tx,
        config=self.config,
        mesh=self.mesh,
        weights_init_fn=weights_init_fn,
        model_params=params.get(JfluxCheckpointer.flux_state_item_name, None) if params is not None else None,
        checkpoint_manager=self.checkpoint_manager,
        checkpoint_item=JfluxCheckpointer.flux_state_item_name,
        training=is_training,
        use_jit=use_jit,
    )

    return flux_state, state_mesh_shardings, learning_rate_scheduler

  def _get_pipeline_class(self):
    return JfluxPipeline

  def save_checkpoint(self, train_step, pipeline, train_states):
    items = {
        JfluxCheckpointer.config_item_name: ocp.args.JsonSave({"model_name": self.config.model_name}),
    }

    items[JfluxCheckpointer.flux_state_item_name] = ocp.args.PyTreeSave(train_states[JfluxCheckpointer.flux_state_item_name])

    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))

  def load_pretrained_model(self, model_name):
    # This code to generate the safetensors filename may not generalize
    # but loading does not work without it
    print(f"loading pretrained model {self.config.pretrained_model_name_or_path}")
    stname = self.config.pretrained_model_name_or_path.split("/")[1].lower().replace(".", "")
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      flux, weights = FluxTransformer2DModel.from_pretrained(
          pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
          subfolder="transformer",
          from_pt=True,
          filename=f"{stname}.safetensors",
          mesh=self.mesh,
      )
      weights = jax.tree_util.tree_map(lambda x: x.astype(self.config.weights_dtype), weights)
    return flux, weights

  def load_checkpoint(self, step=None, scheduler_class=None):
    with jax.default_device(jax.devices("cpu")[0]):
      t5 = HFEmbedder(
          "ariG23498/t5-v1-1-xxl-flax",
          max_length=256 if self.config.model_name == "flux-schnell" else 512,
          dtype=jax.numpy.bfloat16,
      )

      clip = HFEmbedder(
          "ariG23498/clip-vit-large-patch14-text-flax",
          max_length=77,
          dtype=jax.numpy.bfloat16,
      )

    ae = load_ae(self.config.model_name, "cpu")

    precision = max_utils.get_precision(self.config)
    flash_block_sizes = max_utils.get_flash_block_sizes(self.config)
    data_sharding = jax.sharding.NamedSharding(self.mesh, PartitionSpec(*self.config.data_sharding))
    # loading from pretrained here causes a crash when trying to compile the model
    # Failed to load HSACO: HIP_ERROR_NoBinaryForGpu
    model_params = configs[self.config.model_name].params
    flux = FluxTransformer2DModel(
        num_layers=model_params.depth,
        num_single_layers=model_params.depth_single_blocks,
        in_channels=model_params.in_channels,
        attention_head_dim=int(model_params.hidden_size / model_params.num_heads),
        num_attention_heads=model_params.num_heads,
        joint_attention_dim=model_params.context_in_dim,
        pooled_projection_dim=model_params.vec_in_dim,
        mlp_ratio=model_params.mlp_ratio,
        qkv_bias=model_params.qkv_bias,
        theta=model_params.theta,
        guidance_embeds=model_params.guidance_embed,
        axes_dims_rope=model_params.axes_dim,
        dtype=self.config.activations_dtype,
        weights_dtype=self.config.weights_dtype,
        attention_kernel=self.config.attention,
        flash_block_sizes=flash_block_sizes,
        mesh=self.mesh,
        precision=precision,
    )

    return JfluxPipeline(t5, clip, flux, ae, dtype=self.config.activations_dtype, sharding=data_sharding, scheduler=None)
