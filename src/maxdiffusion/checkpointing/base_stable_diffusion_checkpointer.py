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
from contextlib import nullcontext
import os
import json
import functools
import jax
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from maxdiffusion import (
    max_utils,
    FlaxStableDiffusionPipeline,
    FlaxStableDiffusionXLPipeline,
    FlaxUNet2DConditionModel,
    FlaxAutoencoderKL,
    max_logging,
)

from maxdiffusion.transformers import (CLIPTokenizer, FlaxCLIPTextModel, CLIPTextConfig, FlaxCLIPTextModelWithProjection)

from maxdiffusion.checkpointing.checkpointing_utils import (
    create_orbax_checkpoint_manager,
    load_stable_diffusion_configs,
)

STABLE_DIFFUSION_CHECKPOINT = "STABLE_DIFFUSION_CHECKPOINT"
STABLE_DIFFUSION_XL_CHECKPOINT = "STABLE_DIFUSSION_XL_CHECKPOINT"
_CHECKPOINT_FORMAT_DIFFUSERS = "CHECKPOINT_FORMAT_DIFFUSERS"
_CHECKPOINT_FORMAT_ORBAX = "CHECKPOINT_FORMAT_ORBAX"


class BaseStableDiffusionCheckpointer(ABC):

  def __init__(self, config, checkpoint_type):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.checkpoint_format = None

    self.rng = jax.random.PRNGKey(self.config.seed)
    devices_array = max_utils.create_device_mesh(config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    self.total_train_batch_size = self.config.total_train_batch_size

    self.checkpoint_manager = create_orbax_checkpoint_manager(
        self.config.checkpoint_dir, enable_checkpointing=True, save_interval_steps=1, checkpoint_type=checkpoint_type
    )

  def _create_optimizer(self, config, learning_rate):

    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

  def create_unet_state(self, pipeline, params, checkpoint_item_name, is_training):

    tx, learning_rate_scheduler = None, None
    if is_training:
      learning_rate = self.config.learning_rate

      tx, learning_rate_scheduler = self._create_optimizer(self.config, learning_rate)

    weights_init_fn = functools.partial(pipeline.unet.init_weights, rng=self.rng)
    unet_state, state_mesh_shardings = max_utils.setup_initial_state(
        model=pipeline.unet,
        tx=tx,
        config=self.config,
        mesh=self.mesh,
        weights_init_fn=weights_init_fn,
        model_params=None if self.config.train_new_unet else params.get("unet", None),
        checkpoint_manager=self.checkpoint_manager,
        checkpoint_item=checkpoint_item_name,
        training=is_training,
    )
    return unet_state, state_mesh_shardings, learning_rate_scheduler

  def create_vae_state(self, pipeline, params, checkpoint_item_name, is_training=False):

    # Currently VAE training is not supported.
    weights_init_fn = functools.partial(pipeline.vae.init_weights, rng=self.rng)
    return max_utils.setup_initial_state(
        model=pipeline.vae,
        tx=None,
        config=self.config,
        mesh=self.mesh,
        weights_init_fn=weights_init_fn,
        model_params=params.get("vae", None),
        checkpoint_manager=self.checkpoint_manager,
        checkpoint_item=checkpoint_item_name,
        training=is_training,
    )

  def create_text_encoder_state(self, pipeline, params, checkpoint_item_name, is_training):

    tx = None
    if is_training:
      learning_rate = self.config.text_encoder_learning_rate
      tx, learning_rate_scheduler = self._create_optimizer(self.config, learning_rate)
      self.text_encoder_learning_rate_scheduler = learning_rate_scheduler

    weights_init_fn = functools.partial(
        pipeline.text_encoder.init_weights,
        rng=self.rng,
        input_shape=(self.total_train_batch_size, pipeline.tokenizer.model_max_length),
    )

    return max_utils.setup_initial_state(
        model=pipeline.text_encoder,
        tx=tx,
        config=self.config,
        mesh=self.mesh,
        weights_init_fn=weights_init_fn,
        model_params=params.get("text_encoder", None),
        checkpoint_manager=self.checkpoint_manager,
        checkpoint_item=checkpoint_item_name,
        training=is_training,
    )

  def create_text_encoder_2_state(self, pipeline, params, checkpoint_item_name, is_training):
    tx = None
    if is_training:
      learning_rate = self.config.text_encoder_learning_rate
      tx, learning_rate_scheduler = self._create_optimizer(self.config, learning_rate)
      self.text_encoder_learning_rate_scheduler = learning_rate_scheduler

    weights_init_fn = functools.partial(
        pipeline.text_encoder_2.init_weights,
        rng=self.rng,
        input_shape=(self.total_train_batch_size, pipeline.tokenizer.model_max_length),
    )

    return max_utils.setup_initial_state(
        model=pipeline.text_encoder_2,
        tx=tx,
        config=self.config,
        mesh=self.mesh,
        weights_init_fn=weights_init_fn,
        model_params=params.get("text_encoder_2", None),
        checkpoint_manager=self.checkpoint_manager,
        checkpoint_item=checkpoint_item_name,
        training=is_training,
    )

  def _get_pipeline_class(self):
    if self.checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
      pipeline_class = FlaxStableDiffusionPipeline
    else:
      pipeline_class = FlaxStableDiffusionXLPipeline

    return pipeline_class

  def _set_checkpoint_format(self, checkpoint_format):
    self.checkpoint_format = checkpoint_format

  def load_diffusers_checkpoint(self):
    pipeline_class = self._get_pipeline_class()

    precision = max_utils.get_precision(self.config)
    flash_block_sizes = max_utils.get_flash_block_sizes(self.config)

    # Multiprocess computations aren't implemented on cpu backend.
    if jax.device_count() == jax.local_device_count():
      context = jax.default_device(jax.devices("cpu")[0])
    else:
      context = nullcontext()
    with context:
      pipeline, params = pipeline_class.from_pretrained(
          self.config.pretrained_model_name_or_path,
          revision=self.config.revision,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          safety_checker=None,
          feature_extractor=None,
          from_pt=self.config.from_pt,
          split_head_dim=self.config.split_head_dim,
          norm_num_groups=self.config.norm_num_groups,
          attention_kernel=self.config.attention,
          flash_block_sizes=flash_block_sizes,
          mesh=self.mesh,
          precision=precision,
      )

    if len(self.config.unet_checkpoint) > 0:
      unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
          self.config.unet_checkpoint,
          split_head_dim=self.config.split_head_dim,
          norm_num_groups=self.config.norm_num_groups,
          attention_kernel=self.config.attention,
          flash_block_sizes=flash_block_sizes,
          dtype=self.activations_dtype,
          weights_dtype=self.weights_dtype,
          mesh=self.mesh,
      )
      params["unet"] = unet_params
      pipeline.unet = unet
    params = jax.tree_util.tree_map(lambda x: x.astype(self.config.weights_dtype), params)

    return pipeline, params

  def save_checkpoint(self, train_step, pipeline, params, train_states):
    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    items = {
        "unet_config": ocp.args.JsonSave(config_to_json(pipeline.unet)),
        "vae_config": ocp.args.JsonSave(config_to_json(pipeline.vae)),
        "text_encoder_config": ocp.args.JsonSave(config_to_json(pipeline.text_encoder.config)),
        "scheduler_config": ocp.args.JsonSave(config_to_json(pipeline.scheduler)),
    }

    items["unet_state"] = ocp.args.PyTreeSave(train_states["unet_state"])
    items["vae_state"] = ocp.args.PyTreeSave(train_states["vae_state"])
    items["text_encoder_state"] = ocp.args.PyTreeSave(train_states["text_encoder_state"])

    if hasattr(pipeline, "text_encoder_2"):
      items["text_encoder_2_state"] = ocp.args.PyTreeSave(train_states["text_encoder_2_state"])
      items["text_encoder_2_config"] = ocp.args.JsonSave(config_to_json(pipeline.text_encoder_2.config))

    tokenizer_config = {"path": self.config.tokenizer_model_name_or_path}
    items["tokenizer_config"] = ocp.args.JsonSave(tokenizer_config)

    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))

  def load_params(self, step=None):

    self.checkpoint_format = _CHECKPOINT_FORMAT_ORBAX

  def load_checkpoint(self, step=None, scheduler_class=None):

    pipeline_class = self._get_pipeline_class()

    self.checkpoint_format = _CHECKPOINT_FORMAT_ORBAX

    precision = max_utils.get_precision(self.config)
    flash_block_sizes = max_utils.get_flash_block_sizes(self.config)
    # try loading using orbax, if not, use diffusers loading
    model_configs = load_stable_diffusion_configs(self.config, self.checkpoint_manager, self.checkpoint_type, step)

    pipeline, params = None, {}

    if model_configs:
      unet = FlaxUNet2DConditionModel.from_config(
          model_configs[0]["unet_config"],
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          from_pt=self.config.from_pt,
          split_head_dim=self.config.split_head_dim,
          norm_num_groups=self.config.norm_num_groups,
          attention_kernel=self.config.attention,
          flash_block_sizes=flash_block_sizes,
          mesh=self.mesh,
          precision=precision,
      )

      vae = FlaxAutoencoderKL.from_config(
          model_configs[0]["vae_config"],
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          from_pt=self.config.from_pt,
      )

      tokenizer_path = model_configs[0]["tokenizer_config"]["path"]
      if "gs://" in tokenizer_path:
        if "tokenizer" not in tokenizer_path:
          tokenizer_path = os.path.join(tokenizer_path, "tokenizer")
        tokenizer_path = max_utils.download_blobs(tokenizer_path, "/tmp")
      tokenizer = CLIPTokenizer.from_pretrained(
          tokenizer_path, subfolder="tokenizer", dtype=self.config.activations_dtype, weights_dtype=self.config.weights_dtype
      )

      te_pretrained_config = CLIPTextConfig(**model_configs[0]["text_encoder_config"])
      text_encoder = FlaxCLIPTextModel(
          te_pretrained_config,
          seed=self.config.seed,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          _do_init=False,
      )

      scheduler = None
      if scheduler_class:
        scheduler = scheduler_class.from_config(model_configs[0]["scheduler_config"])

      pipeline_kwargs = {
          "unet": unet,
          "vae": vae,
          "text_encoder": text_encoder,
          "scheduler": scheduler,
          "tokenizer": tokenizer,
      }

      if self.checkpoint_type == STABLE_DIFFUSION_XL_CHECKPOINT:
        te_pretrained_2_config = CLIPTextConfig(**model_configs[0]["text_encoder_2_config"])
        text_encoder_2 = FlaxCLIPTextModelWithProjection(
            te_pretrained_2_config, seed=self.config.seed, dtype=self.config.activations_dtype, _do_init=False
        )
        pipeline_kwargs["text_encoder_2"] = text_encoder_2
        # both tokenizers in sdxl are the same.
        pipeline_kwargs["tokenizer_2"] = tokenizer

      pipeline = pipeline_class(**pipeline_kwargs)
    else:
      max_logging.log(f"loading checkpoint specified in config : {self.config.pretrained_model_name_or_path}")
      self.checkpoint_format = _CHECKPOINT_FORMAT_DIFFUSERS
      pipeline, params = self.load_diffusers_checkpoint()

    return pipeline, params
