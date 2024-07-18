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

import importlib
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from maxdiffusion import (
  max_utils,
  FlaxStableDiffusionXLPipeline,
  FlaxStableDiffusionPipeline,
  FlaxUNet2DConditionModel,
  FlaxAutoencoderKL
)
from transformers import (
  CLIPTokenizer,
  FlaxCLIPTextModel,
  PretrainedConfig
)
# from maxdiffusion.checkpointing import (
#   load_stable_diffusion_configs,
#   STABLE_DIFFUSION_CHECKPOINT,
# )


from .models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax

def load_sdxllightning_unet(config, pipeline, params):
  """Load lightning """
  if not config.lightning_from_pt:
    raise ValueError("Only loading lightning models from Pytorch is currently supported.")
  unet_lightning_state_dict = load_file(hf_hub_download(config.lightning_repo, config.lightning_ckpt), device="cpu")
  flax_unet_dict = convert_pytorch_state_dict_to_flax(unet_lightning_state_dict, pipeline.unet)
  params["unet"] = flax_unet_dict
  return pipeline, params

def vae_apply(images, sample_rng, vae, vae_params):
  """Apply vae encoder to images."""
  vae_outputs = vae.apply(
    {"params" : vae_params}, images,
      deterministic=True, method=vae.encode
  )
  latents = vae_outputs.latent_dist.sample(sample_rng)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor

  return latents

def transform_images(
      examples,
      image_column,
      image_resolution,
      rng,
      global_batch_size,
      pixel_ids_key="pixel_values",
      p_vae_apply = None
      ):
    """Preprocess images to latents."""
    images = list(examples[image_column])
    images = [np.asarray(image) for image in images]
    tensor_list = []
    for image in images:
        image = tf.image.resize(image, [image_resolution, image_resolution], method="bilinear", antialias=True)
        image = image / 255.0
        image = (image - 0.5) / 0.5
        image = tf.transpose(image, perm=[2,0,1])
        tensor_list.append(image)

    if p_vae_apply:
        tensor_list = np.stack(tensor_list)
        ds_length = tensor_list.shape[0]
        iters = ds_length // global_batch_size
        latents_list = []
        for i in range(0, iters * global_batch_size, global_batch_size):
            sample_rng, rng = jax.random.split(rng)
            latents = p_vae_apply(tensor_list[i:i+global_batch_size], sample_rng)
            latents_list.append(latents)

        latents_list = np.stack(latents_list)
        b1, b2, c, l1, l2 = latents_list.shape
        latents_list = np.reshape(latents_list, (b1*b2,c, l1, l2))

        # TODO (Juan Acevedo): do last iteration, its required for the Pyarrow dataset
        # to not break due to items being fewer than expected. Is there a better way?
        if tensor_list[i+global_batch_size:].shape[0] != 0:
          sample_rng, rng = jax.random.split(rng)
          latents = p_vae_apply(tensor_list[i+global_batch_size:], sample_rng)
          examples[pixel_ids_key] = np.append(latents_list, latents, axis=0)
        else:
           examples[pixel_ids_key] = latents_list
    else:
        examples[pixel_ids_key] = tf.stack(tensor_list)

    return examples

def get_add_time_ids(original_size, crops_coords_top_left, target_size, bs, dtype):
  add_time_ids = list(original_size + crops_coords_top_left + target_size)
  add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
  return add_time_ids

def override_scheduler_config(scheduler_config, config):
  """Overrides diffusion scheduler params from checkpoint."""

  maxdiffusion_scheduler_config = config.diffusion_scheduler_config

  scheduler_config['_class_name'] = maxdiffusion_scheduler_config.get('_class_name',scheduler_config['_class_name'])
  scheduler_config['prediction_type'] = maxdiffusion_scheduler_config.get('prediction_type',scheduler_config["prediction_type"])
  scheduler_config['timestep_spacing'] = maxdiffusion_scheduler_config.get('timestep_spacing',scheduler_config["timestep_spacing"])
  scheduler_config["rescale_zero_terminal_snr"] = maxdiffusion_scheduler_config.get('rescale_zero_terminal_snr',False)

  return scheduler_config

def create_scheduler(scheduler_config, config):
  """Creates scheduler from config."""
  scheduler_config = override_scheduler_config(scheduler_config, config)

  maxdiffusion_module = importlib.import_module(scheduler_config.__module__.split(".")[0])
  class_name = (
     scheduler_config["_class_name"]
     if scheduler_config["_class_name"].startswith("Flax")
     else "Flax" + scheduler_config["_class_name"]
  )
  cls = getattr(maxdiffusion_module, class_name)
  scheduler = cls.from_config(scheduler_config)

  scheduler_state = scheduler.create_state()
  return scheduler, scheduler_state

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
  """
  Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
  Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
  """
  std_text = jnp.std(noise_pred_text, axis=list(range(1, jnp.ndim(noise_pred_text))), keepdims=True)
  std_cfg = jnp.std(noise_cfg, axis=list(range(1, jnp.ndim(noise_cfg))), keepdims=True)
  # rescale the results from guidance (fixes overexposure)
  noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
  # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
  noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
  return noise_cfg

def get_dummy_unet_inputs(config, pipeline, batch_size):
  """Returns randomly initialized unet inputs."""
  vae_scale_factor = 2 ** (len(pipeline.vae.config['block_out_channels']) -1)
  input_shape = (batch_size,
                  pipeline.unet.config['in_channels'],
                  config.resolution // vae_scale_factor,
                  config.resolution // vae_scale_factor)

  latents = jax.random.normal(jax.random.PRNGKey(0),
                              shape=input_shape,
                              dtype=config.weights_dtype
                              )
  timesteps = jnp.ones((latents.shape[0],))
  encoder_hidden_states_shape = (latents.shape[0],
                                  pipeline.text_encoder.config.max_position_embeddings,
                                  pipeline.unet.cross_attention_dim)
  encoder_hidden_states = jnp.zeros(encoder_hidden_states_shape)
  added_cond_kwargs = None
  if pipeline.unet.addition_embed_type == "text_time":
    unet_config = pipeline.unet.config
    is_refiner = (
      5 * unet_config.addition_time_embed_dim + unet_config.cross_attention_dim
      == unet_config.projection_class_embeddings_input_dim
    )
    num_micro_conditions = 5 if is_refiner else 6

    text_embeds_dim = unet_config.projection_class_embeddings_input_dim - (
      num_micro_conditions * unet_config.addition_time_embed_dim
    )
    time_ids_channels = pipeline.unet.projection_class_embeddings_input_dim - text_embeds_dim
    time_ids_dims = time_ids_channels // pipeline.unet.addition_time_embed_dim
    added_cond_kwargs = {
      "text_embeds": jnp.zeros((batch_size, text_embeds_dim), dtype=config.weights_dtype),
      "time_ids": jnp.zeros((batch_size, time_ids_dims), dtype=config.weights_dtype),
    }
  return (latents, timesteps, encoder_hidden_states, added_cond_kwargs)

def calculate_unet_tflops(config, pipeline, batch_size, rngs, train):
  """
  Calculates unet tflops.
  batch_size should be per_device_batch_size * jax.local_device_count() or attention's shard_map won't
  cache the compilation when flash is enabled.
  """

  (latents, timesteps,
    encoder_hidden_states, added_cond_kwargs) = get_dummy_unet_inputs(config, pipeline, batch_size)
  return max_utils.calculate_model_tflops(
    pipeline.unet,
    rngs,
    train,
    sample=latents,
    timesteps=timesteps,
    encoder_hidden_states=encoder_hidden_states,
    added_cond_kwargs=added_cond_kwargs) / jax.local_device_count()

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

def tokenize_captions(examples, caption_column, tokenizer, input_ids_key="input_ids", p_encode=None):
    """Tokenize captions for sd1.x,sd2.x models."""
    captions = list(examples[caption_column])
    text_inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True
    )

    if p_encode:
        encoder_hidden_states = p_encode(np.stack(text_inputs.input_ids))
        examples[input_ids_key] = encoder_hidden_states
    else:
        examples[input_ids_key] = text_inputs.input_ids
    return examples

def get_shaped_batch(config, pipeline):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078.
  This function works with sd1.x and 2.x.
  """
  vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
  total_train_batch_size = config.per_device_batch_size * jax.device_count()
  if config.cache_latents_text_encoder_outputs:
    batch_image_shape = (total_train_batch_size, 4,
            config.resolution // vae_scale_factor,
            config.resolution // vae_scale_factor)
    #bs, encoder_input, seq_length
    batch_ids_shape = (total_train_batch_size,
                       pipeline.text_encoder.config.max_position_embeddings,
                       pipeline.text_encoder.config.hidden_size)
  else:
    batch_image_shape = (total_train_batch_size, 3, config.resolution, config.resolution)
    batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings)
  shaped_batch = {}
  shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, jnp.float32)
  shaped_batch["input_ids"] = jax.ShapeDtypeStruct(batch_ids_shape, jnp.float32)
  return shaped_batch


# def load_stable_diffusion_checkpoint(
#   checkpoint_manager,
#   mesh,
#   config,
#   checkpoint_type,
#   is_training,
#   train_step = None,
#   scheduler_class = None,
#   step = None
# ):
  
#   # tx = None
#   # if is_training:
#   #   if config.scale_lr:
#   #     config.learning_rate = config.learning_rate * max_utils.get_global_batch_size(config)
    
#   #   learning_rate_scheduler = max_utils.create_learning_rate_schedule(config)

#   #   tx = optax.adamw(
#   #       learning_rate=learning_rate_scheduler,
#   #       b1=config.adam_b1,
#   #       b2=config.adam_b2,
#   #       eps=config.adam_eps,
#   #       weight_decay=config.adam_weight_decay,
#   #   )
  
#   if checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
#     pipeline_class = FlaxStableDiffusionPipeline
#   else:
#     pipeline_class = FlaxStableDiffusionXLPipeline

#   precision = max_utils.get_precision(config)
#   flash_block_sizes = max_utils.get_flash_block_sizes(config)

#   # try loading using orbax, if not, use diffusers loading
#   model_configs = load_stable_diffusion_configs(checkpoint_manager,checkpoint_type, step)
#   if model_configs:
#     # TODO
#     # 1. load from orbax all weights
#     # 1. load states
#     # 1. Create a pipeline
#     # 1. return pipeline, params, states
#     unet = FlaxUNet2DConditionModel.from_config(
#       model_configs[0]["unet_config"],
#       dtype=config.activations_dtype,
#       from_pt=config.from_pt,
#       split_head_dim=config.split_head_dim,
#       norm_num_groups=config.norm_num_groups,
#       attention_kernel=config.attention,
#       flash_block_sizes=flash_block_sizes,
#       mesh=mesh,
#       precision=precision
#     )
#     vae = FlaxAutoencoderKL.from_config(
#       model_configs[0]["vae_config"],
#       dtype=config.activations_dtype,
#       from_pt=config.from_pt
#     )
#     te_pretrained_config = PretrainedConfig.from_dict(model_configs[0]["text_encoder_config"])
#     text_encoder = FlaxCLIPTextModel(
#       te_pretrained_config,
#       seed=config.seed,
#       dtype=config.activations_dtype
#     )
#     tokenizer = CLIPTokenizer.from_pretrained(
#       config.tokenizer_model_name_or_path,
#       subfolder="tokenizer",
#       dtype=config.activations_dtype,
#     )
#     scheduler = None
#     if scheduler_class:
#       scheduler = scheduler_class.from_config(model_configs[0]["scheduler_config"])
    
#     pipeline_kwargs = {
#       "unet" : unet,
#       "vae" : vae,
#       "text_encoder" : text_encoder,
#       "scheduler" : scheduler,
#       "tokenizer" : tokenizer,
#     }

#     if checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
#       pipeline_kwargs["safety_checker"] = None
#       pipeline_kwargs["feature_extractor"] = None
#     else:
#       te_pretrained_2_config = PretrainedConfig.from_dict(model_configs[0]["text_encoder_2_config"])
#       text_encoder_2 = FlaxCLIPTextModel(
#         te_pretrained_2_config,
#         seed=config.seed,
#         dtype=config.activations_dtype
#       )
#       pipeline_kwargs["text_encoder_2"] = text_encoder_2
#       pipeline_kwargs["tokenizer_2"] = tokenizer

#     pipeline = pipeline_class(
#       **pipeline_kwargs
#     )

#     (unet_state,
#      unet_state_mesh_shardings,
#      vae_state,
#      vae_state_shardings) = max_utils.get_states(pipeline)

#   else:
#     pipeline, params = pipeline_class.from_pretrained(
#       config.pretrained_model_name_or_path,
#       revision=config.revision,
#       dtype=config.activations_dtype,
#       safety_checker=None,
#       feature_extractor=None,
#       from_pt=config.from_pt,
#       split_head_dim=config.split_head_dim,
#       norm_num_groups=config.norm_num_groups,
#       attention_kernel=config.attention,
#       flash_block_sizes=flash_block_sizes,
#       mesh=mesh,
#       precision=precision
#     )

#     params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)

#     # (unet_state,
#     #  unet_state_mesh_shardings,
#     #  vae_state,
#     #  vae_state_shardings) = max_utils.get_states(
#     #     mesh, tx, jax.random.PRNGKey(config.seed),
#     #     config, pipeline, params["unet"],
#     #     params["vae"], checkpoint_manager, training=is_training)
    
#     # text_encoder_sharding = PositionalSharding(mesh.devices).replicate()
#     # partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=text_encoder_sharding)
#     # params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])

#     # text_encoder_state = train_state.Train_state.create(
#     #   appy_fn=pipeline.text_encoder.__call__,
#     #   params=params["text_encoder"],
#     #   tx=tx
#     # )

#     # if is_training:
#     #   with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
#     #     p_train_step = jax.jit(
#     #       train_step,
#     #       in_shardings=(unet_state_mesh_shardings, None, my_data_sharding, None),
#     #       out_shardings=(unet_state_mesh_shardings, None, None, None),
#     #       donate_argnums=(0,)
#     #     )

#   return pipeline, params, unet_state, vae_state, text_encoder_state