# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import math
import os
from jax import Array
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union, Tuple
from einops import rearrange
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from maxdiffusion.models.ltx_video.autoencoders.vae_torchax import TorchaxCausalVideoAutoencoder
# from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
import yaml
from transformers import (CLIPTokenizer, FlaxCLIPTextModel,
                          T5EncoderModel, FlaxT5EncoderModel, AutoTokenizer)

from maxdiffusion.models.ltx_video.autoencoders.latent_upsampler import LatentUpsampler
from torchax import interop
from torchax import default_env
import imageio
import json
import numpy as np
import torch
from safetensors import safe_open
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from huggingface_hub import hf_hub_download
from maxdiffusion.models.ltx_video.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from maxdiffusion.models.ltx_video.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    un_normalize_latents,
    normalize_latents,
)
# from diffusers.image_processor import VaeImageProcessor
from maxdiffusion.models.ltx_video.autoencoders.latent_upsampler import LatentUpsampler
from maxdiffusion.models.ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
from math import e
from types import NoneType
from typing import Any, Dict
import numpy as np
import inspect

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from typing import Optional, Union, List
import torch
from maxdiffusion.checkpointing import checkpointing_utils
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.models.ltx_video.transformers.symmetric_patchifier import SymmetricPatchifier
from maxdiffusion.models.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ...pyconfig import HyperParameters
# from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState
from ...schedulers.scheduling_rectified_flow import FlaxRectifiedFlowMultistepScheduler, RectifiedFlowSchedulerState
from ...max_utils import (
    create_device_mesh,
    setup_initial_state,
    get_memory_allocations
)
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel
import os
import json
import functools
import orbax.checkpoint as ocp
import pickle


class PickleCheckpointHandler(ocp.CheckpointHandler):
    def save(self, directory: str, item, args=None):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(item, f)

    def restore(self, directory: str, args=None):
        with open(os.path.join(directory, 'checkpoint.pkl'), 'rb') as f:
            return pickle.load(f)

    def structure(self, directory: str):
        return {}  # not needed for simple pickle-based handling


def save_tensor_dict(tensor_dict, timestep):
    base_dir = os.path.dirname(__file__)
    local_path = os.path.join(base_dir, f"schedulerTest{timestep}")

    try:
        torch.save(tensor_dict, local_path)
        print(f"Dictionary of tensors saved to: {local_path}")
    except Exception as e:
        print(f"Error saving dictionary: {e}")
        raise


def validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids):
    print("prompts_embeds.shape: ", prompt_embeds.shape, prompt_embeds.dtype)
    print("fractional_coords.shape: ",
          fractional_coords.shape, fractional_coords.dtype)
    print("latents.shape: ", latents.shape, latents.dtype)
    print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
    print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
    # print("segment_ids.shape: ", segment_ids.shape, segment_ids.dtype)
    print("encoder_attention_segment_ids.shape: ",
          encoder_attention_segment_ids.shape, encoder_attention_segment_ids.dtype)


def prepare_extra_step_kwargs(generator):
    extra_step_kwargs = {}
    extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


class LTXVideoPipeline:
    def __init__(
        self,
        transformer: Transformer3DModel,
        scheduler: FlaxRectifiedFlowMultistepScheduler, 
        scheduler_state: RectifiedFlowSchedulerState,
        vae: TorchaxCausalVideoAutoencoder,
        text_encoder,
        patchifier,
        tokenizer,
        prompt_enhancer_image_caption_model,
        prompt_enhancer_image_caption_processor,
        prompt_enhancer_llm_model,
        prompt_enhancer_llm_tokenizer,
        devices_array: np.array,
        mesh: Mesh,
        config: HyperParameters,
        transformer_state: Dict[Any, Any] = None,
        transformer_state_shardings: Dict[Any, Any] = NoneType,
    ):
        self.transformer = transformer
        self.devices_array = devices_array
        self.mesh = mesh
        self.config = config
        self.p_run_inference = None
        self.transformer_state = transformer_state
        self.transformer_state_shardings = transformer_state_shardings
        self.scheduler = scheduler
        self.scheduler_state = scheduler_state
        self.vae = vae
        self.text_encoder = text_encoder
        self.patchifier = patchifier
        self.tokenizer = tokenizer
        self.prompt_enhancer_image_caption_model = prompt_enhancer_image_caption_model
        self.prompt_enhancer_image_caption_processor = prompt_enhancer_image_caption_processor
        self.prompt_enhancer_llm_model = prompt_enhancer_llm_model
        self.prompt_enhancer_llm_tokenizer = prompt_enhancer_llm_tokenizer
        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            self.vae
        )
        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor)

    @classmethod
    def load_scheduler(cls, ckpt_path, config):
        # scheduler, scheduler_state = FlaxUniPCMultistepScheduler.from_pretrained(
        #     "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        #     subfolder="scheduler",
        #     flow_shift=5.0  # 5.0 for 720p, 3.0 for 480p
        # )
        # scheduler, scheduler_state = FlaxRectifiedFlowMultistepScheduler.from_pretrained(
        #     "Lightricks/LTX-Video",
        #     subfolder="scheduler",
        #     flow_shift=5.0  # 5.0 for 720p, 3.0 for 480p
        # )
        # import pdb; pdb.set_trace()
        # scheduler = FlaxRectifiedFlowMultistepScheduler(
        #     sampler="LinearQuadratic"
        # )
        # scheduler_state = scheduler.create_state()
        if config.sampler == "from_checkpoint" or not config.sampler:
            scheduler = FlaxRectifiedFlowMultistepScheduler.from_pretrained_jax(ckpt_path)
        else:
            scheduler = FlaxRectifiedFlowMultistepScheduler(
                sampler=("Uniform" if config.sampler.lower() == "uniform" else "LinearQuadratic")
            )
        scheduler_state = scheduler.create_state()
        
        return scheduler, scheduler_state

    @classmethod
    def load_transformer(cls, config):
        devices_array = create_device_mesh(config)
        mesh = Mesh(devices_array, config.mesh_axes)
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(
            base_dir, "../../models/ltx_video/xora_v1.2-13B-balanced-128.json")
        with open(config_path, "r") as f:
            model_config = json.load(f)
        relative_ckpt_path = model_config["ckpt_path"]

        ignored_keys = ["_class_name", "_diffusers_version", "_name_or_path",
                        "causal_temporal_positioning", "in_channels", "ckpt_path"]
        in_channels = model_config["in_channels"]
        for name in ignored_keys:
            if name in model_config:
                del model_config[name]
        transformer = Transformer3DModel(
            # change this sharding back
            **model_config, dtype=jnp.float32, gradient_checkpointing="matmul_without_batch", sharding_mesh=mesh)  #move the key outwards
        transformer_param_shapes = transformer.init_weights(
            in_channels, jax.random.PRNGKey(0), model_config['caption_channels'], eval_only=True)
        weights_init_fn = functools.partial(
            transformer.init_weights,
            in_channels,
            jax.random.PRNGKey(0),
            model_config['caption_channels'],
            eval_only=True
        )
        absolute_ckpt_path = os.path.abspath(relative_ckpt_path)

        checkpoint_manager = ocp.CheckpointManager(absolute_ckpt_path)
        transformer_state, transformer_state_shardings = setup_initial_state(
            model=transformer,
            tx=None,
            config=config,
            mesh=mesh,
            weights_init_fn=weights_init_fn,
            checkpoint_manager=checkpoint_manager,
            checkpoint_item=" ",
            model_params=None,
            training=False,
        )
        transformer_state = jax.device_put(
            transformer_state, transformer_state_shardings)
        get_memory_allocations()

        return transformer, transformer_state, transformer_state_shardings

    @classmethod
    def load_vae(cls, ckpt_path):
   
        torch_vae = CausalVideoAutoencoder.from_pretrained(ckpt_path, torch_dtype = torch.bfloat16)
        with default_env():
            torch_vae = torch_vae.to('jax')
            jax_vae = TorchaxCausalVideoAutoencoder(torch_vae)
        return jax_vae

    @classmethod
    def load_text_encoder(cls, ckpt_path):
        # text_encoder = T5EncoderModel.from_pretrained(
        #     ckpt_path, subfolder="text_encoder"
        # )
        t5_encoder = FlaxT5EncoderModel.from_pretrained(ckpt_path)
        cpus = jax.devices("cpu")
        t5_encoder.params = jax.device_put(t5_encoder.params, device=cpus[0])
        return t5_encoder

    @classmethod
    def load_tokenizer(cls, config, ckpt_path):
        # tokenizer = T5Tokenizer.from_pretrained(
        #   ckpt_path, subfolder="tokenizer"
        # )
        t5_tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, max_length=config.max_sequence_length, use_fast=True
        )
        return t5_tokenizer

    @classmethod
    def load_prompt_enhancement(cls, config):
        prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
            config.prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
        )
        prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
            config.prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
        )
        prompt_enhancer_llm_model = AutoModelForCausalLM.from_pretrained(
            config.prompt_enhancer_llm_model_name_or_path, torch_dtype="bfloat16",
        )
        prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(
            config.prompt_enhancer_llm_model_name_or_path,
        )
        return prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer

    @classmethod
    def from_pretrained(cls, config: HyperParameters, enhance_prompt: bool = False):
        devices_array = create_device_mesh(config)
        mesh = Mesh(devices_array, config.mesh_axes)

        transformer, transformer_state, transformer_state_shardings = cls.load_transformer(
            config)

        # load from pytorch version
        models_dir = "/mnt/disks/diffusionproj"  #edit this!
        ltxv_model_name_or_path = "ltxv-13b-0.9.7-dev.safetensors"
        if not os.path.isfile(ltxv_model_name_or_path):
            ltxv_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=ltxv_model_name_or_path,
                local_dir=models_dir,
                repo_type="model",
            )
        else:
            ltxv_model_path = ltxv_model_name_or_path
        
        scheduler, scheduler_state = cls.load_scheduler(ltxv_model_path, config)
    
        vae = cls.load_vae(ltxv_model_path)
        text_encoder = cls.load_text_encoder(
            config.text_encoder_model_name_or_path)
        patchifier = SymmetricPatchifier(patch_size=1)
        tokenizer = cls.load_tokenizer(
            config, config.text_encoder_model_name_or_path)
        
        enhance_prompt = False
        if enhance_prompt:
            prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = cls.load_prompt_enhancement(
                config)
        else:
            prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None

        return LTXVideoPipeline(
            transformer=transformer,
            scheduler=scheduler,
            scheduler_state=scheduler_state,
            vae=vae,
            text_encoder=text_encoder,
            patchifier=patchifier,
            tokenizer=tokenizer,
            prompt_enhancer_image_caption_model=prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor=prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model=prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer=prompt_enhancer_llm_tokenizer,
            devices_array=devices_array,
            mesh=mesh,
            config=config,
            transformer_state=transformer_state,
            transformer_state_shardings=transformer_state_shardings
        )

    @classmethod
    def _text_preprocessing(self, text):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.strip()
            return text

        return [process(t) for t in text]
    def denoising_step(
        scheduler,
        latents: Array,
        noise_pred: Array,
        current_timestep: Optional[Array],
        conditioning_mask: Optional[Array],
        t: float,
        extra_step_kwargs: Dict,
        t_eps: float = 1e-6,
        stochastic_sampling: bool = False,
    ) -> Array:
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:
        Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
        and will start to be denoised when the current timestep is equal or lower than their
        conditioning timestep.
        (hard-conditioning latents with conditioning_mask = 1.0 are never denoised)
        """
        # Denoise the latents using the scheduler
        denoised_latents = scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            stochastic_sampling=stochastic_sampling,
        )

        if conditioning_mask is None:
            return denoised_latents

        tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).astype(jnp.bool_)
        tokens_to_denoise_mask = jnp.expand_dims(tokens_to_denoise_mask, axis=-1)
        return jnp.where(tokens_to_denoise_mask, denoised_latents, latents)
    
    def retrieve_timesteps(  #currently doesn't support custom timesteps
        self,
        scheduler: FlaxRectifiedFlowMultistepScheduler, 
        latent_shape,
        scheduler_state: RectifiedFlowSchedulerState,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
    ):
        scheduler_state = scheduler.set_timesteps(state=scheduler_state, samples_shape=latent_shape, num_inference_steps=num_inference_steps)
        timesteps = scheduler_state.timesteps 
        if (
            skip_initial_inference_steps < 0
            or skip_final_inference_steps < 0
            or skip_initial_inference_steps + skip_final_inference_steps
            >= num_inference_steps # Use the original num_inference_steps here for the check
        ):
            raise ValueError(
                "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
            )
        timesteps = timesteps[
            skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps
        ]
        scheduler_state = scheduler.set_timesteps(timesteps=timesteps, samples_shape = latent_shape, state=scheduler_state) 


        return scheduler_state
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        text_encoder_max_tokens: int = 256,
        **kwargs,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = (
            text_encoder_max_tokens  # TPU supports only lengths multiple of 128
        )
        if prompt_embeds is None:
            assert (
                self.text_encoder is not None
            ), "You should provide either prompt_embeds or self.text_encoder should not be None,"
            # text_enc_device = next(self.text_encoder.parameters())
            prompt = self._text_preprocessing(prompt)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = jnp.array(text_inputs.input_ids)
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1: -1]
                )

            prompt_attention_mask = jnp.array(text_inputs.attention_mask)
            prompt_embeds = self.text_encoder(
                text_input_ids, attention_mask=prompt_attention_mask
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        print(isinstance(prompt_embeds, Array))
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = jnp.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        # prompt_embeds = prompt_embeds.view(
        #     bs_embed * num_images_per_prompt, seq_len, -1
        # )
        prompt_embeds = jnp.reshape(
            prompt_embeds, (bs_embed * num_images_per_prompt, seq_len, -1))
        prompt_attention_mask = jnp.tile(
            prompt_attention_mask, (1, num_images_per_prompt))
        # prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
        # prompt_attention_mask = prompt_attention_mask.view(
        #     bs_embed * num_images_per_prompt, -1
        # )
        prompt_attention_mask = jnp.reshape(
            prompt_attention_mask, (bs_embed * num_images_per_prompt, -1))

        # get unconditional embeddings for classifier free guidance  hasn't changed yet
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = self._text_preprocessing(negative_prompt)
            uncond_tokens = uncond_tokens * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = jnp.array(uncond_input.attention_mask)

            negative_prompt_embeds = self.text_encoder(
                jnp.array(uncond_input.input_ids),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = jnp.tile(negative_prompt_embeds,
                (1, num_images_per_prompt, 1)
            )
            negative_prompt_embeds = jnp.reshape(negative_prompt_embeds,
                (batch_size * num_images_per_prompt, seq_len, -1)
            )

            negative_prompt_attention_mask = jnp.tile(negative_prompt_attention_mask,
                (1, num_images_per_prompt)
            )
            negative_prompt_attention_mask = jnp.reshape(negative_prompt_attention_mask,
                (bs_embed * num_images_per_prompt, -1)
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
        data_sharding = jax.sharding.NamedSharding(self.mesh, P('data'))
        return (
            jax.device_put(prompt_embeds, data_sharding),#(1, 256, 4096)
            jax.device_put(prompt_attention_mask, data_sharding), #1, 256
            jax.device_put(negative_prompt_embeds, data_sharding),
            jax.device_put(negative_prompt_attention_mask, data_sharding)
        )

    def prepare_latents(
        self,
        latents: torch.Tensor | None,
        media_items: torch.Tensor | None,
        timestep: float,
        latent_shape: torch.Size | Tuple[Any, ...],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | List[torch.Generator],
        vae_per_channel_normalize: bool = True,
    ):
        if isinstance(generator, list) and len(generator) != latent_shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {latent_shape[0]}. Make sure the batch size matches the length of the generators."
            )

        # Initialize the latents with the given latents or encoded media item, if provided
        assert (
            latents is None or media_items is None
        ), "Cannot provide both latents and media_items. Please provide only one of the two."

        assert (
            latents is None and media_items is None or timestep < 1.0
        ), "Input media_item or latents are provided, but they will be replaced with noise."

        if media_items is not None:
            latents = self.vae.encode(
                media_itmes = media_items,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )
        if latents is not None:
            assert (
                latents.shape == latent_shape
            ), f"Latents have to be of shape {latent_shape} but are {latents.shape}."

        # For backward compatibility, generate in the "patchified" shape and rearrange
        b, c, f, h, w = latent_shape
        noise = randn_tensor(  ###need to remove this!
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        # scale the initial noise by the standard deviation required by the scheduler
        # noise = noise * self.scheduler.init_noise_sigma !!this doesn;t have

        if latents is None:
            latents = noise
        else:
            # Noise the latents to the required (first) timestep
            timestep = torch.from_numpy(np.array(timestep))
            latents = timestep * noise + (1 - timestep) * latents

        return latents

    def prepare_conditioning(
        self,
        conditioning_items,
        init_latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        vae_per_channel_normalize: bool = True,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

        assert isinstance(self.vae, TorchaxCausalVideoAutoencoder)

        # if conditioning_items:
        #     batch_size, _, num_latent_frames = init_latents.shape[:3]

        #     init_conditioning_mask = torch.zeros(
        #         init_latents[:, 0, :, :, :].shape,
        #         dtype=torch.float32,
        #         device=init_latents.device,
        #     )

        #     extra_conditioning_latents = []
        #     extra_conditioning_pixel_coords = []
        #     extra_conditioning_mask = []
        #     extra_conditioning_num_latents = 0  # Number of extra conditioning latents added (should be removed before decoding)

        #     # Process each conditioning item
        #     for conditioning_item in conditioning_items:
        #         conditioning_item = self._resize_conditioning_item(
        #             conditioning_item, height, width
        #         )
        #         media_item = conditioning_item.media_item
        #         media_frame_number = conditioning_item.media_frame_number
        #         strength = conditioning_item.conditioning_strength
        #         assert media_item.ndim == 5  # (b, c, f, h, w)
        #         b, c, n_frames, h, w = media_item.shape
        #         assert (
        #             height == h and width == w
        #         ) or media_frame_number == 0, f"Dimensions do not match: {height}x{width} != {h}x{w} - allowed only when media_frame_number == 0"
        #         assert n_frames % 8 == 1
        #         assert (
        #             media_frame_number >= 0
        #             and media_frame_number + n_frames <= num_frames
        #         )

        #         # Encode the provided conditioning media item
        #         media_item_latents = vae_encode(
        #             media_item.to(dtype=self.vae.dtype, device=self.vae.device),
        #             self.vae,
        #             vae_per_channel_normalize=vae_per_channel_normalize,
        #         ).to(dtype=init_latents.dtype)

        #         # Handle the different conditioning cases
        #         if media_frame_number == 0:
        #             # Get the target spatial position of the latent conditioning item
        #             media_item_latents, l_x, l_y = self._get_latent_spatial_position(
        #                 media_item_latents,
        #                 conditioning_item,
        #                 height,
        #                 width,
        #                 strip_latent_border=True,
        #             )
        #             b, c_l, f_l, h_l, w_l = media_item_latents.shape

        #             # First frame or sequence - just update the initial noise latents and the mask
        #             init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l] = (
        #                 torch.lerp(
        #                     init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l],
        #                     media_item_latents,
        #                     strength,
        #                 )
        #             )
        #             init_conditioning_mask[
        #                 :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l
        #             ] = strength
        #         else:
        #             # Non-first frame or sequence
        #             if n_frames > 1:
        #                 # Handle non-first sequence.
        #                 # Encoded latents are either fully consumed, or the prefix is handled separately below.
        #                 (
        #                     init_latents,
        #                     init_conditioning_mask,
        #                     media_item_latents,
        #                 ) = self._handle_non_first_conditioning_sequence(
        #                     init_latents,
        #                     init_conditioning_mask,
        #                     media_item_latents,
        #                     media_frame_number,
        #                     strength,
        #                 )

        #             # Single frame or sequence-prefix latents
        #             if media_item_latents is not None:
        #                 noise = randn_tensor(
        #                     media_item_latents.shape,
        #                     generator=generator,
        #                     device=media_item_latents.device,
        #                     dtype=media_item_latents.dtype,
        #                 )

        #                 media_item_latents = torch.lerp(
        #                     noise, media_item_latents, strength
        #                 )

        #                 # Patchify the extra conditioning latents and calculate their pixel coordinates
        #                 media_item_latents, latent_coords = self.patchifier.patchify(
        #                     latents=media_item_latents
        #                 )
        #                 pixel_coords = latent_to_pixel_coords(
        #                     latent_coords,
        #                     self.vae,
        #                     causal_fix=self.transformer.config.causal_temporal_positioning,
        #                 )

        #                 # Update the frame numbers to match the target frame number
        #                 pixel_coords[:, 0] += media_frame_number
        #                 extra_conditioning_num_latents += media_item_latents.shape[1]

        #                 conditioning_mask = torch.full(
        #                     media_item_latents.shape[:2],
        #                     strength,
        #                     dtype=torch.float32,
        #                     device=init_latents.device,
        #                 )

        #                 extra_conditioning_latents.append(media_item_latents)
        #                 extra_conditioning_pixel_coords.append(pixel_coords)
        #                 extra_conditioning_mask.append(conditioning_mask)

        # Patchify the updated latents and calculate their pixel coordinates
        init_latents, init_latent_coords = self.patchifier.patchify(
            latents=init_latents
        )
        init_pixel_coords = latent_to_pixel_coords(
            init_latent_coords,
            self.vae,
            # causal_fix=self.transformer.config.causal_temporal_positioning, set to false now
            causal_fix=True

        )

        if not conditioning_items:
            return init_latents, init_pixel_coords, None, 0

        # init_conditioning_mask, _ = self.patchifier.patchify(
        #     latents=init_conditioning_mask.unsqueeze(1)
        # )
        # init_conditioning_mask = init_conditioning_mask.squeeze(-1)

        # if extra_conditioning_latents:
        #     # Stack the extra conditioning latents, pixel coordinates and mask
        #     init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
        #     init_pixel_coords = torch.cat(
        #         [*extra_conditioning_pixel_coords, init_pixel_coords], dim=2
        #     )
        #     init_conditioning_mask = torch.cat(
        #         [*extra_conditioning_mask, init_conditioning_mask], dim=1
        #     )

        #     if self.transformer.use_tpu_flash_attention:
        #         # When flash attention is used, keep the original number of tokens by removing
        #         #   tokens from the end.
        #         init_latents = init_latents[:, :-extra_conditioning_num_latents]
        #         init_pixel_coords = init_pixel_coords[
        #             :, :, :-extra_conditioning_num_latents
        #         ]
        #         init_conditioning_mask = init_conditioning_mask[
        #             :, :-extra_conditioning_num_latents
        #         ]

        # return (
        #     init_latents,
        #     init_pixel_coords,
        #     init_conditioning_mask,
        #     extra_conditioning_num_latents,
        # )

    # change the paramters of these, currently pass in dummy inputs
    
    #copied from image_processor ?? change?? move to a seperate file???? 
        
    def denormalize(self, images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Denormalize an image array to [0,1].

        Args:
            images (`np.ndarray` or `torch.Tensor`):
                The image array to denormalize.

        Returns:
            `np.ndarray` or `torch.Tensor`:
                The denormalized image array.
        """
        return (images * 0.5 + 0.5).clamp(0, 1)
    
    def _denormalize_conditionally(
        self, images: torch.Tensor, do_denormalize: Optional[List[bool]] = None
    ) -> torch.Tensor:
        r"""
        Denormalize a batch of images based on a condition list.

        Args:
            images (`torch.Tensor`):
                The input image tensor.
            do_denormalize (`Optional[List[bool]`, *optional*, defaults to `None`):
                A list of booleans indicating whether to denormalize each image in the batch. If `None`, will use the
                value of `do_normalize` in the `VaeImageProcessor` config.
        """
        if do_denormalize is None:
            return self.denormalize(images) 

        return torch.stack(
            [self.denormalize(images[i]) if do_denormalize[i] else images[i] for i in range(images.shape[0])]
        )
    #same as diffusers image_processor.postprocess 
    
    def postprocess_to_output_type(self, image, output_type):  #support latent & pt
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )

        if output_type not in ["latent", "pt", "np", "pil"]:
            output_type = "np"

        if output_type == "latent":
            return image
        image = self._denormalize_conditionally(image, None) #do denormalize set to none
        if output_type == "pt":
            return image
        
    
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        negative_prompt: str = "",
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        frame_rate: int = 30,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_timesteps: Optional[List[int]] = None,
        decode_timestep: Union[List[float], float] = 0.05,
        decode_noise_scale: Optional[List[float]] = 0.025,
        offload_to_cpu: bool = False,
        enhance_prompt: bool = False,
        text_encoder_max_tokens: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: Union[float, List[float]] = 4.5,
        rescaling_scale: Union[float, List[float]] = 0.7,
        stg_scale: Union[float, List[float]] = 1.0,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        cfg_star_rescale: bool = False,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
        **kwargs,
    ):
        enhance_prompt = False
        prompt = self.config.prompt
        is_video = kwargs.get("is_video", False)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        vae_per_channel_normalize = kwargs.get(
            "vae_per_channel_normalize", True)
     
        
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor
        if isinstance(self.vae, TorchaxCausalVideoAutoencoder) and is_video:
            latent_num_frames += 1
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(
            base_dir, "../../models/ltx_video/xora_v1.2-13B-balanced-128.json")
        with open(config_path, "r") as f:
            model_config = json.load(f)

        latent_shape = (
            batch_size * num_images_per_prompt,
            model_config["in_channels"],
            latent_num_frames,
            latent_height,
            latent_width,
        )
        scheduler_state = self.retrieve_timesteps(self.scheduler, latent_shape, self.scheduler_state, num_inference_steps, None, skip_initial_inference_steps, skip_final_inference_steps)
        
        
        guidance_mapping = []

        if guidance_timesteps: 
            for timestep in scheduler_state.timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )
        
        if not isinstance(guidance_scale, list):
            guidance_scale = [guidance_scale] * len(scheduler_state.timesteps)
        else:
            guidance_scale = [guidance_scale[guidance_mapping[i]] for i in range(len(scheduler_state.timesteps))]
        
        if not isinstance(stg_scale, list):
            stg_scale = [stg_scale] * len(scheduler_state.timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(scheduler_state.timesteps))]
       
        if not isinstance(rescaling_scale, list):
            rescaling_scale = [rescaling_scale] * len(scheduler_state.timesteps)
        else:
            rescaling_scale = [rescaling_scale[guidance_mapping[i]] for i in range(len(scheduler_state.timesteps))]
       
        guidance_scale = [x if x > 1.0 else 0.0 for x in guidance_scale]
        do_classifier_free_guidance = any(x > 1.0 for x in guidance_scale)
        do_spatio_temporal_guidance = any(x > 0.0 for x in stg_scale)
        do_rescaling = any(x != 1.0 for x in rescaling_scale)

        
        num_conds = 1
        if do_classifier_free_guidance:
            num_conds += 1
        if do_spatio_temporal_guidance:
            num_conds += 1
      
        is_list_of_lists = bool(skip_block_list) and isinstance(skip_block_list[0], list)

        if not is_list_of_lists:
            skip_block_list = [skip_block_list] * len(scheduler_state.timesteps)
        else:
            new_skip_block_list = []
            for i in range(len(scheduler_state.timesteps)):
                new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                
            skip_block_list = new_skip_block_list
          
        if do_spatio_temporal_guidance:
            if skip_block_list is not None:
                skip_layer_masks = [
                    self.transformer.create_skip_layer_mask(
                        batch_size, num_conds, num_conds - 1, skip_blocks
                    )
                    for skip_blocks in skip_block_list
                ]
        if enhance_prompt:
            prompt = generate_cinematic_prompt(
                self.prompt_enhancer_image_caption_model,
                self.prompt_enhancer_image_caption_processor,
                self.prompt_enhancer_llm_model,
                self.prompt_enhancer_llm_tokenizer,
                prompt,
                None,  # conditioning items set to None
                max_new_tokens=text_encoder_max_tokens,
            )

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,  
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=None,  # device set to none
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )
        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        if do_classifier_free_guidance:
            prompt_embeds_batch = jnp.concatenate(
                [negative_prompt_embeds, prompt_embeds], axis=0 #check negative_prompt_embeds dimension
            )
            prompt_attention_mask_batch = jnp.concatenate(
                [negative_prompt_attention_mask, prompt_attention_mask], axis=0
            )
        if do_spatio_temporal_guidance:
            prompt_embeds_batch = jnp.concatenate([prompt_embeds_batch, prompt_embeds], axis=0)
            prompt_attention_mask_batch = jnp.concatenate(
                [
                    prompt_attention_mask_batch,
                    prompt_attention_mask,
                ],
                axis=0,
            )
        latents = self.prepare_latents(
            latents=latents,
            media_items=None,  # set to None
            timestep=scheduler_state.timesteps[0],  # set to 1.0 for now TODO: fix this
            latent_shape=latent_shape,
            dtype=None,
            device=None,
            generator=generator,
            vae_per_channel_normalize=vae_per_channel_normalize,
        )

        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_conditioning(
                conditioning_items=None,
                init_latents=latents,
                num_frames=num_frames,
                height=height,
                width=width,
                vae_per_channel_normalize=vae_per_channel_normalize,
                generator=generator,
            )
        )

        extra_step_kwargs = prepare_extra_step_kwargs(
            generator=jax.random.PRNGKey(0))

        pixel_coords = torch.cat([pixel_coords] * num_conds)
        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

        # if not isinstance(guidance_scale, List):
        #   guidance_scale = [guidance_scale] * len(self.scheduler_state.timesteps)
        # guidance_scale = [x if x > 1.0 else 0.0 for x in guidance_scale]
        # data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))
        # latents = jax.device_put(example_inputs["latents"], data_sharding)
        # prompt_embeds = jax.device_put(example_inputs["prompt_embeds"], data_sharding)
        # fractional_coords = jax.device_put(example_inputs["fractional_coords"], data_sharding)
        # noise_cond = jax.device_put(example_inputs["timestep"], data_sharding)
        # segment_ids = jax.device_put(example_inputs["segment_ids"], data_sharding)
        # encoder_attention_segment_ids = jax.device_put(example_inputs["encoder_attention_segment_ids"], data_sharding)
        # validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids)
        noise_cond = jnp.ones(  # initialize first round with this!
            (4, 1)
        )

        # # noise_cond = None
        # saved_tensor_path = "/home/serenagu_google_com/LTX-Video/ltx_video/pipelines/schedulerTest1.0"
        # tensor_dict = torch.load(saved_tensor_path)

        # for key, value in tensor_dict.items():
        #   if value is not None:
        #     tensor_dict[key] = jnp.array(value.to(torch.float32).cpu().numpy())
        # example_inputs = tensor_dict
        # latents = jax.device_put(example_inputs["latent_model_input"])
        # # prompt_embeds = jax.device_put(example_inputs["encoder_hidden_states"])
        # fractional_coords = jax.device_put(example_inputs["indices_grid"])
        # encoder_attention_segment_ids = jax.device_put(example_inputs["encoder_attention_segment_ids"])
        # segment_ids = None
        # # validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids)

        # #only run this for the first time!
        # scheduler_state = self.scheduler.set_timesteps(state=self.scheduler_state, shape=latents.shape, num_inference_steps=num_inference_steps)
        # extra_step_kwargs = prepare_extra_step_kwargs(generator = jax.random.PRNGKey(0)) #check if this value needs to be changed, for unipc eta is not taken
        # scheduler_state = self.scheduler_state
        # num_warmup_steps = max(len(self.scheduler_state.timesteps) - num_inference_steps * self.scheduler.order, 0) #no paramter order here
        # p_run_inference = jax.jit(
        #   functools.partial(
        #       run_inference,
        #       transformer=self.transformer,
        #       config=self.config,
        #       mesh=self.mesh,
        #       fractional_cords=fractional_coords,
        #       prompt_embeds = prompt_embeds,
        #       segment_ids=segment_ids,

        #       encoder_attention_segment_ids=encoder_attention_segment_ids,
        #       num_inference_steps=num_inference_steps,
        #       scheduler=self.scheduler,
        #   ),
        #   in_shardings=(self.state_shardings, data_sharding, data_sharding, None),   #not sure if this sharding is correct
        #   out_shardings=None,
        # )
        segment_ids = None
        # num_warmup_steps = max(len(self.scheduler_state.timesteps) - num_inference_steps * self.scheduler.order, 0) #no paramter order here
        # p_run_inference = functools.partial(
        #       run_inference,
        #       transformer=self.transformer,
        #       config=self.config,
        #       mesh=self.mesh,
        #       fractional_cords=fractional_coords,
        #       prompt_embeds = prompt_embeds,
        #       segment_ids=segment_ids,
        #       encoder_attention_segment_ids=encoder_attention_segment_ids,
        #       num_inference_steps=num_inference_steps,
        #       scheduler=self.scheduler,
        #       # guidance_scale=guidance_scale
        #   )
        p_run_inference = functools.partial(
            run_inference,
            transformer=self.transformer,
            config=self.config,
            mesh=self.mesh,
            fractional_cords=jnp.array(
                fractional_coords.to(torch.float32).detach().numpy()),
            prompt_embeds=prompt_embeds_batch,
            segment_ids=None,
            encoder_attention_segment_ids=prompt_attention_mask_batch,
            num_inference_steps=num_inference_steps,
            scheduler=self.scheduler,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_conds=num_conds,
            guidance_scale=guidance_scale,
            do_spatio_temporal_guidance=do_spatio_temporal_guidance,
            stg_scale=stg_scale,
            do_rescaling = do_rescaling,
            rescaling_scale = rescaling_scale,
            batch_size=batch_size,
            skip_layer_masks = skip_layer_masks,
            skip_layer_strategy = skip_layer_strategy,
            cfg_star_rescale = cfg_star_rescale
        )

        with self.mesh:
            latents, scheduler_state = p_run_inference(transformer_state=self.transformer_state, latents=jnp.array(latents.to(
                # add scheduler state back in
                torch.float32).detach().numpy()), timestep=noise_cond, scheduler_state=scheduler_state)
        # latents = torch.from_numpy(np.array(latents))
        latents = latents[:, num_cond_latents:]

        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=model_config["in_channels"]
            // math.prod(self.patchifier.patch_size),
        )
        
        if output_type != "latent":
            if self.vae.decoder.timestep_conditioning:
                noise = jax.random.normal(jax.random.PRNGKey(5), latents.shape, dtype=latents.dtype)  #move the key to outer layer

                # Convert decode_timestep to a list if it's not already one
                if not isinstance(decode_timestep, (list, jnp.ndarray)):
                    decode_timestep = [decode_timestep] * latents.shape[0]

                # Handle decode_noise_scale
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, (list, jnp.ndarray)):
                    decode_noise_scale = [decode_noise_scale] * latents.shape[0]

                # Convert lists to JAX arrays
                decode_timestep = jnp.array(decode_timestep, dtype=jnp.float32)
                
                # Reshape decode_noise_scale for broadcasting
                decode_noise_scale = jnp.array(decode_noise_scale, dtype=jnp.float32)
                decode_noise_scale = jnp.reshape(decode_noise_scale, (latents.shape[0],) + (1,) * (latents.ndim - 1))

                # Apply the noise and scale
                latents = (
                    latents * (1 - decode_noise_scale) +
                    noise * decode_noise_scale
                )
            else:
                decode_timestep = None
            image = self.vae.decode(
                latents = jax.device_put(latents, jax.devices('tpu')[0]), #.astype(jnp.bfloat16), #jax.device_put(latents, jax.devices('cpu')[0]),
                is_video = is_video,
                vae_per_channel_normalize=kwargs.get(
                    "vae_per_channel_normalize", True),
                timestep=decode_timestep #.astype(jnp.bfloat16),
            )
            image = self.postprocess_to_output_type(    #swap this out!
                torch.from_numpy(np.asarray(image.astype(jnp.float16))), output_type=output_type) 

        else:
            image = latents

        # Offload all models

        if not return_dict:
            return (image,)

        return image
    # save states here


def transformer_forward_pass(  # need to jit this? wan didnt
    latents,
    state,
    noise_cond,
    transformer,
    fractional_cords,
    prompt_embeds,
    segment_ids,
    encoder_attention_segment_ids,
    skip_layer_mask,
    skip_layer_strategy,
):

    noise_pred = transformer.apply(
        {"params": state.params},
        hidden_states=latents,
        indices_grid=fractional_cords,
        encoder_hidden_states=prompt_embeds,
        timestep=noise_cond,
        segment_ids=segment_ids,
        encoder_attention_segment_ids=encoder_attention_segment_ids,
        skip_layer_mask=skip_layer_mask,
        skip_layer_strategy=skip_layer_strategy
    )  # need .param here?
    return noise_pred, state


def run_inference(
    transformer_state, transformer, config, mesh, latents, fractional_cords, prompt_embeds, timestep, num_inference_steps, scheduler, segment_ids, encoder_attention_segment_ids, scheduler_state, do_classifier_free_guidance, num_conds, guidance_scale, do_spatio_temporal_guidance, stg_scale, do_rescaling, rescaling_scale, batch_size, skip_layer_masks, skip_layer_strategy, cfg_star_rescale
):
    # do_classifier_free_guidance = guidance_scale > 1.0
    # for step in range(num_inference_steps):
    for i, t in enumerate(scheduler_state.timesteps):
        current_timestep = t
        latent_model_input = (
            jnp.concatenate([latents] * num_conds) if num_conds > 1 else latents
        )
        # t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        # # timestep = jnp.broadcast_to(t, timestep.shape)  # (4, 256)
        # timestep = jnp.broadcast_to(
        #     t, (latent_model_input.shape[0],)
        # ).reshape(-1, 1)
        if not isinstance(current_timestep, (jnp.ndarray, jax.Array)):
        # Determine the correct dtype based on the device (similar to PyTorch)
            is_mps = False  # MPS is not a JAX concept, remove it. JAX handles devices automatically.
            if isinstance(current_timestep, float):
                dtype = jnp.float32 
            else:
                dtype = jnp.int32
        
            current_timestep = jnp.array(
                [current_timestep],
                dtype=dtype,
            )
        elif current_timestep.ndim == 0:
            current_timestep = jnp.expand_dims(current_timestep, axis=0)

    # Broadcast to batch dimension (compatible with ONNX/Core ML)
        current_timestep = jnp.broadcast_to(
            current_timestep, (latent_model_input.shape[0],1)
        )

        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules): #error out with this line
            noise_pred, transformer_state = transformer_forward_pass(
                latent_model_input, transformer_state, current_timestep, transformer, fractional_cords, prompt_embeds, segment_ids, encoder_attention_segment_ids, skip_layer_mask=(
                                skip_layer_masks[i]
                                if skip_layer_masks is not None
                                else None
                            ), skip_layer_strategy = skip_layer_strategy)
        # ValueError: One of pjit outputs with pytree key path result was given the sharding of NamedSharding(mesh=Mesh('data': 4, 'fsdp': 1, 'tensor': 1, 'fsdp_transpose': 1, 'expert': 1, 'tensor_transpose': 1, 'tensor_sequence': 1, 'sequence': 1, axis_types=(Auto, Auto, Auto, Auto, Auto, Auto, Auto, Auto)), spec=PartitionSpec(('data', 'fsdp'), None, None), memory_kind=device), which implies that the global size of its dimension 0 should be divisible by 4, but it is equal to 1 (full shape: (1, 1, 128))

        # # latents = self.denoising
        #   latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        # with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        # noise_pred, transformer_state = transformer_forward_pass(latents, transformer_state, timestep, transformer, fractional_cords, prompt_embeds, segment_ids, encoder_attention_segment_ids) #need to check if transformer_state is successfully updated
        if do_spatio_temporal_guidance:
        # JAX uses jnp.split for splitting along an axis.
        # Equivalent to .chunk() for equal-sized chunks
            chunks = jnp.split(noise_pred, num_conds, axis=0)
            noise_pred_text = chunks[-2]
            noise_pred_text_perturb = chunks[-1]

        if do_classifier_free_guidance:
            
            chunks = jnp.split(noise_pred, num_conds, axis=0)
            noise_pred_uncond = chunks[0]
            noise_pred_text = chunks[1]
            if cfg_star_rescale:
                positive_flat = noise_pred_text.reshape(batch_size, -1)
                negative_flat = noise_pred_uncond.reshape(batch_size, -1)
                dot_product = jnp.sum(   #(1, 1)
                    positive_flat * negative_flat, axis=1, keepdims=True
                )
                squared_norm = ( #(1, 1)
                    jnp.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
                )
                alpha = dot_product / squared_norm  #might need to reshape this (1, 1)
                alpha = alpha.reshape(batch_size, 1, 1)

                noise_pred_uncond = alpha * noise_pred_uncond   #error here (1, 3072, 128)
            noise_pred = noise_pred_uncond + guidance_scale[i] * (
                noise_pred_text - noise_pred_uncond
            )
        elif do_spatio_temporal_guidance:
            noise_pred = noise_pred_text

        if do_spatio_temporal_guidance:
            noise_pred = noise_pred + stg_scale[i] * (
                noise_pred_text - noise_pred_text_perturb
            )
            if do_rescaling and stg_scale[i] > 0.0: 
                noise_pred_text_std = jnp.std(noise_pred_text.reshape(batch_size, -1), axis=1, keepdims=True)
                noise_pred_std = jnp.std(noise_pred.reshape(batch_size, -1), axis=1, keepdims=True)

                factor = noise_pred_text_std / noise_pred_std
                factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

                
                noise_pred = noise_pred * factor.reshape(batch_size, 1, 1) 
        current_timestep = current_timestep[:1] # JAX slicing is similar
        latents, scheduler_state = scheduler.step(
            scheduler_state, noise_pred, current_timestep[0][0], latents).to_tuple()

    return latents, scheduler_state

def adain_filter_latent(
    latents: torch.Tensor, reference_latents: torch.Tensor, factor=1.0
):
    """
    Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on
    statistics from a reference latent tensor.

    Args:
        latent (torch.Tensor): Input latents to normalize
        reference_latent (torch.Tensor): The reference latents providing style statistics.
        factor (float): Blending factor between original and transformed latent.
                       Range: -10.0 to 10.0, Default: 1.0

    Returns:
        torch.Tensor: The transformed latent tensor
    """
    with default_env():
        result = latents.clone()

        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(
                    reference_latents[i, c], dim=None
                )  # index by original dim order
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

        result = torch.lerp(latents, result, factor)
    return result

class LTXMultiScalePipeline:  ##figure these methods out
    
    @classmethod
    def load_latent_upsampler(cls, config):
        spatial_upscaler_model_name_or_path = config.spatial_upscaler_model_path

        if spatial_upscaler_model_name_or_path and not os.path.isfile(spatial_upscaler_model_name_or_path):
            spatial_upscaler_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=spatial_upscaler_model_name_or_path,
                local_dir=config.models_dir,
                repo_type="model",
            )
        else:
            spatial_upscaler_model_path = spatial_upscaler_model_name_or_path
        if not config.spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
            )
        latent_upsampler = LatentUpsampler.from_pretrained(spatial_upscaler_model_path)
        latent_upsampler.eval()
        return latent_upsampler
    
    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        latents = jax.device_put(latents, jax.devices('tpu')[0])
        #assert latents.device == latest_upsampler.device
        with default_env():
            latents = un_normalize_latents(  #need to switch this out?
                interop.torch_view(latents), self.vae, vae_per_channel_normalize=True
            )
            upsampled_latents = latest_upsampler(torch.from_numpy(np.array(latents))) #here converted back to torch, cause upsampler in pytorch
            upsampled_latents = normalize_latents(
                interop.torch_view(jnp.array(upsampled_latents.detach().numpy())), self.vae, vae_per_channel_normalize=True
            )
        return upsampled_latents

    def __init__(
        self, video_pipeline: LTXVideoPipeline
    ):
        self.video_pipeline = video_pipeline
        self.vae = video_pipeline.vae

    def __call__(
        self,
        height,
        width,
        num_frames, 
        output_type, 
        generator, 
        config
    ) -> Any:
        original_output_type = output_type
        original_width = width
        original_height = height
        x_width = int(width * config.downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(height * config.downscale_factor)
        downscaled_height = x_height - (x_height % self.video_pipeline.vae_scale_factor)
        # #use original height and width here to see
        output_type = 'latent'
        result = self.video_pipeline(height=original_height, width=original_width, num_frames=num_frames,
        is_video=True, output_type=output_type, generator=generator, guidance_scale = config.first_pass["guidance_scale"], stg_scale = config.first_pass["stg_scale"], rescaling_scale = config.first_pass["rescaling_scale"], skip_initial_inference_steps= config.first_pass["skip_initial_inference_steps"], skip_final_inference_steps= config.first_pass["skip_final_inference_steps"], 
        num_inference_steps = config.first_pass["num_inference_steps"], guidance_timesteps = config.first_pass["guidance_timesteps"], cfg_star_rescale = config.first_pass["cfg_star_rescale"], skip_layer_strategy = None, skip_block_list=config.first_pass["skip_block_list"])
        latents = result
        print("done")
        latent_upsampler = self.load_latent_upsampler(config)
        upsampled_latents = self._upsample_latents(latent_upsampler, latents) #convert back to pytorch here
        ##maybe change this?
        latents = torch.from_numpy(np.array(latents)) #.to(torch.device('cpu'))
        upsampled_latents = torch.from_numpy(np.array(upsampled_latents)) #.to(torch.device('cpu'))
        upsampled_latents = adain_filter_latent(
            latents=upsampled_latents, reference_latents=latents
        )
        latents = upsampled_latents
        output_type = original_output_type
        width = downscaled_width * 2
        height = downscaled_height * 2
        # import pdb; pdb.set_trace()
        result = self.video_pipeline(height=original_height*2, width=original_width*2, num_frames=num_frames,
        is_video=True, output_type=original_output_type, latents = latents, generator=generator, guidance_scale = config.second_pass["guidance_scale"], stg_scale = config.second_pass["stg_scale"], rescaling_scale = config.second_pass["rescaling_scale"], skip_initial_inference_steps= config.second_pass["skip_initial_inference_steps"], skip_final_inference_steps= config.second_pass["skip_final_inference_steps"], 
        num_inference_steps = config.second_pass["num_inference_steps"], guidance_timesteps = config.second_pass["guidance_timesteps"], cfg_star_rescale = config.second_pass["cfg_star_rescale"], skip_layer_strategy = None, skip_block_list=config.second_pass["skip_block_list"])

        if original_output_type != "latent":
            num_frames = result.shape[2]
            videos = rearrange(result, "b c f h w -> (b f) c h w")

            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
            result = videos

        return result