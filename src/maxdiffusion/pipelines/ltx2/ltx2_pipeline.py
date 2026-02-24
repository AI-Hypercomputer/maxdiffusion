# Copyright 2025 Google LLC
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

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import torch  # For tokenizer and some utils if needed

from ...pyconfig import HyperParameters
from ... import max_logging
from ... import max_utils
from ...models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from ...models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from ...models.ltx2.text_encoders.text_encoders_ltx2 import LTX2VideoGemmaTextEncoder
from ...schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler, FlowMatchSchedulerState
from maxdiffusion.image_processor import PipelineImageInput
from transformers import AutoTokenizer

import sys
import os


# Import MaxText
try:
  import maxtext.models.gemma3 as gemma3
  import maxtext.models.models as max_models
  from maxtext.layers import embeddings, normalizations
  from maxtext import common_types
except ImportError:
   max_logging.log("maxtext not found. Please install MaxText.")
   gemma3 = None
   max_models = None
   embeddings = None
   normalizations = None
   common_types = None

class MaxTextGemma3FeatureExtractor(nnx.Module):
    """
    Wrapper around MaxText Gemma3 components to return all hidden states.
    Mimics MaxText.models.models.Transformer but optimized for feature extraction.
    """
    def __init__(self, config, mesh, quant=None, rngs=None):
        self.config = config
        self.mesh = mesh
        self.quant = quant
        
        # Embeddings
        self.token_embedder = embeddings.Embed(
            mesh=mesh,
            num_embeddings=config.vocab_size,
            num_features=config.emb_dim,
            dtype=config.dtype,
            embedding_init=nnx.initializers.normal(stddev=1.0),
            config=config,
            rngs=rngs
        )
        
        # Layers
        self.layers = []
        for i in range(config.num_decoder_layers):
            layer = gemma3.Gemma3DecoderLayer(
                config=config,
                mesh=mesh,
                model_mode=common_types.MODEL_MODE_PREFILL, # Default to prefill/inference
                rngs=rngs,
                quant=quant,
                attention_type=gemma3.get_attention_type(i)
            )
            self.layers.append(layer)
            
        # Final Norm
        self.norm = normalizations.rms_norm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            epsilon=config.normalization_layer_epsilon,
            kernel_axes=("norm",),
            name="decoder_norm"
        )

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L) - Optional, used for bidirectional mask generation if needed.
        Returns:
            Mock object with .hidden_states tuple if output_hidden_states=True
        """
        # Embed
        x = self.token_embedder(input_ids.astype("int32"), model_mode=common_types.MODEL_MODE_PREFILL)
        
        # Scaling if needed (Gemma usually scales embeddings? Check MaxText implementation)
        # MaxText Embed layer usually handles scaling if configured? 
        # Checking models.py: self.token_embedder(...) -> returns embeddings.
        # Gemma3 uses 'query_pre_attn_scalar' inside attention, not embedding scaling usually.
        # But wait, MaxText models.py _apply_embedding:
        # y = y.astype(cfg.dtype)
        # + positional embeddings if any.
        
        # Basic loop
        hidden_states = []
        if output_hidden_states:
            hidden_states.append(x)
            
        # Create dummy positions if needed or handle them
        # MaxText layers usually take decoder_positions
        batch, seq_len = input_ids.shape
        positions = jnp.arange(seq_len)[None, :]
        positions = jnp.broadcast_to(positions, (batch, seq_len))
        
        # Scan over layers
        for layer in self.layers:
            # We assume non-scanned for flexibility in feature extraction for now,
            # or we could scan if we pack the loop.
            # Calling layer:
            x, _ = layer(
                inputs=x,
                decoder_segment_ids=None,
                decoder_positions=positions,
                deterministic=True,
                model_mode=common_types.MODEL_MODE_PREFILL
            )
            if output_hidden_states:
                hidden_states.append(x)
                
        # Final Norm
        x = self.norm(x)
        if output_hidden_states:
            hidden_states.append(x)
            
        class Output:
            pass
        out = Output()
        out.hidden_states = tuple(hidden_states)
        out.last_hidden_state = x
        return out

class LTX2Pipeline:
  """
  Pipeline for LTX-2 Image-to-Video generation.
  """

  def __init__(
      self,
      tokenizer: AutoTokenizer,
      text_encoder: Any, # Gemma3 model
      text_encoder_connector: LTX2VideoGemmaTextEncoder,
      transformer: LTX2VideoTransformer3DModel,
      vae: LTX2VideoAutoencoderKL,
      scheduler: FlaxFlowMatchScheduler,
      scheduler_state: FlowMatchSchedulerState,
      devices_array: np.array,
      mesh: Mesh,
      config: HyperParameters,
      audio_vae: Optional[Any] = None, # Placeholder for Audio VAE
      vocoder: Optional[Any] = None, # Placeholder for Vocoder
  ):
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    self.text_encoder_connector = text_encoder_connector
    self.transformer = transformer
    self.vae = vae
    self.scheduler = scheduler
    self.scheduler_state = scheduler_state
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.audio_vae = audio_vae
    self.vocoder = vocoder

    self.vae_scale_factor_temporal = 8 # Default for LTX2
    self.vae_scale_factor_spatial = 32 # Default for LTX2

  def _encode_image(self, image, device, dtype, generator):
    if not isinstance(image, (jnp.ndarray, np.ndarray)):
      raise ValueError("`image` must be a jnp.ndarray or np.ndarray")
    
    # Normalize if needed (-1, 1) usually
    if image.min() >= 0:
        image = 2.0 * image - 1.0
    
    # Check shape: (B, H, W, C) -> (B, F, H, W, C)
    if image.ndim == 4:
        image = jnp.expand_dims(image, axis=1) # Add frame dim
    
    # VAE encode
    posterior = self.vae.encode(image).latent_dist
    latents = posterior.sample(generator)
    
    # Normalize and Scale
    latents = (latents - self.vae.latents_mean) * self.vae.config.scaling_factor / self.vae.latents_std
    return latents

  def prepare_latents(
      self,
      batch_size: int,
      height: int,
      width: int,
      num_frames: int,
      num_channels_latents: int,
      dtype: jnp.dtype,
      rng: jax.Array,
      image=None,
      timestep=None,
      noise=None,
  ):
    # VAE spatial compression is 32x32, temporal is 8
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    
    shape = (
        batch_size,
        num_latent_frames,
        height // self.vae_scale_factor_spatial,
        width // self.vae_scale_factor_spatial,
        num_channels_latents,
    )
    
    # Init latents with noise
    if noise is None:
        if rng is None:
             raise ValueError("RNG must be provided if noise is None")
        latents = jax.random.normal(rng, shape=shape, dtype=dtype)
    else:
        latents = noise
        
    conditioning_mask = jnp.zeros((batch_size, num_latent_frames, shape[2], shape[3]), dtype=dtype)
    image_latents = None

    if image is not None:
        # Encode image
        image_latents = self._encode_image(image, None, dtype, rng)
        
        # In I2V, we typically condition on the first frame(s).
        # We need to ensure image_latents matches the target shape in H/W if not already resized.
        # Assuming input image was pre-resized for now.
        
        num_image_frames = image_latents.shape[1]
        
        # Create a mask that is 1.0 where we have conditioning image
        # And 0.0 where we want to generate
        # For LTX-2 I2V, usually the first frame is conditioned.
        
        # We replace the initial noise with the noisy image latents during the loop, 
        # but here we just return the clean image latents for the scheduler to use.
        
        # Just to be safe, we only mark valid frames
        valid_frames = min(num_image_frames, num_latent_frames)
        conditioning_mask = conditioning_mask.at[:, :valid_frames, ...].set(1.0)
        
        # We don't replace latents here yet; we do it in the loop (Noisy Replacement)
        # OR we can initialize latents with image_latents if t=T (all noise) -> actually no, t=T is random noise.
    
    return latents, conditioning_mask, image_latents

  def retrieve_timesteps(
      self,
      scheduler,
      num_inference_steps: Optional[int] = None,
      timesteps: Optional[List[int]] = None,
      sigmas: Optional[List[float]] = None,
      **kwargs,
  ):
      self.scheduler_state = scheduler.set_timesteps(
        self.scheduler_state,
        num_inference_steps=num_inference_steps,
      )
      return self.scheduler_state.timesteps

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 128,
      device: Any = None,
      dtype: jnp.dtype = jnp.float32,
      do_classifier_free_guidance: bool = False,
      negative_prompt: Optional[Union[str, List[str]]] = None,
  ):
    if self.text_encoder is None:
        raise ValueError("Text encoder is not initialized.")

    if isinstance(prompt, str):
        prompt = [prompt]

    if self.tokenizer.padding_side != "left":
         self.tokenizer.padding_side = "left"
    
    # 1. Encode Positive Prompt
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = jnp.array(text_inputs.input_ids)
    attention_mask = jnp.array(text_inputs.attention_mask)
    
    outputs = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs
    
    prompt_embeds = self.text_encoder_connector(
        hidden_states=hidden_states,
        attention_mask=attention_mask
    )
    
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = jnp.repeat(prompt_embeds, num_videos_per_prompt, axis=0)
    attention_mask = jnp.repeat(attention_mask, num_videos_per_prompt, axis=0)
    
    # 2. Encode Negative Prompt if CFG is enabled
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
            
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="np"
        )
        uncond_input_ids = jnp.array(uncond_inputs.input_ids)
        uncond_attention_mask = jnp.array(uncond_inputs.attention_mask)
        
        uncond_outputs = self.text_encoder(
            input_ids=uncond_input_ids,
            attention_mask=uncond_attention_mask,
            output_hidden_states=True
        )
        uncond_hidden_states = uncond_outputs.hidden_states if hasattr(uncond_outputs, 'hidden_states') else uncond_outputs
        
        negative_prompt_embeds = self.text_encoder_connector(
            hidden_states=uncond_hidden_states,
            attention_mask=uncond_attention_mask
        )
        
        negative_prompt_embeds = jnp.repeat(negative_prompt_embeds, num_videos_per_prompt, axis=0)
        uncond_attention_mask = jnp.repeat(uncond_attention_mask, num_videos_per_prompt, axis=0)
        
        # Concatenate for CFG
        prompt_embeds = jnp.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)
        attention_mask = jnp.concatenate([uncond_attention_mask, attention_mask], axis=0)

    return prompt_embeds, attention_mask

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      image: PipelineImageInput = None,
      height: int = 480,
      width: int = 704,
      num_frames: int = 121,
      num_inference_steps: int = 50,
      guidance_scale: float = 3.0,
      num_videos_per_prompt: int = 1,
      generator: Optional[jax.Array] = None,
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      prompt_attention_mask: Optional[jax.Array] = None,
      negative_prompt: Optional[Union[str, List[str]]] = None,
      output_type: str = "pil",
      return_dict: bool = True,
      **kwargs,
  ):
    # 1. Inputs
    if prompt is None and prompt_embeds is None:
        raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0] // num_videos_per_prompt

    do_classifier_free_guidance = guidance_scale > 1.0

    # 2. Encode Prompt
    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )
    
    # 3. Timesteps
    timesteps = self.retrieve_timesteps(self.scheduler, num_inference_steps)
    
    # 4. Prepare Latents
    num_channels_latents = self.transformer.in_channels
    if latents is None:
        if generator is None:
             generator = jax.random.PRNGKey(0)
        latents, conditioning_mask, image_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            height,
            width,
            num_frames,
            num_channels_latents,
            jnp.float32,
            generator,
            image=image
        )
    else:
        conditioning_mask = jnp.zeros((latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]), dtype=latents.dtype)
        image_latents = None

    # 5. Denoising Loop
    for i, t in enumerate(timesteps):
        # I2V Conditioning: Add noise to image_latents and replace in latents
        if image_latents is not None:
            # We need to noise the image_latents to the current timestep t
            # Assume scheduler has add_noise. If not, we need a manual fallback or update scheduler.
            # Using FlaxFlowMatchScheduler convention.
            # noise = jax.random.normal(generator, image_latents.shape) # Re-using generator might be risky if not split?
            # Ideally we split generator every step or use a deterministic noise for I2V if consistent.
            # For simplicity, we generate new noise for mixing.
            
            generator, noise_rng = jax.random.split(generator)
            noise_i2v = jax.random.normal(noise_rng, image_latents.shape, dtype=image_latents.dtype)
            
            # Broadcast t for valid frames
            # t is scalar usually
            # image_latents_noisy = self.scheduler.add_noise(image_latents, noise_i2v, t)
            
            # Manual add_noise if scheduler.add_noise expects array t
            t_array = jnp.broadcast_to(t, (image_latents.shape[0],))
            image_latents_noisy = self.scheduler.add_noise(
                self.scheduler_state,
                image_latents,
                noise_i2v,
                t_array
            )
            
            # Replace latents with noisy image latents where mask is 1
            # conditioning_mask: (B, F, H, W) -> expand to (B, F, H, W, 1) or (B, F, H, W, C)
            mask = conditioning_mask[..., None]
            latents = latents * (1 - mask) + image_latents_noisy * mask

        # CFG: Expand latents if needed
        latent_model_input = jnp.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        t_batch = jnp.broadcast_to(t, (latent_model_input.shape[0],))
        
        # Flatten Latents: (B, F, H, W, C) -> (B, S, C)
        B_in, F, H, W, C = latent_model_input.shape
        latents_flat = latent_model_input.reshape(B_in, F * H * W, C)
        
        # Audio Placeholders (Correct dimensionality)
        # Transformer expects audio_hidden_states: (B, S_a, D_a)
        audio_hidden_states = jnp.zeros((B_in, 1, 128), dtype=latents.dtype)
        audio_encoder_hidden_states = jnp.zeros((B_in, 1, 128), dtype=latents.dtype)
        audio_encoder_attention_mask = jnp.zeros((B_in, 1), dtype=jnp.int32)
        
        # Timestep Embeddings
        temb, _ = self.transformer.time_embed(t_batch)
        temb_audio, _ = self.transformer.audio_time_embed(t_batch)
        temb_ca_scale_shift, _ = self.transformer.av_cross_attn_video_scale_shift(t_batch)
        temb_ca_audio_scale_shift, _ = self.transformer.av_cross_attn_audio_scale_shift(t_batch)
        temb_ca_gate, _ = self.transformer.av_cross_attn_video_a2v_gate(t_batch)
        temb_ca_audio_gate, _ = self.transformer.av_cross_attn_audio_v2a_gate(t_batch)

        # Transformer Call
        noise_pred_flat, _ = self.transformer(
            hidden_states=latents_flat,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=prompt_embeds,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            temb=temb,
            temb_audio=temb_audio,
            temb_ca_scale_shift=temb_ca_scale_shift,
            temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
            temb_ca_gate=temb_ca_gate,
            temb_ca_audio_gate=temb_ca_audio_gate,
            encoder_attention_mask=prompt_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        
        # Unflatten Output: (B, S, C) -> (B, F, H, W, C)
        noise_pred = noise_pred_flat.reshape(B_in, F, H, W, C)
        
        # CFG Guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler Step
        scheduler_output = self.scheduler.step(
            self.scheduler_state,
            noise_pred,
            t,
            latents,
            return_dict=True
        )
        latents = scheduler_output.prev_sample
        self.scheduler_state = scheduler_output.state
    
    # 6. Un-normalize and Decode
    latents = (latents / self.vae.config.scaling_factor) * self.vae.latents_std + self.vae.latents_mean
    video = self.vae.decode(latents).sample
    
    return video
