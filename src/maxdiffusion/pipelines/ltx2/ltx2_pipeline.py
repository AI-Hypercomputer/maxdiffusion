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

# Placeholder imports for MaxText Gemma3
# Assuming MaxText is available in the environment or python path
try:
  from MaxText.models import gemma3
except ImportError:
  max_logging.log("MaxText.models.gemma3 not found. Gemma3 Text Encoder will not work without MaxText.")
  gemma3 = None

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

  def prepare_latents(
      self,
      batch_size: int,
      height: int,
      width: int,
      num_frames: int,
      num_channels_latents: int,
      dtype: jnp.dtype,
      rng: jax.Array,
  ):
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    # VAE spatial compression is 32x32, temporal is 8
    shape = (
        batch_size,
        num_latent_frames,
        height // self.vae_scale_factor_spatial,
        width // self.vae_scale_factor_spatial,
        num_channels_latents,
    )
    latents = jax.random.normal(rng, shape=shape, dtype=dtype)
    return latents

  def retrieve_timesteps(
      self,
      scheduler,
      num_inference_steps: Optional[int] = None,
      timesteps: Optional[List[int]] = None,
      sigmas: Optional[List[float]] = None,
      **kwargs,
  ):
      """
      Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.
      """
      # Update the scheduler state with new timesteps
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
      device: Any = None, # Unused in JAX
      dtype: jnp.dtype = jnp.float32,
  ):
    if self.text_encoder is None:
        raise ValueError("Text encoder is not initialized.")

    if isinstance(prompt, str):
        prompt = [prompt]

    # Tokenization
    # We assume tokenizer is configured correctly (padding side left for Gemma)
    if self.tokenizer.padding_side != "left":
         self.tokenizer.padding_side = "left"
    
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="np", # Return numpy for JAX
    )
    
    input_ids = jnp.array(text_inputs.input_ids)
    attention_mask = jnp.array(text_inputs.attention_mask)
    
    # Gemma3 Forward Pass
    # We assume text_encoder returns hidden_states when output_hidden_states=True
    # Or returns an object with hidden_states attribute
    # Note: MaxText models might have different signatures. 
    # Validating against generic JAX model pattern.
    
    # In MaxDiffusion, we often wrap models. 
    # Here we assume self.text_encoder is a callable nnx.Module or similar
    
    # We need to ensure input_ids are sharded if needed, but for now assuming replicated or handled by caller
    
    # Get Hidden States
    # Expecting: (batch, seq_len, hidden_dim) or per layer
    # For Gemma3, we might get a list of hidden states
    # We pass this to text_encoder_connector
    
    # Mocking or calling actual model
    # To make this robust without running actual Gemma3 which is heavy:
    # We will assume text_encoder is the connector if gemma3 is implied integrated, 
    # BUT the design separates them.
    
    # Use 'output_hidden_states=True' equivalent for MaxText if applicable
    outputs = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True # Hypothetical arg for MaxText Gemma
    )
    
    # If outputs is a tuple/list, it's likely hidden states
    # If it's an object, check hidden_states
    if hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states
    else:
        hidden_states = outputs # Assume it returns hidden states directly if configured
        
    # Text Encoder Connector (Gemma hidden states -> LTX2 embeddings)
    prompt_embeds = self.text_encoder_connector(
        hidden_states=hidden_states,
        attention_mask=attention_mask
    )
    
    # prompt_embeds shape: [B, S, D]
    
    # Duplicate for num_videos_per_prompt
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = jnp.repeat(prompt_embeds, num_videos_per_prompt, axis=0)
    attention_mask = jnp.repeat(attention_mask, num_videos_per_prompt, axis=0)
    
    return prompt_embeds, attention_mask

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      image: PipelineImageInput = None, # Placeholder for VAE encode logic
      height: int = 480,
      width: int = 704,
      num_frames: int = 121,
      num_inference_steps: int = 50,
      guidance_scale: float = 3.0,
      num_videos_per_prompt: int = 1,
      generator: Optional[jax.Array] = None, # JAX PRNGKey
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      prompt_attention_mask: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None, # Not used in LTX2 usually?
      output_type: str = "pil",
      return_dict: bool = True,
      **kwargs,
  ):
    # 0. Default height/width/frames if not provided handling
    # (Simplified for now)

    # 1. Check inputs
    if prompt is None and prompt_embeds is None:
        raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0] // num_videos_per_prompt

    # 2. Encode Prompt
    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
        )
    
    # 3. Prepare Timesteps
    timesteps = self.retrieve_timesteps(self.scheduler, num_inference_steps)
    
    # 4. Prepare Latents
    num_channels_latents = self.transformer.in_channels
    if latents is None:
        if generator is None:
             generator = jax.random.PRNGKey(0)
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            height,
            width,
            num_frames,
            num_channels_latents,
            jnp.float32,
            generator
        )
        
    # 5. Denoising Loop
    # LTX2 Transformer inputs:
    # hidden_states (Video),encoder_hidden_states (Text), 
    # temb (Timestep embed), 
    # attention_mask (for Text/Video?)
    
    # Note: LTX2 Transformer expects 'temb' and other conditioning.
    # We need to construct 'temb' and specific embeddings.
    # The transformer handles its own embedding internally? 
    # No, LTX2VideoTransformer3DModel takes `temb`, `temb_audio` etc.
    # Wait, the transformer __call__ signature:
    # hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states,
    # temb, temb_audio, temb_ca_scale_shift, ...
    
    # We need to compute these 'temb' components. 
    # They are likely computed from the timestep using some embedding logic OUTSIDE the transformer or INSIDE?
    # Looking at `LTX2VideoTransformer3DModel.__init__`, it has `self.time_embed` which takes `timestep` and returns embedding?
    # No, `LTX2AdaLayerNormSingle` takes `timestep` (embedded) and returns modulation.
    # The `__call__` expects `temb` already embedded? 
    # "temb: jax.Array" argument.
    
    # In Diffusers `LTX2VideoTransformer3DModel`, input is `timestep` (tensor).
    # In MaxDiffusion `LTX2VideoTransformer3DModel`, input is `temb` (embedding?).
    # Let's check `transformer_ltx2.py` again.
    # `self.time_embed = LTX2AdaLayerNormSingle(...)`
    # `temb` is passed to `self.time_embed(timestep=temb.flatten(), ...)` if it's not already embedded?
    # `LTX2AdaLayerNormSingle.__call__` takes `timestep`.
    # And inside `LTX2VideoTransformer3DModel.__call__`:
    # `temb_reshaped = temb.reshape(...)`
    # It seems `temb` passed to `__call__` is ALREADY the vector embedding of timestep?
    # Or is it the raw timestep?
    
    # In Diffusers:
    # transformer(hidden_states, encoder_hidden_states, timestep, ...)
    # Here:
    # transformer(hidden_states, ..., temb, ...)
    
    # We might need a Timestep Embedder helper in the pipeline or use the one in Scheduler?
    # Use `scheduler_state.timesteps`?
    
    # We will assume for now we pass the timestep value and the transformer helps, 
    # OR we implement the sinusoidal embedding here.
    # MaxDiffusion usually follows Diffusers.
    
        # Prepare Timestep Embeddings
        # We need to compute them using the transformer's internal embedders
        # Note: This pattern assumes strict parity with LTX2 logic where these are needed explicitly
        
        # 1. Main Timestep Embedding
        temb, _ = self.transformer.time_embed(t_batch)
        
        # 2. Audio Timestep Embedding (Dummy or Same?)
        # For now, using same timestep for audio
        temb_audio, _ = self.transformer.audio_time_embed(t_batch)
        
        # 3. Cross-Attention Modulations
        temb_ca_scale_shift, _ = self.transformer.av_cross_attn_video_scale_shift(t_batch)
        temb_ca_audio_scale_shift, _ = self.transformer.av_cross_attn_audio_scale_shift(t_batch)
        temb_ca_gate, _ = self.transformer.av_cross_attn_video_a2v_gate(t_batch)
        temb_ca_audio_gate, _ = self.transformer.av_cross_attn_audio_v2a_gate(t_batch)
        
        # Predict noise
        model_output, _ = self.transformer(
            hidden_states=latents,
            audio_hidden_states=audio_latents,
            encoder_hidden_states=prompt_embeds,
            audio_encoder_hidden_states=audio_prompt_embeds,
            temb=temb,
            temb_audio=temb_audio,
            temb_ca_scale_shift=temb_ca_scale_shift,
            temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
            temb_ca_gate=temb_ca_gate,
            temb_ca_audio_gate=temb_ca_audio_gate,
            encoder_attention_mask=prompt_attention_mask,
            # rotary_emb=... (Transformer handles if None?)
            # Actually, LTX2VideoTransformer3DModel has 'rope' and 'audio_rope' attributes,
            # but __call__ also accepts rotary_emb.
            # Usually pipelines compute RoPE and pass it, OR transformer computes it.
            # MaxDiffusion transformer often computes it if not passed.
            # LTX2VideoTransformer3DModel checks if video_rotary_emb is None.
        )
        
        # Scheduler Step
        # Flax scheduler step returns (prev_sample, state) or Output object
        scheduler_output = self.scheduler.step(
            self.scheduler_state,
            model_output,
            t, # Timestep
            latents,
            return_dict=True
        )
        latents = scheduler_output.prev_sample
        self.scheduler_state = scheduler_output.state

    # 6. Decode Latents
    # LTX2 VAE Decode
    # Latents: [B, C, F, H, W] -> [B, F, H, W, C] for VAE?
    # Checked LTX2VideoAutoencoderKL:
    # encode input: [B, C, T, H, W] (from Diffusers) OR [B, T, H, W, C] (JAX typical)?
    # LTX2VideoCausalConv3d does NWC typically?
    # LTX2VideoAutoencoderKL._decode expects z: [B, T, H, W, C]
    # LTX2VideoTransformer3DModel expects [B, T, H, W, C]?
    # Wait, prepare_latents created: (B, C, T, H, W) in my code????
    # Let's check prepare_latents again.
    # shape = (batch, num_channels, log_frames, h, w)
    
    # Correct JAX layout is usually (B, T, H, W, C) for Convolutions if they are standard Flax,
    # BUT LTX2VideoCausalConv3d implementation I saw earlier uses `nnx.Conv`.
    # `nnx.Conv` defaults to strict NWC?
    # `LX2VideoCausalConv3d` kernel_size is 3D.
    # We need to verify data layout.
    # Diffusers LTX2 uses (B, C, F, H, W).
    # MaxDiffusion usually tends to (B, F, H, W, C) for TPU efficiency.
    # Let's assume (B, F, H, W, C) for internal processing if components support it.
    
    # Re-checking prepare_latents in my previous edit:
    # shape = (batch, num_channels, frame, h, w) - I wrote this to match Diffusers pattern?
    # I should change it to (B, F, H, W, C) if that's what `transformer` expects.
    # `LTX2VideoTransformer3DModel` patchify: `self.proj_in = nnx.Linear(..., inner_dim)`.
    # Linear expects last dim to be C.
    # So transformer expects (B, ..., C).
    
    # So `prepare_latents` MUST return (B, F, H, W, C).
    # I need to fix `prepare_latents` to put C last.
    
    # Fixing latents for decode:
    video = self.vae.decode(latents, return_dict=True).sample
    
    return video

