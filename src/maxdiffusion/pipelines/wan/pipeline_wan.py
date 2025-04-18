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

from typing import Union, List
from transformers import AutoTokenizer, UMT5EncoderModel
import torch
from ...models.wan.transformers.transformer_flux_wan_nnx import WanModel
from ...models.wan.autoencoder_kl_wan import AutoencoderKLWan 
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from ...video_processor import VideoProcessor
from ...schedulers import FlowMatchEulerDiscreteScheduler

class WanPipeline(FlaxDiffusionPipeline):

  def __init__(
      self,
      tokenizer: AutoTokenizer,
      text_encoder: UMT5EncoderModel,
      transformer: WanModel,
      vae: AutoencoderKLWan,
      scheduler: FlowMatchEulerDiscreteScheduler,
  ):
    super().__init__()

    self.register_modules(
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      transformer=transformer,
      scheduler=scheduler
    )

    self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
    self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)


  def _get_t5_prompt_embds(
    self,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = self.tokenizer(
      prompt,
      padding="max_length",
      max_length=max_sequence_length,
      truncation=True,
      add_special_tokens=True,
      return_attention_mask=True,
      return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
    # prompt_embeds = prompt_embeds.to(dtype=dtype)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds