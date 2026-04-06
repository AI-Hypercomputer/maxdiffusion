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

import os
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
DIFFUSERS_SRC = os.path.join(REPO_ROOT, "diffusers", "src")
if DIFFUSERS_SRC not in sys.path:
  sys.path.insert(0, DIFFUSERS_SRC)

import jax.numpy as jnp
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from diffusers.pipelines.wan.image_processor import WanAnimateImageProcessor as HFWanAnimateImageProcessor
from diffusers.pipelines.wan.pipeline_wan_animate import WanAnimatePipeline as HFWanAnimatePipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor as HFVideoProcessor

from ..image_processor import VaeImageProcessor as MaxVaeImageProcessor
from ..pipelines.wan.wan_pipeline import WanPipeline as MaxWanPipeline
from ..pipelines.wan.wan_pipeline_animate import (
    WanAnimatePipeline as MaxWanAnimatePipeline,
    animate_transformer_forward_pass,
)
from ..schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler
from ..video_processor import VideoProcessor as MaxVideoProcessor


def to_numpy(array):
  if isinstance(array, torch.Tensor):
    if array.dtype == torch.bfloat16:
      array = array.float()
    return array.detach().cpu().numpy()
  return np.asarray(array)


def hf_channel_first_to_last(array):
  return np.transpose(to_numpy(array), (0, 2, 3, 4, 1))


class FakeTokenBatch:

  def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    self.input_ids = input_ids
    self.attention_mask = attention_mask


class FakeTokenizer:

  def __call__(
      self,
      prompt,
      padding,
      max_length,
      truncation,
      add_special_tokens,
      return_attention_mask,
      return_tensors,
  ):
    del padding, truncation, add_special_tokens, return_attention_mask, return_tensors
    input_ids = []
    attention_mask = []
    for text in prompt:
      seq_len = max(1, min(max_length, len(text.split()) + 1))
      base = (sum(ord(ch) for ch in text) % 37) + 1
      ids = torch.arange(base, base + seq_len, dtype=torch.long)
      pad = torch.zeros(max_length - seq_len, dtype=torch.long)
      input_ids.append(torch.cat([ids, pad], dim=0))
      attention_mask.append(
          torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(max_length - seq_len, dtype=torch.long)], dim=0)
      )
    return FakeTokenBatch(torch.stack(input_ids), torch.stack(attention_mask))


class FakeTextEncoder:
  dtype = torch.float32

  def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    hidden = torch.stack(
        [
            input_ids.float(),
            input_ids.float() * 0.5 + attention_mask.float(),
            attention_mask.float() * 2.0,
            input_ids.float() - attention_mask.float(),
        ],
        dim=-1,
    )
    return SimpleNamespace(last_hidden_state=hidden)


class FakeImageBatch(dict):

  @property
  def pixel_values(self):
    return self["pixel_values"]

  def to(self, device=None):
    pixel_values = self.pixel_values if device is None else self.pixel_values.to(device)
    return FakeImageBatch(pixel_values=pixel_values)


class FakeImageProcessor:

  def __call__(self, images, return_tensors):
    del return_tensors
    if not isinstance(images, list):
      images = [images]
    pixel_values = []
    for idx, _ in enumerate(images):
      base = idx + 1
      pixel_values.append(
          torch.tensor(
              [[[base, base + 1], [base + 2, base + 3]]],
              dtype=torch.float32,
          )
      )
    return FakeImageBatch(pixel_values=torch.stack(pixel_values))


class FakeImageEncoder:

  def __call__(self, pixel_values, output_hidden_states: bool):
    del output_hidden_states
    hidden = pixel_values.reshape(pixel_values.shape[0], pixel_values.shape[1], -1)
    if isinstance(pixel_values, torch.Tensor):
      hidden = hidden.transpose(1, 2)
    else:
      hidden = jnp.transpose(hidden, (0, 2, 1))
    return SimpleNamespace(hidden_states=[hidden * 0.25, hidden * 0.5, hidden * 0.75])


class FakeTorchLatentDist:

  def __init__(self, latents: torch.Tensor):
    self._latents = latents

  def sample(self, generator=None):
    del generator
    return self._latents

  def mode(self):
    return self._latents


class FakeTorchEncodeOutput:

  def __init__(self, latents: torch.Tensor):
    self.latent_dist = FakeTorchLatentDist(latents)


class FakeTorchVAE:
  dtype = torch.float32

  class config:
    z_dim = 2
    latents_mean = [0.5, -0.25]
    latents_std = [2.0, 4.0]

  def encode(self, x: torch.Tensor):
    latents = x[:, :2, ::4, ::8, ::8] + 1.0
    return FakeTorchEncodeOutput(latents)


class FakeJaxEncodeOutput:

  def __init__(self, latents: jnp.ndarray):
    self._latents = latents

  def mode(self):
    return self._latents


class FakeJaxVAE:
  dtype = jnp.float32
  z_dim = 2
  latents_mean = [0.5, -0.25]
  latents_std = [2.0, 4.0]

  def encode(self, x: jnp.ndarray, cache):
    del cache
    latents = jnp.transpose(x[:, :2, ::4, ::8, ::8] + 1.0, (0, 2, 3, 4, 1))
    return (FakeJaxEncodeOutput(latents),)


class WanAnimateDiffusersParityTest(unittest.TestCase):

  def setUp(self):
    self.max_pipeline = MaxWanAnimatePipeline.__new__(MaxWanAnimatePipeline)
    self.max_pipeline.tokenizer = FakeTokenizer()
    self.max_pipeline.text_encoder = FakeTextEncoder()
    self.max_pipeline.image_processor = FakeImageProcessor()
    self.max_pipeline.image_encoder = FakeImageEncoder()
    self.max_pipeline.vae = FakeJaxVAE()
    self.max_pipeline.vae_scale_factor_temporal = 4
    self.max_pipeline.vae_scale_factor_spatial = 8
    self.max_pipeline.mesh = nullcontext()
    self.max_pipeline.vae_mesh = nullcontext()
    self.max_pipeline.config = SimpleNamespace(logical_axis_rules=())
    self.max_pipeline.vae_logical_axis_rules = ()
    self.max_pipeline.vae_cache = None
    self.max_pipeline.video_processor_for_mask = MaxVideoProcessor(
        vae_scale_factor=8, do_normalize=False, do_convert_grayscale=True
    )

    self.hf_pipeline = HFWanAnimatePipeline.__new__(HFWanAnimatePipeline)
    self.hf_pipeline.tokenizer = self.max_pipeline.tokenizer
    self.hf_pipeline.text_encoder = self.max_pipeline.text_encoder
    self.hf_pipeline.image_processor = self.max_pipeline.image_processor
    self.hf_pipeline.image_encoder = self.max_pipeline.image_encoder
    self.hf_pipeline.vae = FakeTorchVAE()
    self.hf_pipeline.vae_scale_factor_temporal = 4
    self.hf_pipeline.vae_scale_factor_spatial = 8
    self.hf_pipeline.video_processor_for_mask = HFVideoProcessor(
        vae_scale_factor=8, do_normalize=False, do_convert_grayscale=True
    )

  def test_encode_prompt_matches_diffusers(self):
    prompt = ["  Hello   world  ", "test &amp; check"]
    negative_prompt = ["bad motion", "low detail"]

    max_prompt, max_negative = MaxWanPipeline.encode_prompt(
        self.max_pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=2,
        max_sequence_length=8,
    )
    hf_prompt, hf_negative = HFWanAnimatePipeline.encode_prompt(
        self.hf_pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=2,
        max_sequence_length=8,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(to_numpy(max_prompt), to_numpy(hf_prompt), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(to_numpy(max_negative), to_numpy(hf_negative), atol=0.0, rtol=0.0)

  def test_encode_image_matches_diffusers_call_semantics(self):
    image = object()

    max_image = MaxWanPipeline.encode_image(self.max_pipeline, image, num_videos_per_prompt=3)
    hf_image = HFWanAnimatePipeline.encode_image(self.hf_pipeline, image, device=torch.device("cpu")).repeat(3, 1, 1)

    np.testing.assert_allclose(to_numpy(max_image), to_numpy(hf_image), atol=0.0, rtol=0.0)

  def test_pad_video_frames_matches_diffusers(self):
    frames = [1, 2, 3, 4, 5]

    max_frames = MaxWanAnimatePipeline.pad_video_frames(frames, 10)
    hf_frames = HFWanAnimatePipeline.pad_video_frames(self.hf_pipeline, frames, 10)

    self.assertEqual(max_frames, hf_frames)

  def test_prepare_reference_image_latents_matches_diffusers(self):
    image = torch.arange(1 * 3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)

    max_latents = MaxWanAnimatePipeline.prepare_reference_image_latents(
        self.max_pipeline, jnp.array(image.numpy()), batch_size=2, dtype=jnp.float32
    )
    hf_latents = HFWanAnimatePipeline.prepare_reference_image_latents(
        self.hf_pipeline, image, batch_size=2, dtype=torch.float32, device=torch.device("cpu")
    )

    np.testing.assert_allclose(to_numpy(max_latents), hf_channel_first_to_last(hf_latents), atol=0.0, rtol=0.0)

  def test_prepare_pose_latents_matches_diffusers(self):
    pose_video = torch.arange(1 * 3 * 9 * 16 * 16, dtype=torch.float32).reshape(1, 3, 9, 16, 16)

    max_latents = MaxWanAnimatePipeline.prepare_pose_latents(
        self.max_pipeline, jnp.array(pose_video.numpy()), batch_size=2, dtype=jnp.float32
    )
    hf_latents = HFWanAnimatePipeline.prepare_pose_latents(
        self.hf_pipeline, pose_video, batch_size=2, dtype=torch.float32, device=torch.device("cpu")
    )

    np.testing.assert_allclose(to_numpy(max_latents), hf_channel_first_to_last(hf_latents), atol=0.0, rtol=0.0)

  def test_prepare_segment_latents_matches_diffusers_when_latents_are_provided(self):
    max_input = jnp.arange(1 * 4 * 2 * 2 * 2, dtype=jnp.float32).reshape(1, 4, 2, 2, 2) / 10.0
    hf_input = torch.tensor(np.transpose(to_numpy(max_input), (0, 4, 1, 2, 3)))

    max_latents = MaxWanAnimatePipeline.prepare_segment_latents(
        self.max_pipeline,
        batch_size=1,
        height=16,
        width=16,
        segment_frame_length=9,
        dtype=jnp.bfloat16,
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        latents=max_input,
    )
    hf_latents = HFWanAnimatePipeline.prepare_latents(
        self.hf_pipeline,
        batch_size=1,
        num_channels_latents=2,
        height=16,
        width=16,
        num_frames=9,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        latents=hf_input,
    )

    np.testing.assert_allclose(to_numpy(max_latents), hf_channel_first_to_last(hf_latents), atol=0.0, rtol=0.0)

  def test_prepare_prev_segment_cond_latents_matches_diffusers_for_animate(self):
    prev_segment = torch.arange(1 * 3 * 1 * 16 * 16, dtype=torch.float32).reshape(1, 3, 1, 16, 16)

    max_latents = MaxWanAnimatePipeline.prepare_prev_segment_cond_latents(
        self.max_pipeline,
        prev_segment_cond_video=jnp.array(prev_segment.numpy()),
        background_video=None,
        mask_video=None,
        batch_size=1,
        segment_frame_length=9,
        start_frame=4,
        height=16,
        width=16,
        prev_segment_cond_frames=1,
        task="animate",
        dtype=jnp.float32,
    )
    hf_latents = HFWanAnimatePipeline.prepare_prev_segment_cond_latents(
        self.hf_pipeline,
        prev_segment_cond_video=prev_segment,
        background_video=None,
        mask_video=None,
        batch_size=1,
        segment_frame_length=9,
        start_frame=4,
        height=16,
        width=16,
        prev_segment_cond_frames=1,
        task="animate",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(to_numpy(max_latents), hf_channel_first_to_last(hf_latents), atol=0.0, rtol=0.0)

  def test_prepare_prev_segment_cond_latents_animate_encodes_full_segment_like_diffusers(self):
    call_lengths = []

    def fake_encode(video, dtype):
      del dtype
      call_lengths.append(video.shape[2])
      latent_t = (video.shape[2] - 1) // self.max_pipeline.vae_scale_factor_temporal + 1
      latent_h = video.shape[3] // self.max_pipeline.vae_scale_factor_spatial
      latent_w = video.shape[4] // self.max_pipeline.vae_scale_factor_spatial
      return jnp.ones((video.shape[0], latent_t, latent_h, latent_w, self.max_pipeline.vae.z_dim), dtype=jnp.float32)

    self.max_pipeline._encode_video_to_latents = fake_encode
    prev_segment = jnp.ones((1, 3, 1, 16, 16), dtype=jnp.float32)

    _ = MaxWanAnimatePipeline.prepare_prev_segment_cond_latents(
        self.max_pipeline,
        prev_segment_cond_video=prev_segment,
        background_video=None,
        mask_video=None,
        batch_size=1,
        segment_frame_length=9,
        start_frame=4,
        height=16,
        width=16,
        prev_segment_cond_frames=1,
        task="animate",
        dtype=jnp.float32,
    )

    self.assertEqual(call_lengths, [9])

  def test_prepare_prev_segment_cond_latents_animate_first_segment_encodes_zero_filled_segment_like_diffusers(self):
    call_lengths = []

    def fake_encode(video, dtype):
      del dtype
      call_lengths.append(video.shape[2])
      latent_t = (video.shape[2] - 1) // self.max_pipeline.vae_scale_factor_temporal + 1
      latent_h = video.shape[3] // self.max_pipeline.vae_scale_factor_spatial
      latent_w = video.shape[4] // self.max_pipeline.vae_scale_factor_spatial
      return jnp.zeros(
          (video.shape[0], latent_t, latent_h, latent_w, self.max_pipeline.vae.z_dim), dtype=jnp.float32
      )

    self.max_pipeline._encode_video_to_latents = fake_encode

    _ = MaxWanAnimatePipeline.prepare_prev_segment_cond_latents(
        self.max_pipeline,
        prev_segment_cond_video=None,
        background_video=None,
        mask_video=None,
        batch_size=1,
        segment_frame_length=9,
        start_frame=0,
        height=16,
        width=16,
        prev_segment_cond_frames=1,
        task="animate",
        dtype=jnp.float32,
    )

    self.assertEqual(call_lengths, [9])

  def test_prepare_prev_segment_cond_latents_matches_diffusers_for_replace(self):
    prev_segment = torch.arange(1 * 3 * 1 * 16 * 16, dtype=torch.float32).reshape(1, 3, 1, 16, 16)
    background = torch.arange(1 * 3 * 9 * 16 * 16, dtype=torch.float32).reshape(1, 3, 9, 16, 16) / 10.0
    mask = (torch.arange(1 * 1 * 9 * 16 * 16, dtype=torch.float32).reshape(1, 1, 9, 16, 16) % 3 == 0).float()

    max_latents = MaxWanAnimatePipeline.prepare_prev_segment_cond_latents(
        self.max_pipeline,
        prev_segment_cond_video=jnp.array(prev_segment.numpy()),
        background_video=jnp.array(background.numpy()),
        mask_video=jnp.array(mask.numpy()),
        batch_size=1,
        segment_frame_length=9,
        start_frame=4,
        height=16,
        width=16,
        prev_segment_cond_frames=1,
        task="replace",
        dtype=jnp.float32,
    )
    hf_latents = HFWanAnimatePipeline.prepare_prev_segment_cond_latents(
        self.hf_pipeline,
        prev_segment_cond_video=prev_segment,
        background_video=background,
        mask_video=mask,
        batch_size=1,
        segment_frame_length=9,
        start_frame=4,
        height=16,
        width=16,
        prev_segment_cond_frames=1,
        task="replace",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(to_numpy(max_latents), hf_channel_first_to_last(hf_latents), atol=0.0, rtol=0.0)

  def test_resize_mask_to_latent_spatial_matches_torch_nearest(self):
    mask = (torch.arange(1 * 1 * 9 * 16 * 16, dtype=torch.float32).reshape(1, 1, 9, 16, 16) % 5 == 0).float()
    hf_mask = mask.permute(0, 2, 1, 3, 4).flatten(0, 1)
    hf_mask = F.interpolate(hf_mask, size=(2, 2), mode="nearest")
    hf_mask = hf_mask.unflatten(0, (1, -1)).permute(0, 2, 1, 3, 4)

    max_mask = MaxWanAnimatePipeline._resize_mask_to_latent_spatial(self.max_pipeline, jnp.array(mask.numpy()), 2, 2)

    np.testing.assert_allclose(to_numpy(max_mask), to_numpy(hf_mask), atol=0.0, rtol=0.0)

  def test_reference_image_processor_matches_diffusers_fill_resize(self):
    image = PIL.Image.fromarray(np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3))
    max_processor = MaxVaeImageProcessor(
        vae_scale_factor=8,
        spatial_patch_size=(2, 2),
        resize_mode="fill",
        fill_color=0,
    )
    hf_processor = HFWanAnimateImageProcessor(vae_scale_factor=8, spatial_patch_size=(2, 2), fill_color=0)

    max_image = max_processor.preprocess(image, height=16, width=16)
    hf_image = hf_processor.preprocess(image, height=16, width=16, resize_mode="fill")

    np.testing.assert_allclose(to_numpy(max_image), to_numpy(hf_image), atol=0.0, rtol=0.0)

  def test_video_processor_matches_diffusers(self):
    frames = [
        PIL.Image.fromarray(np.full((9, 13, 3), fill_value=value, dtype=np.uint8)) for value in (16, 96, 224)
    ]
    max_processor = MaxVideoProcessor(vae_scale_factor=8)
    hf_processor = HFVideoProcessor(vae_scale_factor=8)

    max_video = max_processor.preprocess_video(frames, height=16, width=16)
    hf_video = hf_processor.preprocess_video(frames, height=16, width=16)

    np.testing.assert_allclose(to_numpy(max_video), to_numpy(hf_video), atol=0.0, rtol=0.0)

  def test_mask_video_preprocessing_matches_diffusers(self):
    masks = [
        PIL.Image.fromarray(np.full((9, 13), fill_value=value, dtype=np.uint8)) for value in (0, 128, 255)
    ]

    max_mask = self.max_pipeline.video_processor_for_mask.preprocess_video(masks, height=16, width=16)
    hf_mask = self.hf_pipeline.video_processor_for_mask.preprocess_video(masks, height=16, width=16)

    np.testing.assert_allclose(to_numpy(max_mask), to_numpy(hf_mask), atol=0.0, rtol=0.0)

  def test_check_inputs_matches_diffusers_validation(self):
    invalid_calls = [
        dict(
            prompt="prompt",
            negative_prompt=None,
            image=PIL.Image.new("RGB", (16, 16)),
            pose_video=[PIL.Image.new("RGB", (16, 16))],
            face_video=[PIL.Image.new("RGB", (16, 16))],
            background_video=None,
            mask_video=None,
            height=16,
            width=16,
            prompt_embeds=jnp.zeros((1, 1, 1)),
            negative_prompt_embeds=None,
            image_embeds=None,
            mode="animate",
            prev_segment_conditioning_frames=1,
        ),
        dict(
            prompt="prompt",
            negative_prompt=None,
            image=PIL.Image.new("RGB", (16, 16)),
            pose_video=[PIL.Image.new("RGB", (16, 16))],
            face_video=[PIL.Image.new("RGB", (16, 16))],
            background_video=None,
            mask_video=None,
            height=18,
            width=16,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            image_embeds=None,
            mode="animate",
            prev_segment_conditioning_frames=1,
        ),
        dict(
            prompt="prompt",
            negative_prompt=None,
            image=PIL.Image.new("RGB", (16, 16)),
            pose_video=[PIL.Image.new("RGB", (16, 16))],
            face_video=[PIL.Image.new("RGB", (16, 16))],
            background_video=None,
            mask_video=None,
            height=16,
            width=16,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            image_embeds=None,
            mode="replace",
            prev_segment_conditioning_frames=3,
        ),
    ]

    for kwargs in invalid_calls:
      with self.subTest(kwargs=kwargs):
        with self.assertRaises(ValueError) as max_ctx:
          self.max_pipeline.check_inputs(**kwargs)
        with self.assertRaises(ValueError) as hf_ctx:
          self.hf_pipeline.check_inputs(**kwargs)
        self.assertEqual(str(max_ctx.exception), str(hf_ctx.exception))

  def test_animate_transformer_forward_pass_matches_diffusers_layout(self):
    capture = {}

    class FakeTransformer:

      def __call__(
          self,
          hidden_states,
          timestep,
          encoder_hidden_states,
          encoder_hidden_states_image,
          pose_hidden_states,
          face_pixel_values,
          motion_encode_batch_size,
          return_dict,
      ):
        capture["hidden_states"] = hidden_states
        capture["timestep"] = timestep
        capture["encoder_hidden_states"] = encoder_hidden_states
        capture["encoder_hidden_states_image"] = encoder_hidden_states_image
        capture["pose_hidden_states"] = pose_hidden_states
        capture["face_pixel_values"] = face_pixel_values
        capture["motion_encode_batch_size"] = motion_encode_batch_size
        capture["return_dict"] = return_dict
        return (hidden_states[:, :2],)

    latents = jnp.arange(1 * 3 * 2 * 2 * 2, dtype=jnp.float32).reshape(1, 3, 2, 2, 2)
    reference_latents = latents + 100.0
    pose_latents = latents + 200.0
    face_video = jnp.arange(1 * 3 * 9 * 4 * 4, dtype=jnp.float32).reshape(1, 3, 9, 4, 4)
    timestep = jnp.array([5], dtype=jnp.int32)
    prompt_embeds = jnp.arange(1 * 4 * 3, dtype=jnp.float32).reshape(1, 4, 3)
    image_embeds = prompt_embeds + 10.0

    with mock.patch("maxdiffusion.pipelines.wan.wan_pipeline_animate.nnx.merge", return_value=FakeTransformer()):
      noise_pred = animate_transformer_forward_pass.__wrapped__(
          graphdef=None,
          sharded_state=None,
          rest_of_state=None,
          latents=latents,
          reference_latents=reference_latents,
          pose_latents=pose_latents,
          face_video_segment=face_video,
          timestep=timestep,
          encoder_hidden_states=prompt_embeds,
          encoder_hidden_states_image=image_embeds,
          motion_encode_batch_size=7,
      )

    expected_hidden = jnp.transpose(jnp.concatenate([latents, reference_latents], axis=-1), (0, 4, 1, 2, 3))
    expected_pose = jnp.transpose(pose_latents, (0, 4, 1, 2, 3))

    np.testing.assert_allclose(to_numpy(capture["hidden_states"]), to_numpy(expected_hidden), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(to_numpy(capture["pose_hidden_states"]), to_numpy(expected_pose), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(to_numpy(capture["face_pixel_values"]), to_numpy(face_video), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(to_numpy(capture["encoder_hidden_states"]), to_numpy(prompt_embeds), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        to_numpy(capture["encoder_hidden_states_image"]), to_numpy(image_embeds), atol=0.0, rtol=0.0
    )
    self.assertEqual(capture["motion_encode_batch_size"], 7)
    self.assertFalse(capture["return_dict"])
    np.testing.assert_allclose(to_numpy(noise_pred), to_numpy(latents), atol=0.0, rtol=0.0)

  def test_single_denoising_step_matches_diffusers_with_cfg(self):
    class FakeDenoiseTransformer:

      def __call__(
          self,
          hidden_states,
          timestep,
          encoder_hidden_states,
          encoder_hidden_states_image,
          pose_hidden_states,
          face_pixel_values,
          motion_encode_batch_size,
          return_dict,
      ):
        del motion_encode_batch_size, return_dict

        def _scalar(x):
          if isinstance(x, torch.Tensor):
            return x.float().mean(dim=tuple(range(1, x.ndim))).view(-1, 1, 1, 1, 1)
          return jnp.mean(x.astype(jnp.float32), axis=tuple(range(1, x.ndim))).reshape((-1, 1, 1, 1, 1))

        noise = (
            hidden_states[:, :2] * 0.5
            + pose_hidden_states[:, :2] * 0.1
            + _scalar(encoder_hidden_states) * 0.01
            + _scalar(encoder_hidden_states_image) * 0.02
            + _scalar(face_pixel_values) * 0.03
            + _scalar(timestep) * 0.001
        )
        return (noise,)

    guidance_scale = 3.0
    timestep_count = 4
    fake_transformer = FakeDenoiseTransformer()

    max_latents = jnp.arange(1 * 3 * 2 * 2 * 2, dtype=jnp.float32).reshape(1, 3, 2, 2, 2) / 10.0
    max_reference = max_latents + 10.0
    max_pose = max_latents + 20.0
    max_face = jnp.arange(1 * 3 * 9 * 4 * 4, dtype=jnp.float32).reshape(1, 3, 9, 4, 4) / 255.0
    max_prompt = jnp.arange(1 * 4 * 3, dtype=jnp.float32).reshape(1, 4, 3) / 7.0
    max_negative = max_prompt - 0.5
    max_image = max_prompt + 1.0

    hf_latents = torch.tensor(np.transpose(to_numpy(max_latents), (0, 4, 1, 2, 3)))
    hf_reference = torch.tensor(np.transpose(to_numpy(max_reference), (0, 4, 1, 2, 3)))
    hf_pose = torch.tensor(np.transpose(to_numpy(max_pose), (0, 4, 1, 2, 3)))
    hf_face = torch.tensor(to_numpy(max_face))
    hf_prompt = torch.tensor(to_numpy(max_prompt))
    hf_negative = torch.tensor(to_numpy(max_negative))
    hf_image = torch.tensor(to_numpy(max_image))

    scheduler_config = dict(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0)
    max_scheduler = FlaxUniPCMultistepScheduler(**scheduler_config)
    max_state = max_scheduler.create_state()
    max_state = max_scheduler.set_timesteps(max_state, num_inference_steps=timestep_count, shape=max_latents.shape)

    hf_scheduler = UniPCMultistepScheduler(**scheduler_config)
    hf_scheduler.set_timesteps(timestep_count, device="cpu")

    timestep = int(to_numpy(hf_scheduler.timesteps[0]))
    max_timestep = jnp.full((max_latents.shape[0],), timestep, dtype=jnp.int32)
    hf_timestep = torch.full((hf_latents.shape[0],), timestep, dtype=torch.int64)

    with mock.patch("maxdiffusion.pipelines.wan.wan_pipeline_animate.nnx.merge", return_value=fake_transformer):
      max_noise_cond = animate_transformer_forward_pass.__wrapped__(
          graphdef=None,
          sharded_state=None,
          rest_of_state=None,
          latents=max_latents,
          reference_latents=max_reference,
          pose_latents=max_pose,
          face_video_segment=max_face,
          timestep=max_timestep,
          encoder_hidden_states=max_prompt,
          encoder_hidden_states_image=max_image,
          motion_encode_batch_size=5,
      )
      max_noise_uncond = animate_transformer_forward_pass.__wrapped__(
          graphdef=None,
          sharded_state=None,
          rest_of_state=None,
          latents=max_latents,
          reference_latents=max_reference,
          pose_latents=max_pose,
          face_video_segment=max_face * 0 - 1,
          timestep=max_timestep,
          encoder_hidden_states=max_negative,
          encoder_hidden_states_image=max_image,
          motion_encode_batch_size=5,
      )

    max_noise = max_noise_uncond + guidance_scale * (max_noise_cond - max_noise_uncond)

    hf_latent_model_input = torch.cat([hf_latents, hf_reference], dim=1)
    hf_noise_cond = fake_transformer(
        hidden_states=hf_latent_model_input,
        timestep=hf_timestep,
        encoder_hidden_states=hf_prompt,
        encoder_hidden_states_image=hf_image,
        pose_hidden_states=hf_pose,
        face_pixel_values=hf_face,
        motion_encode_batch_size=5,
        return_dict=False,
    )[0]
    hf_noise_uncond = fake_transformer(
        hidden_states=hf_latent_model_input,
        timestep=hf_timestep,
        encoder_hidden_states=hf_negative,
        encoder_hidden_states_image=hf_image,
        pose_hidden_states=hf_pose,
        face_pixel_values=hf_face * 0 - 1,
        motion_encode_batch_size=5,
        return_dict=False,
    )[0]
    hf_noise = hf_noise_uncond + guidance_scale * (hf_noise_cond - hf_noise_uncond)

    max_next, _ = max_scheduler.step(max_state, max_noise, timestep, max_latents, return_dict=False)
    hf_next = hf_scheduler.step(hf_noise, timestep, hf_latents, return_dict=False)[0]

    np.testing.assert_allclose(to_numpy(max_noise), hf_channel_first_to_last(hf_noise), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(max_next), hf_channel_first_to_last(hf_next), atol=1e-5, rtol=1e-5)

  def test_flax_unipc_flow_sigmas_match_diffusers(self):
    scheduler_config = dict(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0)

    max_scheduler = FlaxUniPCMultistepScheduler(**scheduler_config)
    max_state = max_scheduler.create_state()
    max_state = max_scheduler.set_timesteps(max_state, num_inference_steps=4, shape=(1, 2, 3, 4, 5))

    hf_scheduler = UniPCMultistepScheduler(**scheduler_config)
    hf_scheduler.set_timesteps(4, device="cpu")

    np.testing.assert_array_equal(to_numpy(max_state.timesteps), to_numpy(hf_scheduler.timesteps))
    np.testing.assert_allclose(to_numpy(max_state.sigmas), to_numpy(hf_scheduler.sigmas), atol=1e-7, rtol=0.0)

    max_sample = jnp.arange(1 * 2 * 3 * 4 * 5, dtype=jnp.float32).reshape(1, 2, 3, 4, 5) / 10.0
    hf_sample = torch.tensor(to_numpy(max_sample))

    for step_index, timestep in enumerate(to_numpy(hf_scheduler.timesteps[:3])):
      hf_model_output = torch.full_like(hf_sample, 0.1 * (step_index + 1))
      max_model_output = jnp.array(to_numpy(hf_model_output))

      hf_sample = hf_scheduler.step(hf_model_output, int(timestep), hf_sample, return_dict=False)[0]
      max_sample, max_state = max_scheduler.step(
          max_state, max_model_output, int(timestep), max_sample, return_dict=False
      )

      np.testing.assert_allclose(to_numpy(max_sample), to_numpy(hf_sample), atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
