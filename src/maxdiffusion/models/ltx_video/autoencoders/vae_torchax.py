from maxdiffusion.models.ltx_video.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from maxdiffusion.models.ltx_video.autoencoders import causal_conv3d
from maxdiffusion.models.ltx_video.autoencoders.vae_encode import vae_encode, vae_decode

import jax
from torchax import interop
import os
from torchax import default_env
import jax.numpy as jnp

# remove weight attribute to avoid error in JittableModule
# in the future, this will be fixed in ltxv public repo
delattr(causal_conv3d.CausalConv3d, "weight")


class TorchaxCausalVideoAutoencoder(interop.JittableModule):

  def __init__(self, vae: CausalVideoAutoencoder):
    super().__init__(vae, extra_jit_args=dict(static_argnames=["split_size", "vae_per_channel_normalize"]))

  def encode(self, media_items: jax.Array, split_size: int = 1, vae_per_channel_normalize: bool = True) -> jax.Array:
    if media_items.ndim != 5:
      raise ValueError(
          f"Expected media_items to have 5 dimensions (batch, channels, frames, height, width), but got {media_items.ndim} dimensions."
      )
    num_frames = media_items.shape[2]
    if (num_frames - 1) % 8 != 0:
      raise ValueError(
          f"Expected media_items to have a number of frames that is 1 + 8 * k for some integer k, but got {num_frames} frames."
      )
    with default_env():
      media_items = interop.torch_view(media_items)

      output = self.functional_call(
          self._vae_encoder_inner,
          params=self.params,
          buffers=self.buffers,
          media_items=media_items,
          split_size=split_size,
          vae_per_channel_normalize=vae_per_channel_normalize,
      )

    return interop.jax_view(output)

  def decode(
      self,
      latents: jax.Array,
      timestep: jax.Array,
      split_size: int = 1,
      vae_per_channel_normalize: bool = True,
      is_video: bool = True,
  ) -> jax.Array:
    with default_env():
      latents = interop.torch_view(latents)
      timestep = interop.torch_view(timestep)
      output = self.functional_call(
          self._vae_decoder_inner,
          params=self.params,
          buffers=self.buffers,
          latents=latents,
          timestep=timestep,
          split_size=split_size,
          vae_per_channel_normalize=vae_per_channel_normalize,
          is_video=is_video,
      )

    return interop.jax_view(output)

  @staticmethod
  def _vae_encoder_inner(model, media_items, split_size, vae_per_channel_normalize):
    return vae_encode(
        media_items=media_items,
        vae=model,
        split_size=split_size,
        vae_per_channel_normalize=vae_per_channel_normalize,
    )

  @staticmethod
  def _vae_decoder_inner(
      model, latents, timestep, is_video: bool = True, split_size: int = 1, vae_per_channel_normalize: bool = False
  ):
    return vae_decode(
        latents=latents,
        vae=model,
        is_video=is_video,
        split_size=split_size,
        vae_per_channel_normalize=vae_per_channel_normalize,
        timestep=timestep,
    )

  @staticmethod
  def normalize_img(image):
    return (image - 128) / 128

  @staticmethod
  def denormalize_img(image):
    return (image * 128 + 128).clip(0, 255)
