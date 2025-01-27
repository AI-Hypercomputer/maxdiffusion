# Adapted from pipeline_flax_stable_diffusion.py
from functools import partial
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from maxdiffusion.models.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.models.embeddings_flax import HFEmbedder
from maxdiffusion.models.ae_flux_nnx import AutoEncoder
from jax.sharding import Sharding
from jax.typing import DTypeLike

import jax
import math
import jax.numpy as jnp
from chex import Array
from einops import rearrange, repeat
from typing import Dict, List, Optional, Union
from ...utils import replace_example_docstring
from flax.core.frozen_dict import FrozenDict
import einops

from ...schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)

# Set to True to use python for loop instead of jax.fori_loop for easier FOR_LOOPging
FOR_LOOP = True

EXAMPLE_DOC_STRING = """
    Examples: COMING SOON
"""


class JfluxPipeline(FlaxDiffusionPipeline):

  def __init__(
      self,
      t5: HFEmbedder,
      clip: HFEmbedder,
      flux: FluxTransformer2DModel,
      ae: AutoEncoder,
      scheduler: Union[FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler],
      dtype: jnp.dtype,
      sharding: Sharding,
  ):
    super().__init__()
    self.dtype = dtype
    self.data_sharding = sharding
    self.register_modules(t5=t5, clip=clip, flux=flux, ae=ae, scheduler=scheduler)

  @staticmethod
  def unpack(x: Array, height: int, width: int) -> Array:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

  @staticmethod
  def create_noise(
      num_samples: int,
      height: int,
      width: int,
      dtype: DTypeLike,
      seed: jax.random.PRNGKey,
  ):
    return jax.random.normal(
        key=seed,
        shape=(num_samples, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16)),
        dtype=dtype,
    )

  # this is the reverse of the unpack function
  @staticmethod
  def pack_img(img):
    bs, c, h, w = img.shape
    return einops.rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

  def prepare_inputs(self, prompt: Union[str, List[str]], img: Array):
    if not isinstance(prompt, (str, list)):
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if isinstance(prompt, str):
      prompt = [prompt]

    bs = len(prompt)
    txt = jax.device_put(jnp.asarray(self.t5(prompt), dtype=jnp.bfloat16), self.data_sharding)

    if txt.shape[0] == 1 and bs > 1:
      txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = jax.device_put(jnp.zeros((bs, txt.shape[1], 3), dtype=txt.dtype), self.data_sharding)

    vec = jax.device_put(self.clip(prompt), self.data_sharding)

    if vec.shape[0] == 1 and bs > 1:
      vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return (txt, txt_ids, vec, img)

  def prepare_img_ids(self, img, guidance_scale):
    img = jax.device_put(img, self.data_sharding)
    batch_size, _, h, w = img.shape
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_ids = jnp.zeros((h // 2, w // 2, 3), dtype=img.dtype)
    img_ids = img_ids.at[..., 1].set(jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].set(jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    img_ids = jax.device_put(img_ids, self.data_sharding)
    guidance_vec = jnp.full((img.shape[0],), guidance_scale, dtype=img.dtype)

    return img, img_ids, guidance_vec

  def _generate(
      self,
      params: Union[Dict, FrozenDict],
      txt: jnp.array,
      txt_ids: jnp.array,
      vec: jnp.array,
      timesteps: jnp.array,
      height: int,
      width: int,
      guidance_scale: float,
      img: Array,
      shift: bool = False,
  ):
    img, img_ids, guidance_vec = self.prepare_img_ids(img, guidance_scale)

    print(f"{len(timesteps) - 1} steps")

    @partial(
        jax.jit,
        in_shardings=(
            self.data_sharding,
            self.data_sharding,
            self.data_sharding,
            self.data_sharding,
            self.data_sharding,
            self.data_sharding,
            None,
            None,
            None,
        ),
        out_shardings=(self.data_sharding),
    )
    def loop_body(params, img, img_ids, txt, txt_ids, vec, guidance_vec, t_curr, t_prev):
      # the order of timesteps is unintuitive...
      t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
      pred = self.flux.apply(
          {"params": params},
          hidden_states=img,
          img_ids=img_ids,
          encoder_hidden_states=txt,
          txt_ids=txt_ids,
          pooled_projections=vec,
          timestep=t_vec,
          guidance=guidance_vec,
      )

      img = img + (t_prev - t_curr) * pred.sample

      return img

    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]

    for i in range(len(timesteps) - 1):
      img = loop_body(params, img, img_ids, txt, txt_ids, vec, guidance_vec, c_ts[i], p_ts[i])

    # decode latents to pixel space
    img = self.unpack(x=img, height=height, width=width)
    img = self.ae.decode(img)
    return img

  @replace_example_docstring(EXAMPLE_DOC_STRING)
  def __call__(
      self,
      params: Union[Dict, FrozenDict],
      txt: jnp.array,
      txt_ids: jnp.array,
      vec: jnp.array,
      timesteps: int,
      height: int,
      width: int,
      guidance_scale: float,
      img: Optional[jnp.ndarray] = None,
      shift: bool = False,
  ):
    r"""
    The call function to the pipeline for generation.

    Args:
      txt: jnp.array,
      txt_ids: jnp.array,
      vec: jnp.array,
      num_inference_steps: int,
      height: int,
      width: int,
      guidance_scale: float,
      img: Optional[jnp.ndarray] = None,
      shift: bool = False,
      jit (`bool`, defaults to `False`):

    Examples:

    """

    if isinstance(timesteps, int):
      timesteps = jnp.linspace(1, 0, timesteps + 1)

    images = self._generate(
        params,
        txt,
        txt_ids,
        vec,
        timesteps,
        height,
        width,
        guidance_scale,
        img,
        shift,
    )

    images = images
    return images

  def init_flux_weights(self, rng: jax.Array, eval_only: bool = False) -> FrozenDict:
    return self.flux.init_weights(rng, eval_only)


@staticmethod
def unshard(x: jnp.ndarray):
  # einops.rearrange(x, 'd b ... -> (d b) ...')
  num_devices, batch_size = x.shape[:2]
  rest = x.shape[2:]
  return x.reshape(num_devices * batch_size, *rest)
