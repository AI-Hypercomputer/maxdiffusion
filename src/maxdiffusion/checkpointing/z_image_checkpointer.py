"""
Copyright 2026 Google LLC

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

"""Loading and Orbax caching for the Z-Image pipeline.

All of the pipeline's loading logic lives here: `ZImagePipeline` itself is a
pure runtime object. The Orbax cache holds the whole pipeline -- the denoiser,
the VAE and the Qwen3 text encoder -- so a warm start skips the PyTorch to
Flax conversion entirely.

Every component goes through the same three steps, whichever way it is loaded:

  1. build it abstractly (shapes only, no weights),
  2. resolve its logical annotations against the mesh into shardings,
  3. fill it, either from safetensors or from the Orbax cache, placing each
     parameter directly into its target sharding.

`_nnx_component` does 1 and 2 for the denoiser and the text encoder;
`_linen_component` does the same for the Diffusers VAE, which is still Linen.
"""

import inspect
import json
import os
from typing import Optional

from etils import epath
import flax
from flax import nnx
import flax.linen as nn
from flax.linen import spmd as flax_spmd
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from transformers import AutoConfig, AutoTokenizer

from .. import max_logging, max_utils
from ..models import FlaxAutoencoderKL
from ..models.qwen3_flax import FlaxQwen3Config, NNXFlaxQwen3Model
from ..models.qwen3_utils import load_qwen3_weights
from ..models.z_image.transformer_z_image import ZImageTransformer2DModel
from ..models.z_image.logical_sharding_z_image import get_sharding_specs
from ..models.z_image.z_image_utils import load_z_image_transformer
from ..pipelines.z_image import ZImagePipeline
from .checkpointing_utils import create_orbax_checkpoint_manager


Z_IMAGE_CHECKPOINT = "Z_IMAGE_CHECKPOINT"

CONFIG_ITEM = "z_image_config"
# The denoiser is the only component a Z-Image workflow ever updates. The VAE
# and the text encoder are frozen, so they are saved as separate items and are
# never restored into the transformer's trainable `nnx.Param` collection.
TRAINABLE_ITEMS = ("transformer_state",)
NON_TRAINABLE_ITEMS = ("vae_state", "text_encoder_state")

# Diffusers' Flax VAE config loader drops keys its constructor does not
# declare, so the published shift factor is carried in the cache metadata.
_DEFAULT_VAE_SHIFT_FACTOR = 0.1159


def _plain(tree):
  """Plain dict of plain arrays, which is all Orbax's Standard handler takes."""
  tree = flax.core.unfreeze(tree) if isinstance(tree, flax.core.FrozenDict) else tree
  return jax.tree_util.tree_map(
      lambda leaf: leaf.unbox() if isinstance(leaf, flax_spmd.LogicallyPartitioned) else leaf,
      tree,
      is_leaf=lambda leaf: isinstance(leaf, flax_spmd.LogicallyPartitioned),
  )


def _abstract_tree(shapes, shardings):
  """Nested tree of ShapeDtypeStructs carrying each leaf's target sharding."""
  return unflatten_dict(
      {
          path: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=shardings[path] if shardings else None)
          for path, leaf in flatten_dict(shapes).items()
      }
  )


def _nnx_component(factory, mesh, axis_rules, seed=0):
  """Build an NNX module abstractly: its graph, params and target shardings.

  Shardings come from the module's own `nnx.with_partitioning` annotations,
  resolved against the mesh.
  """
  graphdef, state, rest = nnx.split(nnx.eval_shape(factory, nnx.Rngs(jax.random.key(seed))), nnx.Param, ...)
  shardings = None
  if mesh is not None:
    sharding_tree = nn.logical_to_mesh_sharding(nnx.get_partition_spec(state), mesh, axis_rules)
    shardings = {path: variable.value for path, variable in nnx.to_flat_state(sharding_tree)}
  return graphdef, state, rest, shardings


def _linen_component(module, init_args, mesh, axis_rules):
  """The Linen equivalent of `_nnx_component`, for the Diffusers VAE.

  A module with no logical annotations (`vae_flax.FlaxAutoencoderKL` has none)
  resolves to fully replicated.
  """
  abstract = jax.eval_shape(lambda: module.init(jax.random.key(0), *init_args))
  shapes = _plain(abstract["params"])
  if mesh is None:
    return shapes, None
  mesh_shardings = nn.logical_to_mesh_sharding(nn.get_partition_spec(abstract), mesh, axis_rules)
  return shapes, dict(flatten_dict(_plain(mesh_shardings["params"])))


def _fill(graphdef, state, rest, params):
  """Put loaded params into an abstract NNX state and rebuild the module."""
  flat_state = dict(nnx.to_flat_state(state))
  for path, value in flatten_dict(params).items():
    flat_state[path].value = value
  return nnx.merge(graphdef, nnx.from_flat_state(flat_state), rest)


def _transformer_factory(transformer_config, config, mesh):
  sharding_config = getattr(config, "sharding", {})
  transformer_strategy = sharding_config.get("transformer", "default")
  dit_specs = get_sharding_specs(transformer_strategy, "z_image_dit")

  def factory(rngs):
    return ZImageTransformer2DModel(
        rngs=rngs,
        attention_kernel=config.attention,
        mesh=mesh,
        flash_block_sizes=max_utils.get_flash_block_sizes(config),
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
        sharding_specs=dit_specs,
        **transformer_config,
    )

  return factory


def create_z_image_transformer(model_id: str, config, mesh: Optional[jax.sharding.Mesh] = None):
  """Instantiate and stream a Diffusers Z-Image checkpoint into an NNX model."""
  transformer_config = ZImageTransformer2DModel.load_config(model_id, subfolder="transformer")
  graphdef, state, rest, shardings = _nnx_component(
      _transformer_factory(transformer_config, config, mesh), mesh, config.logical_axis_rules, config.seed
  )
  params = load_z_image_transformer(model_id, state.to_pure_dict(), target_shardings=shardings)
  return _fill(graphdef, state, rest, params)


def _vae_constructor_kwargs(raw_config: dict) -> dict:
  """Keep only the keys FlaxAutoencoderKL declares, so it can be rebuilt offline."""
  accepted = set(inspect.signature(FlaxAutoencoderKL.__init__).parameters) - {"self", "parent", "name"}
  return {key: value for key, value in raw_config.items() if key in accepted and key not in ("dtype", "weights_dtype")}


def _text_encoder_dir(model_id: str) -> str:
  """Local directory holding the text encoder's safetensors shards."""
  if os.path.isdir(model_id):
    return os.path.join(model_id, "text_encoder")
  from huggingface_hub import snapshot_download

  return os.path.join(snapshot_download(model_id, allow_patterns=["text_encoder/*"]), "text_encoder")


class ZImageCheckpointer:
  """Builds a `ZImagePipeline`, using an Orbax cache when one is configured.

  `config.pretrained_orbax_dir` turns the cache on. The first run loads from
  Diffusers/HuggingFace (slow: safetensors are converted to Flax on the host)
  and writes the cache; later runs restore straight into the target shardings.
  """

  def __init__(self, config, mesh: Optional[jax.sharding.Mesh] = None, checkpoint_type: str = Z_IMAGE_CHECKPOINT):
    self.config = config
    self.mesh = mesh
    self.checkpoint_type = checkpoint_type

  @property
  def pretrained_orbax_dir(self) -> str:
    return getattr(self.config, "pretrained_orbax_dir", "")

  def load_pipeline(self) -> ZImagePipeline:
    """Restore the pipeline from the Orbax cache, else load it from Diffusers."""
    orbax_dir = self.pretrained_orbax_dir
    if orbax_dir:
      pipeline = self._restore_pipeline(orbax_dir)
      if pipeline is not None:
        return pipeline

    max_logging.log("Loading Z-Image pipeline from Diffusers.")
    pipeline = self.load_diffusers_pipeline()
    if orbax_dir:
      self._save_pipeline(orbax_dir, pipeline)
    return pipeline

  # ------------------------------------------------------- component builders

  def build_vae(self, vae_config: dict):
    """VAE module, its abstract params and their target shardings."""
    vae = FlaxAutoencoderKL(**vae_config, dtype=self.config.activations_dtype, weights_dtype=self.config.weights_dtype)
    sample = jnp.ones((1, vae.config.in_channels, 64, 64), jnp.float32)
    return vae, *_linen_component(vae, (sample,), self.mesh, self._vae_axis_rules())

  def build_text_encoder(self, text_encoder_config: dict):
    """Qwen3 graph, its abstract params and their target shardings."""
    qwen3_config = FlaxQwen3Config(
        vocab_size=text_encoder_config["vocab_size"],
        hidden_size=text_encoder_config["hidden_size"],
        intermediate_size=text_encoder_config["intermediate_size"],
        num_hidden_layers=text_encoder_config["num_hidden_layers"],
        num_attention_heads=text_encoder_config["num_attention_heads"],
        num_key_value_heads=text_encoder_config["num_key_value_heads"],
        head_dim=text_encoder_config["head_dim"],
        rms_norm_eps=text_encoder_config["rms_norm_eps"],
        rope_theta=text_encoder_config["rope_theta"],
        max_position_embeddings=text_encoder_config["max_position_embeddings"],
        dtype=self.config.weights_dtype,
    )
    # The norms declare float32 scales, so the abstract params already carry the
    # right dtypes for both the loader and the Orbax restore target.
    return _nnx_component(
        lambda rngs: NNXFlaxQwen3Model(rngs=rngs, config=qwen3_config),
        self.mesh,
        self._text_encoder_axis_rules(),
        self.config.seed,
    )

  def _vae_axis_rules(self):
    """`vae_logical_axis_rules` if the config declares them, as WAN does."""
    return tuple(getattr(self.config, "vae_logical_axis_rules", None) or self.config.logical_axis_rules)

  def _text_encoder_axis_rules(self):
    """Qwen3 annotates its embedding table with `vocab`, which Z-Image's rules
    do not name; leaving it unmapped keeps that axis replicated."""
    rules = tuple(self.config.logical_axis_rules)
    if not any(rule[0] == "vocab" for rule in rules):
      rules += (("vocab", None),)
    return rules

  # ------------------------------------------------------------------- load

  def load_diffusers_pipeline(self) -> ZImagePipeline:
    """Load every component from the Diffusers checkpoint, converting to Flax."""
    config = self.config
    model_id = config.pretrained_model_name_or_path
    raw_vae_config = self._raw_vae_config(model_id)

    transformer = create_z_image_transformer(model_id, config, self.mesh)

    vae, _, vae_shardings = self.build_vae(_vae_constructor_kwargs(raw_vae_config))
    # from_pretrained rebuilds the module too, but only its weights are wanted:
    # the module above already carries the shardings resolved from the mesh.
    _, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_id, subfolder="vae", from_pt=True, use_safetensors=True, dtype=config.weights_dtype
    )
    vae_params = self._place(vae_params, vae_shardings)

    graphdef, state, rest, shardings = self.build_text_encoder(
        AutoConfig.from_pretrained(model_id, subfolder="text_encoder").to_dict()
    )
    params = load_qwen3_weights(_text_encoder_dir(model_id), state.to_pure_dict(), target_shardings=shardings)
    text_encoder = _fill(graphdef, state, rest, params)

    return self._assemble(
        transformer,
        vae,
        vae_params,
        AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=True),
        text_encoder,
        raw_vae_config.get("shift_factor", _DEFAULT_VAE_SHIFT_FACTOR),
    )

  def _restore_pipeline(self, orbax_dir: str) -> Optional[ZImagePipeline]:
    try:
      manager = self._checkpoint_manager(orbax_dir)
      step = manager.latest_step()
      if step is None:
        max_logging.log(f"No pretrained orbax checkpoint found in {orbax_dir}")
        return None

      metadata = manager.restore(step, args=ocp.args.Composite(**{CONFIG_ITEM: ocp.args.JsonRestore()}))[CONFIG_ITEM]
      mismatch = self._metadata_mismatch(metadata)
      if mismatch:
        max_logging.log(f"Ignoring orbax checkpoint in {orbax_dir}: {mismatch}")
        return None

      max_logging.log(f"Loading Z-Image pipeline from orbax checkpoint step {step} in {orbax_dir}")
      graphdef, state, rest, shardings = _nnx_component(
          _transformer_factory(metadata["transformer_config"], self.config, self.mesh),
          self.mesh,
          self.config.logical_axis_rules,
          self.config.seed,
      )
      vae, vae_shapes, vae_shardings = self.build_vae(metadata["vae_config"])
      text_graphdef, text_state, text_rest, text_shardings = self.build_text_encoder(metadata["text_encoder_config"])

      # Shapes come from the models themselves, so every item lands directly in
      # the sharding its component was built for -- no host staging.
      restored = manager.restore(
          step,
          args=ocp.args.Composite(
              **{
                  "transformer_state": ocp.args.StandardRestore(_abstract_tree(state.to_pure_dict(), shardings)),
                  "vae_state": ocp.args.StandardRestore(_abstract_tree(vae_shapes, vae_shardings)),
                  "text_encoder_state": ocp.args.StandardRestore(_abstract_tree(text_state.to_pure_dict(), text_shardings)),
              }
          ),
      )

      return self._assemble(
          _fill(graphdef, state, rest, restored["transformer_state"]),
          vae,
          restored["vae_state"],
          self._load_tokenizer(orbax_dir),
          _fill(text_graphdef, text_state, text_rest, restored["text_encoder_state"]),
          metadata["vae_shift_factor"],
      )
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Failed to load orbax checkpoint from {orbax_dir}, falling back to Diffusers: {e}")
      return None

  def _assemble(self, transformer, vae, vae_params, tokenizer, text_encoder, vae_shift_factor):
    return ZImagePipeline(
        transformer,
        vae,
        vae_params,
        tokenizer,
        text_encoder,
        dtype=self.config.activations_dtype,
        mesh=self.mesh,
        logical_axis_rules=self.config.logical_axis_rules,
        vae_shift_factor=vae_shift_factor,
        offload_encoders=getattr(self.config, "offload_encoders", False),
    )

  def _place(self, params, shardings):
    """Move host params onto the mesh in their target shardings."""
    if self.mesh is None or shardings is None:
      return params
    flat = flatten_dict(_plain(params))
    return unflatten_dict({path: jax.device_put(value, shardings[path]) for path, value in flat.items()})

  def _metadata_mismatch(self, metadata) -> str:
    """Cached weights are dtype-cast at save time, so a dtype change must miss."""
    expected = {
        "model_id": self.config.pretrained_model_name_or_path,
        "weights_dtype": str(self.config.weights_dtype),
    }
    for key, value in expected.items():
      if metadata.get(key) != value:
        return f"{key} is {metadata.get(key)!r}, config asks for {value!r}"
    return ""

  def _load_tokenizer(self, orbax_dir: str):
    local = os.path.join(orbax_dir, "tokenizer")
    if epath.Path(local).exists():
      return AutoTokenizer.from_pretrained(local, use_fast=True)
    return AutoTokenizer.from_pretrained(self.config.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=True)

  # ------------------------------------------------------------------- save

  def _save_pipeline(self, orbax_dir: str, pipeline: ZImagePipeline):
    try:
      max_logging.log(f"Saving Z-Image pipeline to orbax at {orbax_dir}")
      manager = self._checkpoint_manager(orbax_dir)
      model_id = self.config.pretrained_model_name_or_path
      _, transformer_state, _ = nnx.split(pipeline.transformer, nnx.Param, ...)
      metadata = {
          "model_id": model_id,
          "weights_dtype": str(self.config.weights_dtype),
          "transformer_config": dict(ZImageTransformer2DModel.load_config(model_id, subfolder="transformer")),
          "vae_config": _vae_constructor_kwargs(self._raw_vae_config(model_id)),
          "vae_shift_factor": pipeline.vae_shift_factor,
          "text_encoder_config": AutoConfig.from_pretrained(model_id, subfolder="text_encoder").to_dict(),
          "trainable_items": list(TRAINABLE_ITEMS),
          "non_trainable_items": list(NON_TRAINABLE_ITEMS),
      }
      manager.save(
          0,
          args=ocp.args.Composite(
              **{
                  "transformer_state": ocp.args.StandardSave(transformer_state.to_pure_dict()),
                  "vae_state": ocp.args.StandardSave(_plain(pipeline.vae_params)),
                  "text_encoder_state": ocp.args.StandardSave(
                      nnx.split(pipeline.text_encoder, nnx.Param, ...)[1].to_pure_dict()
                  ),
                  CONFIG_ITEM: ocp.args.JsonSave(json.loads(json.dumps(metadata, default=str))),
              }
          ),
      )
      manager.wait_until_finished()
      pipeline.tokenizer.save_pretrained(os.path.join(orbax_dir, "tokenizer"))
      max_logging.log(f"Z-Image pipeline saved to {orbax_dir}")
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Failed to save orbax checkpoint to {orbax_dir}: {e}")

  # ---------------------------------------------------------------- helpers

  def _checkpoint_manager(self, directory: str) -> ocp.CheckpointManager:
    return create_orbax_checkpoint_manager(
        directory,
        enable_checkpointing=True,
        save_interval_steps=1,
        checkpoint_type=self.checkpoint_type,
        use_async=False,
    )

  def _raw_vae_config(self, model_id: str) -> dict:
    return dict(FlaxAutoencoderKL.load_config(model_id, subfolder="vae"))
