import argparse
import json
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from safetensors.torch import load_file

from maxdiffusion.models.ltx_video.transformers_pytorch.transformer_pt import Transformer3DModel_PT
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel as JaxTranformer3DModel
from maxdiffusion.models.ltx_video.utils.torch_compat import torch_statedict_to_jax

from huggingface_hub import hf_hub_download
import os


class Checkpointer:
  """
  Checkpointer - to load and store JAX checkpoints
  """

  STATE_DICT_SHAPE_KEY = "shape"
  STATE_DICT_DTYPE_KEY = "dtype"
  TRAIN_STATE_FILE_NAME = "train_state"

  def __init__(
      self,
      checkpoint_dir: str,
      use_zarr3: bool = False,
      save_buffer_size: Optional[int] = None,
      restore_buffer_size: Optional[int] = None,
  ):
    """
    Constructs the checkpointer object
    """
    opts = ocp.CheckpointManagerOptions(
        enable_async_checkpointing=True,
        step_format_fixed_length=8,  # to make the format of "00000000"
    )
    self.use_zarr3 = use_zarr3
    self.save_buffer_size = save_buffer_size
    self.restore_buffer_size = restore_buffer_size
    registry = ocp.DefaultCheckpointHandlerRegistry()
    self.train_state_handler = ocp.PyTreeCheckpointHandler(
        save_concurrent_gb=save_buffer_size,
        restore_concurrent_gb=restore_buffer_size,
        use_zarr3=use_zarr3,
    )
    registry.add(
        self.TRAIN_STATE_FILE_NAME,
        ocp.args.PyTreeSave,
        self.train_state_handler,
    )
    self.manager = ocp.CheckpointManager(
        directory=checkpoint_dir,
        options=opts,
        handler_registry=registry,
    )

  @property
  def save_buffer_size_bytes(self) -> Optional[int]:
    if self.save_buffer_size is None:
      return None
    return self.save_buffer_size * 2**30

  @staticmethod
  def state_dict_to_structure_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a state dict to a dictionary stating the shape and dtype of the state_dict elements.
    With this, we can reconstruct the state_dict structure later on.
    """
    return jax.tree_util.tree_map(
        lambda t: {
            Checkpointer.STATE_DICT_SHAPE_KEY: tuple(t.shape),
            Checkpointer.STATE_DICT_DTYPE_KEY: t.dtype.name,
        },
        state_dict,
        is_leaf=lambda t: isinstance(t, jax.Array),
    )

  def save(
      self,
      step: int,
      state: train_state.TrainState,
      config: Dict[str, Any],
  ):
    """
    Saves the checkpoint asynchronously

    NOTE that state is going to be copied for this operation

    Args:
        step (int): The step of the checkpoint
        state (TrainStateWithEma): A trainstate containing both the parameters and the optimizer state
        config (Dict[str, Any]): A dictionary containing the configuration of the model
    """
    self.wait()
    args = ocp.args.Composite(
        train_state=ocp.args.PyTreeSave(
            state,
            ocdbt_target_data_file_size=self.save_buffer_size_bytes,
        ),
        config=ocp.args.JsonSave(config),
        meta_params=ocp.args.JsonSave(self.state_dict_to_structure_dict(state.params)),
    )
    self.manager.save(
        step,
        args=args,
    )

  def wait(self):
    """
    Waits for the checkpoint save operation to complete
    """
    self.manager.wait_until_finished()


"""
Convert Torch checkpoints to JAX.

This script loads a Torch checkpoint (either regular or sharded), converts it to Jax weights, and saved it.
"""


def main(args):
  """
  Convert a Torch checkpoint into JAX.
  """

  if args.output_step_num > 1:
    print(
        "⚠️ Warning: The optimizer state is not converted. Changing the output step may lead to a mismatch between "
        "the model parameters and optimizer state. This can affect optimizer moments and may result in a spike in "
        "training loss when resuming from the converted checkpoint."
    )

  print("Loading safetensors, flush = True")
  weight_file = "ltxv-13b-0.9.7-dev.safetensors"

  # download from huggingface, otherwise load from local
  if args.local_ckpt_path is None:
    print("Loading from HF", flush=True)
    model_name = "Lightricks/LTX-Video"
    local_file_path = hf_hub_download(
        repo_id=model_name,
        filename=weight_file,
        local_dir=args.download_ckpt_path,
        local_dir_use_symlinks=False,
    )
  else:
    base_dir = args.local_ckpt_path
    local_file_path = os.path.join(base_dir, weight_file)
  torch_state_dict = load_file(local_file_path)

  print("Initializing pytorch transformer..", flush=True)
  transformer_config = json.loads(open(args.transformer_config_path, "r").read())
  ignored_keys = ["_class_name", "_diffusers_version", "_name_or_path", "causal_temporal_positioning", "ckpt_path"]
  for key in ignored_keys:
    if key in transformer_config:
      del transformer_config[key]

  transformer = Transformer3DModel_PT.from_config(transformer_config)

  print("Loading torch weights into transformer..", flush=True)
  transformer.load_state_dict(torch_state_dict)
  torch_state_dict = transformer.state_dict()

  print("Creating jax transformer with params..", flush=True)
  transformer_config["use_tpu_flash_attention"] = True
  in_channels = transformer_config["in_channels"]
  del transformer_config["in_channels"]
  jax_transformer3d = JaxTranformer3DModel(
      **transformer_config, dtype=jnp.bfloat16, gradient_checkpointing="matmul_without_batch"
  )
  example_inputs = {}
  batch_size, num_tokens = 2, 256
  input_shapes = {
      "hidden_states": (batch_size, num_tokens, in_channels),
      "indices_grid": (batch_size, 3, num_tokens),
      "encoder_hidden_states": (batch_size, 128, transformer_config["caption_channels"]),
      "timestep": (batch_size, 256),
      "segment_ids": (batch_size, 256),
      "encoder_attention_segment_ids": (batch_size, 128),
  }
  for name, shape in input_shapes.items():
    example_inputs[name] = jnp.ones(
        shape, dtype=jnp.float32 if name not in ["attention_mask", "encoder_attention_mask"] else jnp.bool
    )
  params_jax = jax_transformer3d.init(jax.random.PRNGKey(42), **example_inputs)

  print("Converting torch params to jax..", flush=True)
  params_jax = torch_statedict_to_jax(params_jax, torch_state_dict)

  print("Creating checkpointer and jax state for saving..", flush=True)
  relative_ckpt_path = args.output_dir
  absolute_ckpt_path = os.path.abspath(relative_ckpt_path)
  tx = optax.adamw(learning_rate=1e-5)
  with jax.default_device("cpu"):
    state = train_state.TrainState(
        step=args.output_step_num,
        apply_fn=jax_transformer3d.apply,
        params=params_jax,
        tx=tx,
        opt_state=tx.init(params_jax),
    )
  with ocp.CheckpointManager(absolute_ckpt_path) as mngr:
    mngr.save(args.output_step_num, args=ocp.args.StandardSave(state.params))
  print("Done.", flush=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert Torch checkpoints to Jax format.")
  parser.add_argument(
      "--local_ckpt_path",
      type=str,
      required=False,
      help="Local path of the checkpoint to convert. If not provided, will download from huggingface for example '/mnt/ckpt/00536000' or '/opt/dmd-torch-model/ema.pt'",
  )

  parser.add_argument(
      "--download_ckpt_path",
      type=str,
      required=False,
      help="Location to download safetensors from huggingface",
  )

  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Path to save the checkpoint to. for example 'gs://lt-research-mm-europe-west4/jax_trainings/converted-from-torch'",
  )
  parser.add_argument(
      "--output_step_num",
      default=1,
      type=int,
      required=False,
      help=(
          "The step number to assign to the output checkpoint. The result will be saved using this step value. "
          "⚠️ Warning: The optimizer state is not converted. Changing the output step may lead to a mismatch between "
          "the model parameters and optimizer state. This can affect optimizer moments and may result in a spike in "
          "training loss when resuming from the converted checkpoint."
      ),
  )
  parser.add_argument(
      "--transformer_config_path",
      default="/opt/txt2img/txt2img/config/transformer3d/ltxv2B-v1.0.json",
      type=str,
      required=False,
      help="Path to Transformer3D structure config to load the weights based on.",
  )

  args = parser.parse_args()
  main(args)
