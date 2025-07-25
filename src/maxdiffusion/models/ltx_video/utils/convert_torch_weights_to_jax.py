# Copyright 2025 Lightricks Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Lightricks/LTX-Video/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation is based on the Torch version available at:
# https://github.com/Lightricks/LTX-Video/tree/main
import argparse
import json
from typing import Any, Dict, Optional


import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from safetensors.torch import load_file
import requests
from urllib.parse import urljoin

from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel as JaxTranformer3DModel
from maxdiffusion.models.ltx_video.transformers_pytorch.transformer3d import Transformer3DModel as Transformer3DModel
from maxdiffusion.models.ltx_video.utils.torch_compat import torch_statedict_to_jax

from huggingface_hub import hf_hub_download
import os
import importlib


def download_and_move_files(github_base_url, base_path, target_folder_name, files_to_move, module_to_import):
  """
  Downloads files from a GitHub repo, moves them to a local folder, and then dynamically imports a module.

  Args:
      github_base_url (str): The base URL of the GitHub repo.
      base_path (str): The base path where the new folder will be created.
      target_folder_name (str): The name of the folder to create.
      files_to_move (list): A list of file names to download and move.
      module_to_import (str): The full module path to import.
  """

  target_path = os.path.join(base_path, target_folder_name)

  try:
    # Create the target directory
    os.makedirs(target_path, exist_ok=True)
    print(f"Created directory: {target_path}")

    # Download and move files
    for file_name in files_to_move:
      file_url = urljoin(github_base_url, file_name)
      destination_path = os.path.join(target_path, file_name)

      try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(destination_path, "wb") as f:
          for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

        print(f"Downloaded and moved: {file_name} -> {destination_path}")

      except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_name}: {e}")
      except OSError as e:
        print(f"Error writing file {file_name}: {e}")
    print("Files downloaded and moved successfully.")

    # Verify that the folder exists
    if not os.path.exists(target_path):
      print(f"Error: Target folder {target_path} does not exist after files download.")
    # Dynamically import the module
    try:
      imported_module = importlib.import_module(module_to_import)
      print(f"Module '{module_to_import}' imported successfully.")
      # Access the class
      transformer_class = getattr(imported_module, "Transformer3DModel")
      print(f"Class 'Transformer3DModel' accessed successfully: {transformer_class}")
      return transformer_class
    except ImportError as e:
      print(f"Error importing module '{module_to_import}': {e}")
    except AttributeError as e:
      print(f"Error accessing class 'Transformer3DModel': {e}")

  except OSError as e:
    print(f"Error during file system operation: {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")


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

  print("Loading from HF", flush=True)
  model_name = "Lightricks/LTX-Video"
  absolute_ckpt_path = os.path.abspath(args.ckpt_path)
  local_file_path = hf_hub_download(
      repo_id=model_name,
      filename=weight_file,
      local_dir=absolute_ckpt_path,
      local_dir_use_symlinks=False,
  )
  torch_state_dict = load_file(local_file_path)

  print("Initializing pytorch transformer..", flush=True)
  transformer_config = json.loads(open(args.transformer_config_path, "r").read())
  ignored_keys = ["_class_name", "_diffusers_version", "_name_or_path", "causal_temporal_positioning", "ckpt_path"]
  for key in ignored_keys:
    if key in transformer_config:
      del transformer_config[key]

  transformer = Transformer3DModel.from_config(transformer_config)

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
  relative_ckpt_path = os.path.join(args.ckpt_path, "jax_weights")
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
      "--ckpt_path",
      type=str,
      required=False,
      help="Local path of the checkpoint to convert. If not provided, will download from huggingface for example '/mnt/ckpt/00536000' or '/opt/dmd-torch-model/ema.pt'",
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
