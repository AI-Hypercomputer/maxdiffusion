"""
 Copyright 2024 Google LLC

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

# pylint: disable=missing-module-docstring
import os
import ast
import json
import sys
from collections import OrderedDict
from typing import Any, Union

import jax
import yaml
from . import max_logging
from . import max_utils
from .models.wan.wan_utils import CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH, WAN_21_FUSION_X_MODEL_NAME_OR_PATH
from maxdiffusion.common_types import LENGTH, KV_LENGTH, WAN2_1, WAN2_2, RING_ATTENTION_AXIS_RULES, SEQUENCE_PARALLEL_AXIS_RULES

_ALLOWED_MODEL_NAMES = {WAN2_1, WAN2_2}
_ALLOWED_TRAINING_MODEL_NAMES = {WAN2_1}

def _validate_model_name(model_name: str | None):
  """Raise if model_name is not in the allowed list."""
  if model_name is None:
    return
  if model_name not in _ALLOWED_MODEL_NAMES:
    raise ValueError(f"Invalid config.model_name '{model_name}'. Allowed values: {sorted(_ALLOWED_MODEL_NAMES)}")

def _validate_training_model_name(model_name: str | None):
  """Raise if model_name is not in the allowed training list."""
  if model_name is None:
    return
  if model_name not in _ALLOWED_TRAINING_MODEL_NAMES:
    raise ValueError(f"Invalid config.model_name '{model_name}' for training. Allowed values: {sorted(_ALLOWED_TRAINING_MODEL_NAMES)}")

def string_to_bool(s: str) -> bool:
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")


def string_to_list(string_list: str) -> list:
  return ast.literal_eval(string_list)


_yaml_types_to_parser = {str: str, int: int, float: float, bool: string_to_bool, list: string_to_list}

_config = None
config = None


def print_system_information():
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.lib.xla_bridge.get_backend().platform_version}")


def _lists_to_tuples(l: list[Any]) -> Union[tuple[Any], list[Any]]:
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l


class _HyperParameters:
  # pylint: disable=missing-class-docstring
  def __init__(self, argv: list[str], **kwargs):
    with open(argv[1], "r", encoding="utf-8") as yaml_file:
      raw_data_from_yaml = yaml.safe_load(yaml_file)
    raw_data_from_cmd_line = self._load_kwargs(argv)

    for k in raw_data_from_cmd_line:
      if k not in raw_data_from_yaml:
        raise ValueError(f"Key {k} was passed at the command line but isn't in config.")

    raw_keys = OrderedDict()
    for k in raw_data_from_yaml:
      # support command line json to dict
      if (
          k in raw_data_from_cmd_line
          and type(raw_data_from_yaml[k]) is dict
          and not isinstance(raw_data_from_cmd_line[k], type(raw_data_from_yaml[k]))
      ):
        raw_data_from_cmd_line[k] = json.loads(raw_data_from_cmd_line[k])

      if (
          k in raw_data_from_cmd_line
          and not isinstance(raw_data_from_cmd_line[k], type(raw_data_from_yaml[k]))
          and type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in {_yaml_types_to_parser.keys()}, can't pass"
            " at the command line"
        )

      if k in raw_data_from_cmd_line and isinstance(raw_data_from_cmd_line[k], type(raw_data_from_yaml[k])):
        raw_keys[k] = raw_data_from_cmd_line[k]  # take the raw data, no type conversion
      elif k in raw_data_from_cmd_line:
        try:
          raw_keys[k] = _yaml_types_to_parser[type(raw_data_from_yaml[k])](
              raw_data_from_cmd_line[k]
          )  # take the command line value, but type it like the config value.
        except ValueError as e:
          raise ValueError(f"Couldn't parse value from command line '{raw_data_from_cmd_line[k]}' for key '{k}'") from e
      else:
        raw_keys[k] = raw_data_from_yaml[k]

    is_unittest = kwargs.get("unittest", False)
    if not is_unittest:
      max_utils.maybe_initialize_jax_distributed_system(raw_keys)

    if raw_keys["jax_cache_dir"]:
      jax.config.update("jax_compilation_cache_dir", raw_keys["jax_cache_dir"])

    _HyperParameters.user_init(raw_keys)
    _HyperParameters.wan_init(raw_keys)
    self.keys = raw_keys
    for k in sorted(raw_keys.keys()):
      max_logging.log(f"Config param {k}: {raw_keys[k]}")

  def _load_kwargs(self, argv: list[str]):
    args_dict = dict(a.split("=", 1) for a in argv[2:])
    return args_dict

  @staticmethod
  def wan_init(raw_keys):
    if not any("layers_per_stage" in inner_tuple for inner_tuple in raw_keys["logical_axis_rules"]):
      raw_keys["logical_axis_rules"] += (("layers_per_stage", None),)
    if "wan_transformer_pretrained_model_name_or_path" in raw_keys:
      transformer_pretrained_model_name_or_path = raw_keys["wan_transformer_pretrained_model_name_or_path"]
      if transformer_pretrained_model_name_or_path == "":
        raw_keys["wan_transformer_pretrained_model_name_or_path"] = raw_keys["pretrained_model_name_or_path"]
      elif (
          transformer_pretrained_model_name_or_path == CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH
          or transformer_pretrained_model_name_or_path == WAN_21_FUSION_X_MODEL_NAME_OR_PATH
      ):
        # Set correct parameters for CausVid in case of user error.
        raw_keys["guidance_scale"] = 1.0
        num_inference_steps = raw_keys["num_inference_steps"]
        if num_inference_steps > 10:
          max_logging.log(
              f"Warning: Try setting num_inference_steps to less than 10 steps when using CausVid, currently you are setting {num_inference_steps} steps."
          )
      else:
        raise ValueError(f"{transformer_pretrained_model_name_or_path} transformer model is not supported for Wan 2.1")
    if "use_qwix_quantization" not in raw_keys:
      raise ValueError("use_qwix_quantization is not set.")
    elif raw_keys["use_qwix_quantization"]:
      if "quantization" not in raw_keys:
        raise ValueError("Quantization type is not set when use_qwix_quantization is enabled.")
      elif raw_keys["quantization"] not in ["int8", "fp8", "fp8_full"]:
        raise ValueError(
            f"Quantization type is not supported when use_qwix_quantization is enabled: {raw_keys['quantization']}"
        )

  @staticmethod
  def calculate_global_batch_sizes(per_device_batch_size):
    num_devices = len(jax.devices())
    if per_device_batch_size < 1:
      # For per_device_batch_size<1, we load the data as if per_device_batch_size=1
      global_batch_size_to_load = num_devices
    else:
      global_batch_size_to_load = int(num_devices * per_device_batch_size)

    global_batch_size_to_train_on = int(num_devices * per_device_batch_size)
    return global_batch_size_to_load, global_batch_size_to_train_on

  @staticmethod
  def user_init(raw_keys):
    """Transformations between the config data and configs used at runtime"""
    raw_keys["weights_dtype"] = jax.numpy.dtype(raw_keys["weights_dtype"])
    raw_keys["activations_dtype"] = jax.numpy.dtype(raw_keys["activations_dtype"])
    if raw_keys["run_name"] == "":
      raw_keys["run_name"] = os.environ.get("JOBSET_NAME")  # using XPK default
    run_name = raw_keys["run_name"]
    base_output_directory = raw_keys["output_dir"]
    if run_name:
      raw_keys["tensorboard_dir"] = os.path.join(base_output_directory, run_name, "tensorboard", "")
      raw_keys["checkpoint_dir"] = os.path.join(base_output_directory, run_name, "checkpoints", "")
      raw_keys["metrics_dir"] = os.path.join(base_output_directory, run_name, "metrics", "")

    max_utils.write_config_raw_keys_for_gcs(raw_keys)

    raw_keys["logical_axis_rules"] = _lists_to_tuples(raw_keys["logical_axis_rules"])
    # Verify qkv is sharded across sequence.
    if "ring" in raw_keys["attention"] or raw_keys["attention_sharding_uniform"]:
      max_logging.log(f"Adding sequence sharding to q and kv if not already present because '{raw_keys['attention']}' contains 'ring' or {raw_keys['attention_sharding_uniform']} is set.")
      logical_axis_rules = list(raw_keys["logical_axis_rules"])
      max_logging.log(f"Initial logical axis rules: {logical_axis_rules}")
      new_rules = []
      q_seq_sharding = (LENGTH, "fsdp")
      kv_seq_sharding = (KV_LENGTH, "fsdp")
      if q_seq_sharding not in logical_axis_rules:
        logical_axis_rules.append(q_seq_sharding)
      if kv_seq_sharding not in logical_axis_rules:
        logical_axis_rules.append(kv_seq_sharding)
      if "ring" in raw_keys["attention"]:
        for ring_attention_axis_rule in RING_ATTENTION_AXIS_RULES:
          if ring_attention_axis_rule not in logical_axis_rules:
            max_logging.log(f"Adding ring attention axis rule {ring_attention_axis_rule}")
            new_rules.append(ring_attention_axis_rule)
      else: # attention contains 'flash' but sequence parallel sharding requested for both self and cross attention
        for seq_parallel_axis_rule in SEQUENCE_PARALLEL_AXIS_RULES:
          if seq_parallel_axis_rule not in logical_axis_rules:
            max_logging.log(f"Adding sequence parallel attention axis rule {seq_parallel_axis_rule}")
            new_rules.append(seq_parallel_axis_rule)
      raw_keys["logical_axis_rules"] = tuple(new_rules) + tuple(logical_axis_rules)
      max_logging.log(f"Final logical axis rules: {raw_keys['logical_axis_rules']}")

    raw_keys["data_sharding"] = _lists_to_tuples(raw_keys["data_sharding"])

    if raw_keys["learning_rate_schedule_steps"] == -1:
      raw_keys["learning_rate_schedule_steps"] = raw_keys["max_train_steps"]

    # Orbax doesn't save the tokenizer params, instead it loads them from the pretrained_model_name_or_path
    raw_keys["tokenizer_model_name_or_path"] = raw_keys["pretrained_model_name_or_path"]
    if "gs://" in raw_keys["pretrained_model_name_or_path"]:
      raw_keys["pretrained_model_name_or_path"] = max_utils.download_blobs(raw_keys["pretrained_model_name_or_path"], "/tmp")
    if "gs://" in raw_keys["unet_checkpoint"]:
      raw_keys["unet_checkpoint"] = max_utils.download_blobs(raw_keys["unet_checkpoint"], "/tmp")
    if "gs://" in raw_keys["tokenizer_model_name_or_path"]:
      raw_keys["tokenizer_model_name_or_path"] = max_utils.download_blobs(raw_keys["tokenizer_model_name_or_path"], "/tmp")
    if "gs://" in raw_keys["dataset_name"]:
      raw_keys["dataset_name"] = max_utils.download_blobs(raw_keys["dataset_name"], raw_keys["dataset_save_location"])
      raw_keys["dataset_save_location"] = raw_keys["dataset_name"]

    if "hf_train_files" in raw_keys and not raw_keys["hf_train_files"]:
      raw_keys["hf_train_files"] = None
    if "hf_access_token" in raw_keys and not raw_keys["hf_access_token"]:
      raw_keys["hf_access_token"] = None

    raw_keys["total_train_batch_size"] = max_utils.get_global_batch_size(raw_keys["per_device_batch_size"])
    raw_keys["num_slices"] = get_num_slices(raw_keys)
    raw_keys["quantization_local_shard_count"] = get_quantization_local_shard_count(raw_keys)
    raw_keys["global_batch_size_to_load"], raw_keys["global_batch_size_to_train_on"] = (
        _HyperParameters.calculate_global_batch_sizes(raw_keys["per_device_batch_size"])
    )

    if getattr(raw_keys, "vae_spatial", -1) == -1 or "vae_spatial" in raw_keys and raw_keys["vae_spatial"] == -1:
      total_device = len(jax.devices())
      dp = raw_keys.get("ici_data_parallelism", 1) * raw_keys.get("dcn_data_parallelism", 1)
      if dp == -1 or dp == 0:
        dp = 1
      raw_keys["vae_spatial"] = (total_device * 2) // dp


def get_num_slices(raw_keys):
  if int(raw_keys["compile_topology_num_slices"]) > 0:
    return raw_keys["compile_topology_num_slices"]
  else:
    devices = jax.devices()
    try:
      return 1 + max([d.slice_index for d in devices])
    except:  # noqa: E722
      return 1


def get_quantization_local_shard_count(raw_keys):
  if raw_keys["quantization_local_shard_count"] == -1:
    return raw_keys["num_slices"]
  else:
    return raw_keys["quantization_local_shard_count"]


def get_num_target_devices(raw_keys):
  return len(jax.devices())


class HyperParameters:  # pylint: disable=missing-class-docstring

  def __init__(self):
    pass

  def __getattr__(self, attr):
    if attr not in _config.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return _config.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return _config.keys


def initialize(argv, **kwargs):
  global _config, config
  _config = _HyperParameters(argv, **kwargs)
  _validate_model_name(_config.keys.get("model_name") if hasattr(_config, "keys") else None)
  if kwargs.get("validate_training", False):
    _validate_training_model_name(_config.keys.get("model_name") if hasattr(_config, "keys") else None)
  config = HyperParameters()


if __name__ == "__main__":
  initialize(sys.argv)
  print(config.steps)
  r = range(config.steps)
