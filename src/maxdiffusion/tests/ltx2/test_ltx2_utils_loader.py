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

import concurrent.futures
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import torch
from huggingface_hub.utils import EntryNotFoundError

from maxdiffusion.models.ltx2 import ltx2_utils


class _FakeSafetensorsFile:

  def __init__(self, tensors):
    self._tensors = tensors

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    return False

  def keys(self):
    return self._tensors.keys()

  def get_tensor(self, key):
    return self._tensors[key]


class LTX2TransformerLoaderTest(unittest.TestCase):

  @staticmethod
  def _eval_shapes(num_layers=2, parameter_names=("bias",)):
    return {
        "transformer_blocks": {
            parameter_name: np.zeros((num_layers, 1), dtype=np.float32) for parameter_name in parameter_names
        }
    }

  @staticmethod
  def _bin_download(_model_name, *, filename, **_kwargs):
    if filename.endswith(".json") or filename.endswith(".safetensors"):
      raise EntryNotFoundError("not found")
    return "checkpoint.bin"

  def _load_bin(self, checkpoint, eval_shapes=None, **loader_kwargs):
    torch_load = MagicMock(return_value=checkpoint)
    with (
        patch.object(ltx2_utils.jax, "local_devices", return_value=["cpu"]),
        patch.object(ltx2_utils, "hf_hub_download", side_effect=self._bin_download),
        patch.object(ltx2_utils.torch, "load", torch_load),
    ):
      result = ltx2_utils.load_transformer_weights(
          "test/model",
          eval_shapes or self._eval_shapes(),
          "cpu",
          **loader_kwargs,
      )
    return result, torch_load

  def test_bin_checkpoint_is_loaded_once(self):
    checkpoint = {
        "transformer_blocks.0.bias": torch.tensor([1.0]),
        "transformer_blocks.1.bias": torch.tensor([2.0]),
    }

    result, torch_load = self._load_bin(checkpoint)

    torch_load.assert_called_once_with("checkpoint.bin", map_location="cpu")
    np.testing.assert_array_equal(result["transformer_blocks"]["bias"], np.array([[1.0], [2.0]]))

  def test_scanned_layer_count_is_derived_from_eval_shapes(self):
    checkpoint = {
        "transformer_blocks.0.bias": torch.tensor([1.0]),
        "transformer_blocks.1.bias": torch.tensor([2.0]),
    }

    with self.assertRaisesRegex(ValueError, "num_layers=3 does not match the 2 layers derived from eval_shapes"):
      self._load_bin(checkpoint, num_layers=3)

  def test_missing_scanned_layer_raises(self):
    checkpoint = {"transformer_blocks.0.bias": torch.tensor([1.0])}

    with self.assertRaisesRegex(ValueError, r"Missing scanned layer indices: transformer_blocks\.bias: \[1\]"):
      self._load_bin(checkpoint)

  def test_duplicate_scanned_layer_raises(self):
    checkpoint = {
        "transformer_blocks.0.bias": torch.tensor([1.0]),
        "transformer_blocks.0.alias": torch.tensor([2.0]),
    }
    original_rename = ltx2_utils.rename_for_ltx2_transformer

    def rename_alias(key):
      return original_rename(key).replace(".alias", ".bias")

    with patch.object(ltx2_utils, "rename_for_ltx2_transformer", side_effect=rename_alias):
      with self.assertRaisesRegex(ValueError, r"Duplicate scanned layer index 0 for transformer_blocks\.bias"):
        self._load_bin(checkpoint)

  def test_out_of_range_scanned_layer_raises(self):
    checkpoint = {"transformer_blocks.2.bias": torch.tensor([1.0])}

    with self.assertRaisesRegex(ValueError, r"Scanned layer index 2 for transformer_blocks\.bias is out of range \[0, 2\)"):
      self._load_bin(checkpoint)

  def test_safetensors_loading_remains_chunked_and_parallel(self):
    parameter_names = tuple(f"param_{index}" for index in range(17))
    checkpoint = {
        f"transformer_blocks.{layer_index}.{parameter_name}": torch.tensor([float(layer_index)])
        for parameter_name in parameter_names
        for layer_index in range(2)
    }
    index_data = {
        "weight_map": dict.fromkeys(checkpoint, "checkpoint.safetensors"),
    }

    def download(_model_name, *, filename, **_kwargs):
      if filename.endswith(".json"):
        return "index.json"
      return filename

    safe_open_mock = MagicMock(side_effect=lambda *_args, **_kwargs: _FakeSafetensorsFile(checkpoint))
    executor_type = concurrent.futures.ThreadPoolExecutor
    with (
        patch.object(ltx2_utils.jax, "local_devices", return_value=["cpu"]),
        patch.object(ltx2_utils, "hf_hub_download", side_effect=download),
        patch.object(ltx2_utils, "safe_open", safe_open_mock),
        patch("builtins.open", mock_open(read_data=json.dumps(index_data))),
        patch.object(concurrent.futures, "ThreadPoolExecutor", wraps=executor_type) as executor,
    ):
      result = ltx2_utils.load_transformer_weights(
          "test/model",
          self._eval_shapes(parameter_names=parameter_names),
          "cpu",
      )

    executor.assert_called_once_with()
    self.assertEqual(safe_open_mock.call_count, 3)
    for parameter_name in parameter_names:
      np.testing.assert_array_equal(
          result["transformer_blocks"][parameter_name],
          np.array([[0.0], [1.0]]),
      )

  def test_converted_cache_hit_skips_bin_reload(self):
    checkpoint = {
        "transformer_blocks.0.bias": torch.tensor([1.0]),
        "transformer_blocks.1.bias": torch.tensor([2.0]),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
      checkpoint_path = os.path.join(tmp_dir, "diffusion_pytorch_model.bin")
      with open(checkpoint_path, "wb") as checkpoint_file:
        checkpoint_file.write(b"source identity")

      def download(_model_name, *, filename, **_kwargs):
        if filename.endswith(".json") or filename.endswith(".safetensors"):
          raise EntryNotFoundError("not found")
        return checkpoint_path

      torch_load = MagicMock(return_value=checkpoint)
      loader_kwargs = {
          "cast_dtype_fn": lambda _key: np.dtype(np.float16),
          "converted_cache_dir": os.path.join(tmp_dir, "converted"),
      }
      with (
          patch.object(ltx2_utils.jax, "local_devices", return_value=["cpu"]),
          patch.object(ltx2_utils, "hf_hub_download", side_effect=download),
          patch.object(ltx2_utils.torch, "load", torch_load),
      ):
        first = ltx2_utils.load_transformer_weights(
            "test/ltx2",
            self._eval_shapes(),
            "cpu",
            **loader_kwargs,
        )
        torch_load.assert_called_once_with(checkpoint_path, map_location="cpu")
        torch_load.reset_mock()
        torch_load.side_effect = AssertionError("cache hit must not reload the PyTorch checkpoint")

        second = ltx2_utils.load_transformer_weights(
            "test/ltx2",
            self._eval_shapes(),
            "cpu",
            **loader_kwargs,
        )

      torch_load.assert_not_called()
      self.assertEqual(first["transformer_blocks"]["bias"].dtype, np.dtype(np.float16))
      np.testing.assert_array_equal(second["transformer_blocks"]["bias"], first["transformer_blocks"]["bias"])


if __name__ == "__main__":
  unittest.main()
