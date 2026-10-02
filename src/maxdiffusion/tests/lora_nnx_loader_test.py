# Copyright 2026 Google LLC
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

from types import SimpleNamespace
from unittest import mock

import pytest

from maxdiffusion.loaders.ltx2_lora_nnx_loader import LTX2NNXLoraLoader
from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader


@pytest.fixture(name="wan_lora_mocks")
def fixture_wan_lora_mocks():
  with (
      mock.patch(
          "maxdiffusion.loaders.wan_lora_nnx_loader.StableDiffusionLoraLoaderMixin.lora_state_dict",
          return_value=({"lora.weight": object()}, None),
      ) as state_dict_mock,
      mock.patch(
          "maxdiffusion.loaders.wan_lora_nnx_loader.lora_conversion_utils.preprocess_wan_lora_dict",
          side_effect=lambda state_dict: state_dict,
      ),
      mock.patch("maxdiffusion.loaders.wan_lora_nnx_loader.lora_nnx.merge_lora") as merge_mock,
  ):
    yield state_dict_mock, merge_mock


def test_duplicate_is_skipped_across_loader_instances_but_not_pipelines(wan_lora_mocks):
  state_dict_mock, merge_mock = wan_lora_mocks
  pipeline = SimpleNamespace(transformer=object())

  Wan2_1NNXLoraLoader().load_lora_weights(pipeline, "./checkpoints/lora", "weights.safetensors", rank=4)
  Wan2_1NNXLoraLoader().load_lora_weights(pipeline, "checkpoints/lora", "weights.safetensors", rank=4)

  assert state_dict_mock.call_count == 1
  assert merge_mock.call_count == 1
  assert pipeline.num_fused_loras == 1

  fresh_pipeline = SimpleNamespace(transformer=object())
  Wan2_1NNXLoraLoader().load_lora_weights(fresh_pipeline, "checkpoints/lora", "weights.safetensors", rank=4)

  assert state_dict_mock.call_count == 2
  assert merge_mock.call_count == 2
  assert fresh_pipeline.num_fused_loras == 1


def test_failed_load_is_not_recorded_and_can_be_retried(wan_lora_mocks):
  state_dict_mock, merge_mock = wan_lora_mocks
  state_dict_mock.side_effect = [OSError("temporary download failure"), ({"lora.weight": object()}, None)]
  pipeline = SimpleNamespace(transformer=object())
  loader = Wan2_1NNXLoraLoader()

  with pytest.raises(OSError, match="temporary download failure"):
    loader.load_lora_weights(pipeline, "checkpoints/lora", "weights.safetensors", rank=4)

  assert getattr(pipeline, "_fused_lora_keys", set()) == set()
  assert getattr(pipeline, "num_fused_loras", 0) == 0

  loader.load_lora_weights(pipeline, "checkpoints/lora", "weights.safetensors", rank=4)

  assert state_dict_mock.call_count == 2
  assert merge_mock.call_count == 1
  assert pipeline.num_fused_loras == 1


def test_failed_ltx2_merge_is_not_recorded_and_can_be_retried():
  pipeline = SimpleNamespace(transformer=object())
  loader = LTX2NNXLoraLoader()
  with (
      mock.patch(
          "maxdiffusion.loaders.ltx2_lora_nnx_loader.StableDiffusionLoraLoaderMixin.lora_state_dict",
          return_value=({"diffusion_model.lora.weight": object()}, None),
      ) as state_dict_mock,
      mock.patch(
          "maxdiffusion.loaders.ltx2_lora_nnx_loader.lora_nnx.merge_lora",
          side_effect=[RuntimeError("merge failed"), None],
      ) as merge_mock,
  ):
    with pytest.raises(RuntimeError, match="merge failed"):
      loader.load_lora_weights(pipeline, "checkpoints/lora", "weights.safetensors", rank=4)

    assert getattr(pipeline, "_fused_lora_keys", set()) == set()
    assert getattr(pipeline, "num_fused_loras", 0) == 0

    loader.load_lora_weights(pipeline, "checkpoints/lora", "weights.safetensors", rank=4)

  assert state_dict_mock.call_count == 2
  assert merge_mock.call_count == 2
  assert pipeline.num_fused_loras == 1


def test_in_memory_state_dict_has_a_stable_hashable_key(wan_lora_mocks):
  state_dict_mock, merge_mock = wan_lora_mocks
  lora_state_dict = {"lora.weight": object()}
  pipeline = SimpleNamespace(transformer=object())

  Wan2_1NNXLoraLoader().load_lora_weights(pipeline, lora_state_dict, "weights.safetensors", rank=4)
  Wan2_1NNXLoraLoader().load_lora_weights(pipeline, lora_state_dict, "weights.safetensors", rank=4)

  assert state_dict_mock.call_count == 1
  assert merge_mock.call_count == 1


def test_missing_wan22_weight_name_is_not_recorded(wan_lora_mocks):
  state_dict_mock, merge_mock = wan_lora_mocks
  pipeline = SimpleNamespace(high_noise_transformer=object(), low_noise_transformer=object())

  Wan2_2NNXLoraLoader().load_lora_weights(
      pipeline,
      "checkpoints/lora",
      high_noise_weight_name=None,
      low_noise_weight_name=None,
      rank=4,
  )

  state_dict_mock.assert_not_called()
  merge_mock.assert_not_called()
  assert getattr(pipeline, "_fused_lora_keys", set()) == set()
  assert getattr(pipeline, "num_fused_loras", 0) == 0
