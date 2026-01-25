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

import json
import os

from maxdiffusion import pyconfig
from maxdiffusion.configuration_utils import ConfigMixin
from maxdiffusion import __version__


class DummyConfigMixin(ConfigMixin):
  config_name = "config.json"

  def __init__(self, **kwargs):
    self.register_to_config(**kwargs)


def test_to_json_string_with_config():
  # Load the YAML config file
  config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base_wan_14b.yml")

  # Initialize pyconfig with the YAML config
  pyconfig.initialize([None, config_path], unittest=True)
  config = pyconfig.config

  # Create a DummyConfigMixin instance
  dummy_config = DummyConfigMixin(**config.get_keys())

  # Get the JSON string
  json_string = dummy_config.to_json_string()

  # Parse the JSON string
  parsed_json = json.loads(json_string)

  # Assertions
  assert parsed_json["_class_name"] == "DummyConfigMixin"
  assert parsed_json["_diffusers_version"] == __version__

  # Check a few values from the config
  assert parsed_json["run_name"] == config.run_name
  assert parsed_json["pretrained_model_name_or_path"] == config.pretrained_model_name_or_path
  assert parsed_json["flash_block_sizes"]["block_q"] == config.flash_block_sizes["block_q"]

  # The following keys are explicitly removed in to_json_string, so we assert they are not present
  assert "weights_dtype" not in parsed_json
  assert "precision" not in parsed_json
