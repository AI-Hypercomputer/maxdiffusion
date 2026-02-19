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

import unittest
from unittest.mock import Mock, patch, MagicMock
import jax.numpy as jnp
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
from maxdiffusion.pyconfig import HyperParameters
import qwix

class LTX2QuantizationTest(unittest.TestCase):

  def test_get_qt_provider(self):
    config = Mock(spec=HyperParameters)
    
    # Test disabled
    config.use_qwix_quantization = False
    self.assertIsNone(LTX2Pipeline.get_qt_provider(config))
    
    # Test int8
    config.use_qwix_quantization = True
    config.quantization = "int8"
    config.qwix_module_path = ".*"
    provider = LTX2Pipeline.get_qt_provider(config)
    self.assertIsNotNone(provider)
    
  @patch("maxdiffusion.pipelines.ltx2.ltx2_pipeline.qwix.quantize_model")
  def test_quantize_transformer(self, mock_quantize_model):
    config = Mock(spec=HyperParameters)
    config.use_qwix_quantization = True
    config.quantization = "int8"
    config.qwix_module_path = ".*"
    config.global_batch_size_to_train_on = 1
    
    model = Mock()
    pipeline = Mock()
    mesh = MagicMock()
    mesh.__enter__.return_value = None
    mesh.__exit__.return_value = None
    
    mock_quantized_model = Mock()
    mock_quantize_model.return_value = mock_quantized_model
    
    result = LTX2Pipeline.quantize_transformer(config, model, pipeline, mesh)
    
    self.assertEqual(result, mock_quantized_model)
    mock_quantize_model.assert_called_once()
    
    # Check arguments passed to quantize_model
    args, _ = mock_quantize_model.call_args
    self.assertEqual(args[0], model)
    # args[1] is rules
    # args[2:] are dummy inputs
    self.assertTrue(len(args) > 2)

if __name__ == "__main__":
  unittest.main()
