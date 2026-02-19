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

from typing import Optional, Any, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import qwix
from ...pyconfig import HyperParameters
from ... import max_logging

class LTX2Pipeline:
  """
  Pipeline for LTX-2.
  """

  @classmethod
  def get_basic_config(cls, dtype, config: HyperParameters):
    rules = [
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=dtype,
            act_qtype=dtype,
            op_names=("dot_general", "einsum", "conv_general_dilated"),
        )
    ]
    return rules

  @classmethod
  def get_fp8_config(cls, config: HyperParameters):
    """
    fp8 config rules with per-tensor calibration.
    """
    rules = [
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            disable_channelwise_axes=True,  # per_tensor calibration
            weight_calibration_method=config.weight_quantization_calibration_method,
            act_calibration_method=config.act_quantization_calibration_method,
            bwd_calibration_method=config.bwd_quantization_calibration_method,
            op_names=("dot_general", "einsum"),
        ),
        qwix.QtRule(
            module_path=config.qwix_module_path,
            weight_qtype=jnp.float8_e4m3fn,  # conv_general_dilated requires the same dtypes
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e4m3fn,
            disable_channelwise_axes=True,  # per_tensor calibration
            weight_calibration_method=config.weight_quantization_calibration_method,
            act_calibration_method=config.act_quantization_calibration_method,
            bwd_calibration_method=config.bwd_quantization_calibration_method,
            op_names=("conv_general_dilated"),
        ),
    ]
    return rules

  @classmethod
  def get_qt_provider(cls, config: HyperParameters) -> Optional[qwix.QtProvider]:
    """Get quantization rules based on the config."""
    if not getattr(config, "use_qwix_quantization", False):
      return None

    if config.quantization == "int8":
      return qwix.QtProvider(cls.get_basic_config(jnp.int8, config))
    elif config.quantization == "fp8":
      return qwix.QtProvider(cls.get_basic_config(jnp.float8_e4m3fn, config))
    elif config.quantization == "fp8_full":
      return qwix.QtProvider(cls.get_fp8_config(config))
    return None

  @classmethod
  def get_dummy_inputs(cls, config: HyperParameters) -> Tuple[Any, ...]:
    """
    Generates dummy inputs for the LTX-2 transformer.
    Used for quantization tracing.
    """
    batch_size = config.global_batch_size_to_train_on
    # Default shapes observed in LTX-2 code/tests
    # These might need to be adjusted based on actual config if variable
    num_tokens = 256 # Typical token count
    in_channels = 128 # LTX-2 out_channels observed in Transformer3DModel default
    caption_channels = 4096 # T5 encoder output dim typically or similar
    
    # input_shapes mapping from transformer3d.py init_weights:
    # "hidden_states": (batch_size, num_tokens, in_channels)
    # "indices_grid": (batch_size, 3, num_tokens)
    # "encoder_hidden_states": (batch_size, 128, caption_channels)
    # "timestep": (batch_size, 256)
    # "segment_ids": (batch_size, 256)
    # "encoder_attention_segment_ids": (batch_size, 128)
    
    # We construct them in the order of __call__
    # __call__(self, hidden_states, indices_grid, encoder_hidden_states, timestep, class_labels, cross_attention_kwargs, segment_ids, encoder_attention_segment_ids, ...)
    
    hidden_states = jnp.ones((batch_size, num_tokens, in_channels), dtype=jnp.float32)
    indices_grid = jnp.ones((batch_size, 3, num_tokens), dtype=jnp.float32)
    encoder_hidden_states = jnp.ones((batch_size, 128, caption_channels), dtype=jnp.float32)
    timestep = jnp.ones((batch_size, 256), dtype=jnp.float32)
    # class_labels defaults to None
    class_labels = None
    # cross_attention_kwargs defaults to None
    cross_attention_kwargs = None
    segment_ids = jnp.ones((batch_size, 256), dtype=jnp.int32)
    encoder_attention_segment_ids = jnp.ones((batch_size, 128), dtype=jnp.int32)
    
    return (hidden_states, indices_grid, encoder_hidden_states, timestep, class_labels, cross_attention_kwargs, segment_ids, encoder_attention_segment_ids)

  @classmethod
  def quantize_transformer(cls, config: HyperParameters, model: Any, pipeline: "LTX2Pipeline", mesh: Mesh):
    """Quantizes the transformer model."""
    q_rules = cls.get_qt_provider(config)
    if not q_rules:
      return model
    max_logging.log("Quantizing transformer with Qwix.")
    
    model_inputs = cls.get_dummy_inputs(config)
    
    with mesh:
      quantized_model = qwix.quantize_model(model, q_rules, *model_inputs)
    max_logging.log("Qwix Quantization complete.")
    return quantized_model
