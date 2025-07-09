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

import os
import torch
import jax
import numpy as np
import jax.numpy as jnp
import unittest
from absl.testing import absltest
from jax.sharding import Mesh
import json
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel
import functools
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import (
    create_device_mesh,
    setup_initial_state,
    get_memory_allocations,
)
from jax.sharding import PartitionSpec as P
import orbax.checkpoint as ocp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_ref_prediction():
  base_dir = os.path.dirname(__file__)
  saved_prediction_path = os.path.join(base_dir, "ltx_vid_transformer_test_ref_pred")
  predict_dict = torch.load(saved_prediction_path)
  noise_pred_pt = predict_dict["noise_pred"].to(torch.float32)
  return noise_pred_pt


def loop_body(step, args, transformer, fractional_cords, prompt_embeds, segment_ids, encoder_attention_segment_ids):
  latents, state, noise_cond = args
  noise_pred = transformer.apply(
      {"params": state.params},
      hidden_states=latents,
      indices_grid=fractional_cords,
      encoder_hidden_states=prompt_embeds,
      timestep=noise_cond,
      segment_ids=segment_ids,
      encoder_attention_segment_ids=encoder_attention_segment_ids,
  )
  return noise_pred, state, noise_cond


def run_inference(
    states,
    transformer,
    config,
    mesh,
    latents,
    fractional_cords,
    prompt_embeds,
    timestep,
    segment_ids,
    encoder_attention_segment_ids,
):
  transformer_state = states["transformer"]
  loop_body_p = functools.partial(
      loop_body,
      transformer=transformer,
      fractional_cords=fractional_cords,
      prompt_embeds=prompt_embeds,
      segment_ids=segment_ids,
      encoder_attention_segment_ids=encoder_attention_segment_ids,
  )
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    latents, transformer_state, _ = jax.lax.fori_loop(0, 1, loop_body_p, (latents, transformer_state, timestep))
  return latents


class LTXTransformerTest(unittest.TestCase):

  def test_one_step_transformer(self):
    pyconfig.initialize(
        [
            None,
            os.path.join(THIS_DIR, "..", "configs", "ltx_video.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config
    noise_pred_pt = load_ref_prediction()

    # set up transformer
    key = jax.random.PRNGKey(42)
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "../models/ltx_video/xora_v1.2-13B-balanced-128.json")

    with open(config_path, "r") as f:
      model_config = json.load(f)
    relative_ckpt_path = model_config["ckpt_path"]
    ignored_keys = [
        "_class_name",
        "_diffusers_version",
        "_name_or_path",
        "causal_temporal_positioning",
        "in_channels",
        "ckpt_path",
    ]
    in_channels = model_config["in_channels"]
    for name in ignored_keys:
      if name in model_config:
        del model_config[name]

    transformer = Transformer3DModel(
        **model_config, dtype=jnp.float32, gradient_checkpointing="matmul_without_batch", sharding_mesh=mesh
    )
    weights_init_fn = functools.partial(
        transformer.init_weights, in_channels, key, model_config["caption_channels"], eval_only=True
    )

    absolute_ckpt_path = os.path.abspath(relative_ckpt_path)

    checkpoint_manager = ocp.CheckpointManager(absolute_ckpt_path)
    transformer_state, transformer_state_shardings = setup_initial_state(
        model=transformer,
        tx=None,
        config=config,
        mesh=mesh,
        weights_init_fn=weights_init_fn,
        checkpoint_manager=checkpoint_manager,
        checkpoint_item=" ",
        model_params=None,
        training=False,
    )

    transformer_state = jax.device_put(transformer_state, transformer_state_shardings)
    get_memory_allocations()

    states = {}
    state_shardings = {}

    state_shardings["transformer"] = transformer_state_shardings
    states["transformer"] = transformer_state
    example_inputs = {}
    batch_size, num_tokens = 4, 256
    input_shapes = {
        "latents": (batch_size, num_tokens, in_channels),
        "fractional_coords": (batch_size, 3, num_tokens),
        "prompt_embeds": (batch_size, 128, model_config["caption_channels"]),
        "timestep": (batch_size, 256),
        "segment_ids": (batch_size, 256),
        "encoder_attention_segment_ids": (batch_size, 128),
    }
    for name, shape in input_shapes.items():
      example_inputs[name] = jnp.ones(
          shape, dtype=jnp.float32 if name not in ["attention_mask", "encoder_attention_mask"] else jnp.bool
      )

    data_sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))
    latents = jax.device_put(example_inputs["latents"], data_sharding)
    prompt_embeds = jax.device_put(example_inputs["prompt_embeds"], data_sharding)
    fractional_coords = jax.device_put(example_inputs["fractional_coords"], data_sharding)
    noise_cond = jax.device_put(example_inputs["timestep"], data_sharding)
    segment_ids = jax.device_put(example_inputs["segment_ids"], data_sharding)
    encoder_attention_segment_ids = jax.device_put(example_inputs["encoder_attention_segment_ids"], data_sharding)

    p_run_inference = jax.jit(
        functools.partial(
            run_inference,
            transformer=transformer,
            config=config,
            mesh=mesh,
            latents=latents,
            fractional_cords=fractional_coords,
            prompt_embeds=prompt_embeds,
            timestep=noise_cond,
            segment_ids=segment_ids,
            encoder_attention_segment_ids=encoder_attention_segment_ids,
        ),
        in_shardings=(state_shardings,),
        out_shardings=None,
    )

    noise_pred = p_run_inference(states).block_until_ready()
    noise_pred = torch.from_numpy(np.array(noise_pred))

    torch.testing.assert_close(noise_pred_pt, noise_pred, atol=0.025, rtol=20)


if __name__ == "__main__":
  absltest.main()
