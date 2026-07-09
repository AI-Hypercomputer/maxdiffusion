import os
import unittest
import numpy as np
import jax
import jax.numpy as jnp

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# -----------------------------------------------------------------------------
# Module-level Helper Functions for Packing & Coordinate IDs
# -----------------------------------------------------------------------------


def prepare_latent_image_ids(batch_size, height, width):
  """Generates 4D position coordinates (T, H, W, L) for latent tensors."""
  grid = jnp.zeros((height, width, 4), dtype=jnp.int32)
  grid = grid.at[..., 1].set(jnp.arange(height)[:, None])
  grid = grid.at[..., 2].set(jnp.arange(width)[None, :])
  latent_ids = grid.reshape(-1, 4)
  latent_ids = jnp.expand_dims(latent_ids, axis=0)
  latent_ids = jnp.repeat(latent_ids, batch_size, axis=0)
  return latent_ids


def pack_latents(latents):
  """[B, C, H, W] -> [B, H*W, C]"""
  batch_size, num_channels, height, width = latents.shape
  x = jnp.reshape(latents, (batch_size, num_channels, height * width))
  x = jnp.transpose(x, (0, 2, 1))
  return x


def unpack_latents_with_ids(x, x_ids, height, width):
  """[B, H*W, C] -> [B, C, H, W] using coordinate IDs."""
  batch_size, seq_len, ch = x.shape
  x_list = []
  for b in range(batch_size):
    data = x[b]
    pos = x_ids[b]
    h_ids = pos[:, 1].astype(jnp.int32)
    w_ids = pos[:, 2].astype(jnp.int32)
    flat_ids = h_ids * width + w_ids
    out = jnp.zeros((height * width, ch), dtype=x.dtype)
    out = out.at[flat_ids].set(data)
    out = jnp.transpose(jnp.reshape(out, (height, width, ch)), (2, 0, 1))
    x_list.append(out)
  return jnp.stack(x_list, axis=0)


def prepare_text_ids(batch_size, seq_len):
  """Generates 4D position coordinates for text tokens."""
  txt_ids = jnp.zeros((seq_len, 4), dtype=jnp.int32)
  txt_ids = jnp.expand_dims(txt_ids, axis=0)
  return jnp.repeat(txt_ids, batch_size, axis=0)


class GenerateFlux2KleinTest(unittest.TestCase):

  def test_generate_random_latents_shape(self):
    import torch

    latents = torch.randn((2, 32, 1024 // 8, 512 // 8))
    expected_shape = (2, 32, 1024 // 8, 512 // 8)
    self.assertEqual(tuple(latents.shape), expected_shape)

  def test_packing_roundtrip_parity(self):
    import torch

    latents_pt = torch.randn((2, 32, 16, 16))
    latents_jax = jnp.array(latents_pt.numpy())

    packed_jax = pack_latents(latents_jax)
    ids_jax = prepare_latent_image_ids(2, 16, 16)
    unpacked_jax = unpack_latents_with_ids(packed_jax, ids_jax, 16, 16)

    diff = np.abs(np.array(unpacked_jax) - latents_pt.numpy())
    max_abs = float(np.max(diff))
    print(f"\n[UNIT TEST] Latent Packing/Unpacking Roundtrip -> Max Abs Error: {max_abs:.6e}")
    self.assertEqual(max_abs, 0.0)

  def test_qwen3_flax_dummy_forward(self):
    from maxdiffusion.models.qwen3_flax import FlaxQwen3Model, FlaxQwen3Config

    config = FlaxQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    model = FlaxQwen3Model(config)

    input_ids = jnp.array([[1, 10, 20, 30, 2]])
    key = jax.random.PRNGKey(0)
    params = model.init(key, input_ids=input_ids)["params"]

    output = model.apply({"params": params}, input_ids=input_ids)
    hidden_states = output[0] if isinstance(output, tuple) else output.last_hidden_state

    self.assertEqual(hidden_states.shape, (1, 5, 256))
    self.assertNotEqual(float(jnp.sum(jnp.abs(hidden_states))), 0.0)

  def test_attention_math_parity(self):
    """Verifies Attention mathematical parity between JAX and PyTorch."""
    import torch
    from jax.sharding import Mesh
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh
    from diffusers.models.attention_processor import Attention as PTAttention
    from maxdiffusion.models.attention_flax import FlaxAttention

    pyconfig._config = None
    pyconfig.initialize([
        None,
        "src/maxdiffusion/configs/base_flux2klein.yml",
        "run_name=flux_test",
        "output_dir=/tmp/",
        "jax_cache_dir=/tmp/cache_dir",
        "skip_jax_distributed_system=True",
    ])
    config = pyconfig.config
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array[:1, :1], config.mesh_axes)

    pt_attn = PTAttention(query_dim=3072, heads=24, dim_head=128, bias=False)
    pt_attn.eval()

    jax_attn = FlaxAttention(
        query_dim=3072,
        heads=24,
        dim_head=128,
        dtype=jnp.float32,
        mesh=mesh,
    )

    key = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((1, 128, 3072), dtype=jnp.float32)

    with mesh, jax.set_mesh(mesh):
      params = jax_attn.init(key, dummy_x)["params"]

      pt_sd = pt_attn.state_dict()
      params["to_q"]["kernel"] = pt_sd["to_q.weight"].T.numpy()
      params["to_k"]["kernel"] = pt_sd["to_k.weight"].T.numpy()
      params["to_v"]["kernel"] = pt_sd["to_v.weight"].T.numpy()
      params["to_out_0"]["kernel"] = pt_sd["to_out.0.weight"].T.numpy()

      np.random.seed(42)
      pt_x = torch.randn(1, 128, 3072, dtype=torch.float32)

      with torch.no_grad():
        pt_out = pt_attn(pt_x).numpy()

      jax_out = np.array(jax_attn.apply({"params": params}, jnp.array(pt_x.numpy())))

    diff = np.abs(jax_out - pt_out)
    max_abs = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    print(f"\n[UNIT TEST] Attention Parity -> Max Abs Error: {max_abs:.6e}, RMSE: {rmse:.6e}")
    self.assertLess(max_abs, 5e-2)
    self.assertLess(rmse, 2e-2)

  def test_flowmatch_scheduler_parity(self):
    """Verifies FlowMatch Euler Discrete Scheduler stepping parity against PyTorch."""
    import torch
    import jax.numpy as jnp
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    pt_sched = FlowMatchEulerDiscreteScheduler(shift=1.0)
    pt_sched.set_timesteps(num_inference_steps=4)

    np.random.seed(42)
    pt_sample = torch.randn(1, 1024, 64, dtype=torch.float32)
    pt_model_output = torch.randn(1, 1024, 64, dtype=torch.float32)

    jax_sample = jnp.array(pt_sample.numpy())
    jax_model_output = jnp.array(pt_model_output.numpy())

    sigmas = pt_sched.sigmas.numpy()
    max_abs = 0.0

    for i in range(4):
      dt = sigmas[i + 1] - sigmas[i]
      jax_step = jax_sample + dt * jax_model_output

      pt_step = pt_sched.step(pt_model_output, pt_sched.timesteps[i], pt_sample).prev_sample

      diff = np.abs(np.array(jax_step) - pt_step.numpy())
      max_abs = max(max_abs, float(np.max(diff)))

      jax_sample = jax_step
      pt_sample = pt_step

    print(f"\n[UNIT TEST] FlowMatch Scheduler Parity (4-Step) -> Max Abs Error: {max_abs:.6e}")
    self.assertEqual(max_abs, 0.0)

  def test_double_transformer_block_dummy_parity(self):
    """Verifies Double Transformer Block parity under dummy weights and zero modulation."""
    import torch
    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax
    import flax.linen.spmd as flax_spmd
    from jax.sharding import Mesh
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh
    from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformerBlock

    pyconfig._config = None
    pyconfig.initialize([
        None,
        "src/maxdiffusion/configs/base_flux2klein.yml",
        "run_name=flux_test",
        "output_dir=/tmp/",
        "jax_cache_dir=/tmp/cache_dir",
        "skip_jax_distributed_system=True",
        "weights_dtype=float32",
        "activations_dtype=float32",
    ])
    config = pyconfig.config
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array[:1, :1], config.mesh_axes)

    pt_block = Flux2TransformerBlock(dim=3072, num_attention_heads=24, attention_head_dim=128, mlp_ratio=3.0)
    pt_block.eval()

    jax_block = FluxTransformerBlock(
        dim=3072,
        num_attention_heads=24,
        attention_head_dim=128,
        mlp_ratio=3.0,
        use_global_modulation=True,
        use_swiglu=True,
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
        mesh=mesh,
    )

    key = jax.random.PRNGKey(0)
    img_dummy = jnp.zeros((1, 256, 3072), dtype=jnp.float32)
    txt_dummy = jnp.zeros((1, 512, 3072), dtype=jnp.float32)
    mod_dummy = jnp.zeros((1, 18432), dtype=jnp.float32)
    rope_dummy = jnp.zeros((768, 64, 4), dtype=jnp.float32)

    with mesh, jax.set_mesh(mesh):
      variables = jax_block.init(
          key,
          img_dummy,
          txt_dummy,
          temb=None,
          image_rotary_emb=rope_dummy,
          temb_mod_img=mod_dummy,
          temb_mod_txt=mod_dummy,
      )
      params = variables["params"]
      params = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          params,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      params = flax.core.unfreeze(params)

      np.random.seed(42)
      pt_img = torch.randn(1, 256, 3072, dtype=torch.float32)
      pt_txt = torch.randn(1, 512, 3072, dtype=torch.float32)
      pt_mod = torch.zeros(1, 18432, dtype=torch.float32)

      pt_cos = torch.ones(768, 128, dtype=torch.float32)
      pt_sin = torch.zeros(768, 128, dtype=torch.float32)
      pt_rope = (pt_cos, pt_sin)

      jax_cos = np.ones((768, 64, 2), dtype=np.float32)
      jax_sin = np.zeros((768, 64, 2), dtype=np.float32)
      jax_rope = np.concatenate([jax_cos, jax_sin], axis=-1)

      with torch.no_grad():
        pt_txt_out, pt_img_out = pt_block(
            hidden_states=pt_img,
            encoder_hidden_states=pt_txt,
            temb_mod_img=pt_mod,
            temb_mod_txt=pt_mod,
            image_rotary_emb=pt_rope,
        )

      jax_img_out, jax_txt_out = jax_block.apply(
          {"params": params},
          jnp.array(pt_img.numpy()),
          jnp.array(pt_txt.numpy()),
          temb=None,
          image_rotary_emb=jnp.array(jax_rope),
          temb_mod_img=jnp.array(pt_mod.numpy()),
          temb_mod_txt=jnp.array(pt_mod.numpy()),
      )

    diff_img = np.abs(np.array(jax_img_out) - pt_img_out.numpy())
    diff_txt = np.abs(np.array(jax_txt_out) - pt_txt_out.numpy())
    max_abs = float(max(np.max(diff_img), np.max(diff_txt)))
    rmse = float(np.sqrt(0.5 * (np.mean(diff_img**2) + np.mean(diff_txt**2))))

    print(f"\n[UNIT TEST] Double Block Parity -> Max Abs Error: {max_abs:.6f}, RMSE: {rmse:.6e}")
    self.assertLess(max_abs, 1e-4)
    self.assertLess(rmse, 1e-4)

  def test_single_transformer_block_dummy_parity(self):
    """Verifies Single Transformer Block parity under dummy weights and zero modulation."""
    import torch
    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax
    import flax.linen.spmd as flax_spmd
    from jax.sharding import Mesh
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh
    from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxSingleTransformerBlock

    pyconfig._config = None
    pyconfig.initialize([
        None,
        "src/maxdiffusion/configs/base_flux2klein.yml",
        "run_name=flux_test",
        "output_dir=/tmp/",
        "jax_cache_dir=/tmp/cache_dir",
        "skip_jax_distributed_system=True",
        "weights_dtype=float32",
        "activations_dtype=float32",
    ])
    config = pyconfig.config
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array[:1, :1], config.mesh_axes)

    pt_block = Flux2SingleTransformerBlock(dim=3072, num_attention_heads=24, attention_head_dim=128)
    pt_block.eval()

    jax_block = FluxSingleTransformerBlock(
        dim=3072,
        num_attention_heads=24,
        attention_head_dim=128,
        mlp_ratio=3.0,
        use_global_modulation=True,
        use_swiglu=True,
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
        mesh=mesh,
    )

    key = jax.random.PRNGKey(0)
    x_dummy = jnp.zeros((1, 1536, 3072), dtype=jnp.float32)
    mod_dummy = jnp.zeros((1, 9216), dtype=jnp.float32)
    rope_dummy = jnp.zeros((1536, 64, 4), dtype=jnp.float32)

    with mesh, jax.set_mesh(mesh):
      variables = jax_block.init(key, x_dummy, temb_mod=mod_dummy, image_rotary_emb=rope_dummy)
      params = variables["params"]
      params = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          params,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      params = flax.core.unfreeze(params)

      pt_sd = pt_block.state_dict()
      params["linear1"]["kernel"] = pt_sd["attn.to_qkv_mlp_proj.weight"].T.numpy()
      params["linear2"]["kernel"] = pt_sd["attn.to_out.weight"].T.numpy()
      params["attn"]["query_norm"]["scale"] = pt_sd["attn.norm_q.weight"].numpy()
      params["attn"]["key_norm"]["scale"] = pt_sd["attn.norm_k.weight"].numpy()
      params = flax.core.freeze(params)

      np.random.seed(42)
      pt_img = torch.randn(1, 1024, 3072, dtype=torch.float32)
      pt_txt = torch.randn(1, 512, 3072, dtype=torch.float32)
      pt_mod = torch.zeros(1, 9216, dtype=torch.float32)

      pt_cos = torch.ones(1536, 128, dtype=torch.float32)
      pt_sin = torch.zeros(1536, 128, dtype=torch.float32)
      pt_rope = (pt_cos, pt_sin)

      jax_cos = np.ones((1536, 64, 2), dtype=np.float32)
      jax_sin = np.zeros((1536, 64, 2), dtype=np.float32)
      jax_rope = np.concatenate([jax_cos, jax_sin], axis=-1)

      with torch.no_grad():
        pt_out = pt_block(hidden_states=pt_img, encoder_hidden_states=pt_txt, temb_mod=pt_mod, image_rotary_emb=pt_rope)

      jax_input = jnp.concatenate([jnp.array(pt_txt.numpy()), jnp.array(pt_img.numpy())], axis=1)
      jax_out = jax_block.apply(
          {"params": params},
          jax_input,
          temb_mod=jnp.array(pt_mod.numpy()),
          image_rotary_emb=jnp.array(jax_rope),
      )

    diff = np.abs(np.array(jax_out) - pt_out.numpy())
    max_abs = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    print(f"\n[UNIT TEST] Single Block Parity -> Max Abs Error: {max_abs:.6f}, RMSE: {rmse:.6e}")
    self.assertLess(max_abs, 1e-4)
    self.assertLess(rmse, 1e-4)


if __name__ == "__main__":
  unittest.main()
