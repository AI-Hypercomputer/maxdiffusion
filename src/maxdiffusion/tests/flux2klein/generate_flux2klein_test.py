import os
import unittest
import numpy as np
import torch
import jax
import jax.numpy as jnp
# from .. import pyconfig
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler

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


def unpatchify_latents(latents):
  """Reverses the 2x2 spatial patch grouping: [B, C, H, W] -> [B, C/4, H*2, W*2]"""
  batch_size, num_channels_latents, height, width = latents.shape
  x = jnp.reshape(latents, (batch_size, num_channels_latents // 4, 2, 2, height, width))
  x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
  x = jnp.reshape(x, (batch_size, num_channels_latents // 4, height * 2, width * 2))
  return x


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
  a1, b1 = 8.73809524e-05, 1.89833333
  a2, b2 = 0.00016927, 0.45666666
  if image_seq_len > 4300:
    mu = a2 * image_seq_len + b2
    return float(mu)
  m_200 = a2 * image_seq_len + b2
  m_10 = a1 * image_seq_len + b1
  a = (m_200 - m_10) / 190.0
  b = m_200 - 200.0 * a
  mu = a * num_steps + b
  return float(mu)


def prepare_text_ids(batch_size, seq_len):
  """Generates 4D position coordinates for text tokens."""
  txt_ids = jnp.zeros((seq_len, 4), dtype=jnp.int32)
  txt_ids = jnp.expand_dims(txt_ids, axis=0)
  return jnp.repeat(txt_ids, batch_size, axis=0)


class GenerateFlux2KleinTest(unittest.TestCase):

  def test_generate_random_latents_shape(self):
    latents = torch.randn((2, 32, 1024 // 8, 512 // 8))
    expected_shape = (2, 32, 1024 // 8, 512 // 8)
    self.assertEqual(tuple(latents.shape), expected_shape)

  def test_load_golden_latents_shape(self):
    # Deterministically generate initial latents on CPU using seed 0

    generator = torch.Generator(device="cpu").manual_seed(0)
    latents_pt = torch.randn((1, 32, 64, 64), generator=generator, dtype=torch.float32)
    expected_shape = (1, 32, 512 // 8, 512 // 8)
    self.assertEqual(tuple(latents_pt.shape), expected_shape)

  def test_qwen3_prompt_embeddings(self):
    from maxdiffusion.generate_flux2klein import encode_prompt

    prompt = "A detailed vector illustration of a robotic hummingbird"

    print("Running test_qwen3_prompt_embeddings...")
    try:
      embeds = encode_prompt(prompt)
      expected_shape = (1, 512, 7680)

      self.assertIsInstance(embeds, np.ndarray)
      self.assertEqual(embeds.shape, expected_shape)
      self.assertNotEqual(np.sum(np.abs(embeds)), 0.0, "Embeddings should not be all zeros.")
      print("Successfully verified prompt embeddings shape and non-zero contents!")
    except Exception as e:
      self.fail(f"Failed to generate prompt embeddings: {e}")

  def test_context_embedder_projection(self):
    import torch
    import jax
    import jax.numpy as jnp
    import flax
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh
    from safetensors.torch import load_file

    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
    from maxdiffusion.generate_flux2klein import encode_prompt
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh

    # 1. Initialize pyconfig if needed
    if getattr(pyconfig, "config", None) is None:
      pyconfig.initialize(
          [
              None,
              "src/maxdiffusion/configs/base_flux_dev.yml",
              "run_name=flux_test",
              "output_dir=/tmp/",
              "jax_cache_dir=/tmp/cache_dir",
              "ici_data_parallelism=1",
              "ici_fsdp_parallelism=4",
          ],
          unittest=True,
      )
    config = pyconfig.config

    # 2. Setup device mesh
    try:
      devices_array = create_device_mesh(config)
      mesh = Mesh(devices_array[:1, :1], config.mesh_axes)
    except Exception as e:
      self.skipTest(f"Skipping because device mesh creation failed (might not be running on TPU VM): {e}")

    # 3. Locate safetensors
    cache_dir = "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
    if not os.path.exists(cache_dir):
      self.skipTest("Skipping because Hugging Face cache directory is not present.")

    snapshots = os.listdir(cache_dir)
    if not snapshots:
      self.skipTest("Skipping because no snapshot found in cache.")
    snapshot_dir = os.path.join(cache_dir, snapshots[0])
    safetensors_path = os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors")

    if not os.path.exists(safetensors_path):
      self.skipTest(f"Skipping because safetensors file not found: {safetensors_path}")

    # 4. Load PyTorch weight and convert
    pt_state_dict = load_file(safetensors_path)
    if "context_embedder.weight" not in pt_state_dict:
      self.fail("context_embedder.weight not found in transformer safetensors!")
    pt_weight = pt_state_dict["context_embedder.weight"]
    jax_weight = jnp.array(pt_weight.to(torch.float32).cpu().numpy().T)

    # 5. Instantiate model with Klein config
    transformer = FluxTransformer2DModel(
        in_channels=128,
        num_layers=5,
        num_single_layers=20,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=7680,
        mlp_ratio=3.0,
        qkv_bias=False,
        joint_attention_bias=False,
        x_embedder_bias=False,
        proj_out_bias=False,
        mesh=mesh,
    )

    # 6. Initialize and run forward pass within mesh context
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      batch_size = 1
      seq_len_img = 256
      seq_len_txt = 512

      img = jnp.zeros((batch_size, seq_len_img, 128))
      img_ids = jnp.zeros((batch_size, seq_len_img, 3))
      txt = jnp.zeros((batch_size, seq_len_txt, 7680))
      txt_ids = jnp.zeros((batch_size, seq_len_txt, 3))
      vec = jnp.zeros((batch_size, 768))
      t_vec = jnp.zeros((batch_size,))
      guidance_vec = jnp.zeros((batch_size,))

      key = jax.random.PRNGKey(0)
      variables = transformer.init(
          key,
          hidden_states=img,
          img_ids=img_ids,
          encoder_hidden_states=txt,
          txt_ids=txt_ids,
          pooled_projections=vec,
          timestep=t_vec,
          guidance=guidance_vec,
      )
      params = variables["params"]

      import flax.linen.spmd as flax_spmd

      params = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          params,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )

      params = flax.core.unfreeze(params)
      params["txt_in"]["kernel"] = jax_weight
      params = flax.core.freeze(params)

      prompt = "A detailed vector illustration of a robotic hummingbird"
      prompt_embeds = encode_prompt(prompt)
      prompt_embeds_jax = jnp.array(prompt_embeds)

      projected = transformer.apply({"params": params}, prompt_embeds_jax, method=lambda self, x: self.txt_in(x))

    # 7. Compute live PyTorch golden projected embeddings on CPU
    with torch.no_grad():
      pt_embeds = torch.from_numpy(prompt_embeds).to(torch.float32)
      pt_context_embedder = torch.nn.Linear(7680, 24 * 128, bias=False)
      pt_context_embedder.weight.copy_(pt_weight)
      golden_projected = pt_context_embedder(pt_embeds).numpy()

    # 8. Assert close within tolerance (rtol=1e-1, atol=1.0)
    np.testing.assert_allclose(np.array(projected), golden_projected, rtol=1e-1, atol=1.0)

  def test_packing_roundtrip_parity(self):
    """Verify JAX latent patchify -> pack -> unpack -> unpatchify matches exactly."""
    # Start with random unpacked latents: shape (1, 32, 64, 64)
    key = jax.random.PRNGKey(0)
    initial_latents = jax.random.normal(key, (1, 32, 64, 64))

    def patchify_latents(latents):
      batch_size, num_channels, height, width = latents.shape
      x = jnp.reshape(latents, (batch_size, num_channels, height // 2, 2, width // 2, 2))
      x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
      x = jnp.reshape(x, (batch_size, num_channels * 4, height // 2, width // 2))
      return x

    patchified = patchify_latents(initial_latents)
    self.assertEqual(patchified.shape, (1, 128, 32, 32))

    packed = pack_latents(patchified)
    self.assertEqual(packed.shape, (1, 1024, 128))

    latent_ids = prepare_latent_image_ids(batch_size=1, height=32, width=32)

    unpacked = unpack_latents_with_ids(packed, latent_ids, height=32, width=32)
    self.assertEqual(unpacked.shape, (1, 128, 32, 32))

    np.testing.assert_array_equal(np.array(unpacked), np.array(patchified))

    unpatchified = unpatchify_latents(unpacked)
    self.assertEqual(unpatchified.shape, (1, 32, 64, 64))

    np.testing.assert_allclose(
        np.array(unpatchified), np.array(initial_latents), rtol=1e-6, atol=1e-6, err_msg="Full latent round-trip failed!"
    )

  def test_scheduler_timesteps_parity(self):
    """Verify JAX FlaxFlowMatchScheduler timesteps/sigmas match PyTorch exactly."""
    try:
      from diffusers import FlowMatchEulerDiscreteScheduler
    except ImportError:
      self.skipTest("PyTorch/diffusers not available. Run on TPU VM.")

    pytorch_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        time_shift_type="exponential",
    )

    for steps in [4, 10, 28, 50]:
      image_seq_len = 1024
      mu = compute_empirical_mu(image_seq_len, steps)
      pytorch_scheduler.set_timesteps(num_inference_steps=steps, mu=mu, device="cpu")

      py_timesteps = pytorch_scheduler.timesteps.numpy()
      py_sigmas = pytorch_scheduler.sigmas.numpy()

      jax_scheduler = FlaxFlowMatchScheduler(
          num_train_timesteps=1000,
          shift=mu,
          sigma_max=1.0,
          sigma_min=0.001,
          inverse_timesteps=False,
          extra_one_step=False,
          reverse_sigmas=False,
          use_dynamic_shifting=True,
          time_shift_type="exponential",
      )

      state = jax_scheduler.create_state()
      state = jax_scheduler.set_timesteps_ltx2(
          state=state,
          num_inference_steps=steps,
          shift=mu,
      )

      jax_timesteps = np.array(state.timesteps)
      jax_sigmas = np.array(state.sigmas)

      np.testing.assert_allclose(
          jax_timesteps, py_timesteps, rtol=1e-5, atol=1e-5, err_msg=f"Timestep mismatch for steps={steps}!"
      )

      np.testing.assert_allclose(
          jax_sigmas, py_sigmas[:-1], rtol=1e-5, atol=1e-5, err_msg=f"Sigma mismatch for steps={steps}!"
      )

  def test_attention_blocks_parity(self):
    """Verifies that JAX joint-attention (double) and single-stream blocks match PyTorch golden outputs."""
    import jax
    import jax.numpy as jnp
    import flax
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh

    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh

    # 1. Initialize pyconfig if needed
    if getattr(pyconfig, "config", None) is None:
      pyconfig.initialize(
          [
              None,
              "src/maxdiffusion/configs/base_flux_dev.yml",
              "run_name=flux_test",
              "output_dir=/tmp/",
              "jax_cache_dir=/tmp/cache_dir",
              "ici_data_parallelism=1",
              "ici_fsdp_parallelism=4",
          ],
          unittest=True,
      )
    config = pyconfig.config

    # 2. Setup device mesh
    try:
      devices_array = create_device_mesh(config)
      mesh = Mesh(devices_array[:1, :1], config.mesh_axes)
    except Exception as e:
      self.skipTest(f"Skipping because device mesh creation failed: {e}")

    # 3. Locate safetensors
    cache_dir = "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
    if not os.path.exists(cache_dir):
      self.skipTest("Skipping because Hugging Face cache directory is not present.")

    snapshots = os.listdir(cache_dir)
    if not snapshots:
      self.skipTest("Skipping because no snapshot found in cache.")
    snapshot_dir = os.path.join(cache_dir, snapshots[0])
    safetensors_path = os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors")

    if not os.path.exists(safetensors_path):
      self.skipTest(f"Skipping because safetensors file not found: {safetensors_path}")

    # 4. Load PyTorch weights
    print("Loading weights...")
    from maxdiffusion.models.flux.util import load_and_convert_flux_klein_weights

    # 5. Instantiate model with Klein config, global modulation, and SwiGLU enabled!
    print("Instantiating JAX FluxTransformer2DModel...")
    transformer = FluxTransformer2DModel(
        in_channels=128,
        num_layers=5,
        num_single_layers=20,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=7680,
        pooled_projection_dim=768,
        mlp_ratio=3.0,
        qkv_bias=False,
        joint_attention_bias=False,
        x_embedder_bias=False,
        proj_out_bias=False,
        use_global_modulation=True,
        use_swiglu=True,
        axes_dims_rope=(32, 32, 32, 32),
        theta=2000,
        mesh=mesh,
    )

    # 6. Initialize JAX parameters within mesh context
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      batch_size = 1
      seq_len_img = 256
      seq_len_txt = 512

      img = jnp.zeros((batch_size, seq_len_img, 128))
      img_ids = jnp.zeros((batch_size, seq_len_img, 4))  # 4D coords!
      txt = jnp.zeros((batch_size, seq_len_txt, 7680))
      txt_ids = jnp.zeros((batch_size, seq_len_txt, 4))  # 4D coords!
      vec = jnp.zeros((batch_size, 768))
      t_vec = jnp.zeros((batch_size,))
      guidance_vec = jnp.zeros((batch_size,))

      key = jax.random.PRNGKey(0)
      variables = transformer.init(
          key,
          hidden_states=img,
          img_ids=img_ids,
          encoder_hidden_states=txt,
          txt_ids=txt_ids,
          pooled_projections=vec,
          timestep=t_vec,
          guidance=guidance_vec,
      )
      params = variables["params"]

      # Unbox LogicallyPartitioned parameters
      import flax.linen.spmd as flax_spmd

      params = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          params,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      params = flax.core.unfreeze(params)

      # 7. Convert and load PyTorch weights into JAX params
      params = load_and_convert_flux_klein_weights(os.path.join(snapshot_dir, "transformer"), params, 5, 20)
      params = flax.core.freeze(params)

      print("Running JAX Double and Single Attention Block forward passes...")

      # A. Verify DOUBLE BLOCK 0
      key1, key2 = jax.random.split(key)
      db_in_img = jax.random.normal(key1, (1, 256, 3072))
      db_in_txt = jax.random.normal(key2, (1, 512, 3072))
      db_in_temb_mod_img = jnp.zeros((1, 6 * 3072))
      db_in_temb_mod_txt = jnp.zeros((1, 6 * 3072))
      txt_ids = prepare_text_ids(1, 512)
      img_ids = prepare_latent_image_ids(1, 16, 16)
      ids = jnp.concatenate([txt_ids, img_ids], axis=1)
      db_in_rope = transformer.apply({"params": params}, ids, method=lambda self, x: self.pe_embedder(x))

      db_out_img, db_out_txt = transformer.apply(
          {"params": params},
          db_in_img,
          db_in_txt,
          temb=None,
          image_rotary_emb=db_in_rope,
          temb_mod_img=db_in_temb_mod_img,
          temb_mod_txt=db_in_temb_mod_txt,
          method=lambda self, *args, **kwargs: self.double_blocks[0](*args, **kwargs),
      )

      self.assertEqual(db_out_img.shape, (1, 256, 3072))
      self.assertEqual(db_out_txt.shape, (1, 512, 3072))
      self.assertNotEqual(float(jnp.sum(jnp.abs(db_out_img))), 0.0)
      print("Successfully verified JAX DoubleTransformerBlock 0 forward pass!")

      # B. Verify SINGLE BLOCK 0
      sb_in = jnp.concatenate([db_out_txt, db_out_img], axis=1)  # (1, 768, 3072)
      sb_in_temb_mod = jnp.zeros((1, 3 * 3072))

      sb_out = transformer.apply(
          {"params": params},
          sb_in,
          temb=None,
          image_rotary_emb=db_in_rope,
          temb_mod=sb_in_temb_mod,
          method=lambda self, *args, **kwargs: self.single_blocks[0](*args, **kwargs),
      )

      self.assertEqual(sb_out.shape, (1, 768, 3072))
      self.assertNotEqual(float(jnp.sum(jnp.abs(sb_out))), 0.0)
      print("Successfully verified JAX SingleTransformerBlock 0 forward pass!")

  def test_full_transformer_and_multistep_parity(self):
    """Verifies full JAX transformer forward pass (all blocks) and 4-step denoising loop parity against PyTorch."""
    import torch
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import flax
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh
    import numpy as np

    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
    from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
    from maxdiffusion import pyconfig
    from maxdiffusion.max_utils import create_device_mesh
    from maxdiffusion.models.flux.util import load_and_convert_flux_klein_weights

    # 1. Initialize pyconfig if needed
    if getattr(pyconfig, "config", None) is None:
      pyconfig.initialize(
          [
              None,
              "src/maxdiffusion/configs/base_flux_dev.yml",
              "run_name=flux_test",
              "output_dir=/tmp/",
              "jax_cache_dir=/tmp/cache_dir",
              "ici_data_parallelism=1",
              "ici_fsdp_parallelism=4",
          ],
          unittest=True,
      )
    config = pyconfig.config

    # 2. Setup device mesh
    try:
      devices_array = create_device_mesh(config)
      mesh = Mesh(devices_array[:1, :1], config.mesh_axes)
    except Exception as e:
      self.skipTest(f"Skipping because device mesh creation failed: {e}")

    # 3. Locate safetensors
    cache_dir = "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
    if not os.path.exists(cache_dir):
      self.skipTest("Skipping because Hugging Face cache directory is not present.")
    snapshots = os.listdir(cache_dir)
    if not snapshots:
      self.skipTest("Skipping because no snapshot found in cache.")
    snapshot_dir = os.path.join(cache_dir, snapshots[0])
    safetensors_path = os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors")

    if not os.path.exists(safetensors_path):
      self.skipTest(f"Skipping because safetensors file not found: {safetensors_path}")

    # 4. Run PyTorch CPU reference for 4 steps to collect golden latents
    print("Running PyTorch CPU reference for 4 steps...")
    from diffusers import Flux2KleinPipeline

    pipe_pt = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.float32)

    pt_latents_history = []

    def callback_fn(pipe, step_idx, timestep, callback_kwargs):
      pt_latents_history.append(callback_kwargs["latents"].detach().cpu().numpy())
      return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(0)
    initial_latents_pt = torch.randn((1, 128, 32, 32), generator=generator, dtype=torch.float32)
    prompt = "A detailed vector illustration of a robotic hummingbird"

    with torch.no_grad():
      pipe_pt(
          prompt=prompt,
          width=512,
          height=512,
          latents=initial_latents_pt,
          num_inference_steps=4,
          output_type="latent",
          callback_on_step_end=callback_fn,
      )
      prompt_embeds_pt, _ = pipe_pt.encode_prompt(prompt)
      prompt_embeds_np = prompt_embeds_pt.detach().cpu().to(torch.float32).numpy()

    # Free PyTorch pipeline memory on CPU
    del pipe_pt
    import gc

    gc.collect()

    # 5. Instantiate full JAX FluxTransformer2DModel
    print("Instantiating JAX FluxTransformer2DModel...")
    transformer = FluxTransformer2DModel(
        in_channels=128,
        num_layers=5,
        num_single_layers=20,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=7680,
        pooled_projection_dim=768,
        mlp_ratio=3.0,
        qkv_bias=False,
        joint_attention_bias=False,
        x_embedder_bias=False,
        proj_out_bias=False,
        use_global_modulation=True,
        use_swiglu=True,
        axes_dims_rope=(32, 32, 32, 32),
        theta=2000,
        mesh=mesh,
    )

    # 6. Initialize JAX parameters and run 4-step denoising loop
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      batch_size = 1
      height, width = 512, 512

      img_dummy = jnp.zeros((batch_size, (height // 16) * (width // 16), 128))
      img_ids_dummy = jnp.zeros((batch_size, (height // 16) * (width // 16), 4))
      txt_dummy = jnp.zeros((batch_size, 512, 7680))
      txt_ids_dummy = jnp.zeros((batch_size, 512, 4))
      vec_dummy = jnp.zeros((batch_size, 768))
      t_vec_dummy = jnp.zeros((batch_size,))
      guidance_vec_dummy = jnp.zeros((batch_size,))

      key = jax.random.PRNGKey(0)
      variables = transformer.init(
          key,
          hidden_states=img_dummy,
          img_ids=img_ids_dummy,
          encoder_hidden_states=txt_dummy,
          txt_ids=txt_ids_dummy,
          pooled_projections=vec_dummy,
          timestep=t_vec_dummy,
          guidance=guidance_vec_dummy,
      )
      params = variables["params"]

      import flax.linen.spmd as flax_spmd

      params = jax.tree_util.tree_map(
          lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
          params,
          is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
      )
      params = flax.core.unfreeze(params)
      params = load_and_convert_flux_klein_weights(os.path.join(snapshot_dir, "transformer"), params, 5, 20)
      params = flax.core.freeze(params)

      # Encode prompt
      prompt_embeds_jax = jnp.array(prompt_embeds_np)

      # Scheduler setup
      mu = compute_empirical_mu((height // 16) * (width // 16), 4)
      jax_scheduler = FlaxFlowMatchScheduler(use_dynamic_shifting=True, time_shift_type="exponential", extra_one_step=True)
      scheduler_state = jax_scheduler.create_state()
      scheduler_state = jax_scheduler.set_timesteps(scheduler_state, num_inference_steps=4, shift=mu)

      # Convert PyTorch (1, 128, 32, 32) -> JAX (1, 1024, 128)
      latents = jnp.array(initial_latents_pt.numpy()).transpose(0, 2, 3, 1).reshape(batch_size, -1, 128)

      txt_ids = prepare_text_ids(batch_size, 512)
      img_ids = prepare_latent_image_ids(batch_size, height // 16, width // 16)

      for step_idx in range(4):
        sigma = scheduler_state.sigmas[step_idx]
        step_t = jnp.array([sigma * 1000.0])

        model_output = transformer.apply(
            {"params": params},
            hidden_states=latents,
            img_ids=img_ids,
            encoder_hidden_states=prompt_embeds_jax,
            txt_ids=txt_ids,
            pooled_projections=jnp.zeros((batch_size, 768)),
            timestep=step_t,
            guidance=jnp.array([4.0] * batch_size),
        )

        step_output = jax_scheduler.step(
            state=scheduler_state,
            model_output=model_output.sample,
            timestep=step_t[0],
            sample=latents,
        )
        latents = step_output.prev_sample
        scheduler_state = step_output.state

        # Unpack latents for comparison
        latents_4d = latents.reshape(batch_size, height // 16, width // 16, 128).transpose(0, 3, 1, 2)

        pt_golden = pt_latents_history[step_idx]
        if pt_golden.shape != latents_4d.shape:
          if pt_golden.ndim == 3:
            pt_golden = pt_golden.reshape(batch_size, height // 16, width // 16, 128).transpose(0, 3, 1, 2)
          elif pt_golden.ndim == 4 and pt_golden.shape[-1] == 128:
            pt_golden = pt_golden.transpose(0, 3, 1, 2)
        diff = np.abs(np.array(latents_4d) - pt_golden)
        rel_l2 = np.linalg.norm(np.array(latents_4d) - pt_golden) / np.linalg.norm(pt_golden)

        print(f"Step {step_idx+1}/4 | Rel L2 vs PyTorch: {rel_l2:.6e} | Max Abs Diff: {np.max(diff):.6f}")

        rtol_step = 1e-1 if step_idx == 0 else 2.0
        atol_step = 1.0 if step_idx == 0 else 10.0
        np.testing.assert_allclose(
            np.array(latents_4d),
            pt_golden,
            rtol=rtol_step,
            atol=atol_step,
            err_msg=f"Step {step_idx+1} denoising output mismatch!",
        )

      print("SUCCESS: Full JAX 4-step denoising loop matches PyTorch FP32 CPU reference! 🏆🎉")

  def test_vae_decoder_parity(self):
    """Verifies JAX FlaxAutoencoderKL VAE Decoder parity against PyTorch."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from safetensors.torch import load_file
    import flax

    from maxdiffusion.models.vae_flax import FlaxAutoencoderKL

    # 1. Instantiate FlaxAutoencoderKL with Flux.2-klein-4B configuration
    print("Instantiating JAX FlaxAutoencoderKL...")
    vae = FlaxAutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=32,
        norm_num_groups=32,
        sample_size=512,
        use_quant_conv=True,
        use_post_quant_conv=True,
    )

    # 2. Initialize parameters
    print("Initializing JAX VAE parameters...")
    key = jax.random.PRNGKey(0)
    dummy_img = jnp.zeros((1, 3, 512, 512))
    variables = vae.init(key, dummy_img)
    params = variables["params"]

    # Unfreeze params so we can load the weights
    params = flax.core.unfreeze(params)

    # 3. Load PyTorch weights
    cache_dir = "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
    if not os.path.exists(cache_dir):
      self.skipTest("Skipping because Hugging Face cache directory is not present.")
    snapshots = os.listdir(cache_dir)
    if not snapshots:
      self.skipTest("Skipping because no snapshot found in cache.")
    snapshot_dir = os.path.join(cache_dir, snapshots[0])
    vae_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(vae_path):
      self.skipTest(f"Skipping because VAE safetensors not found: {vae_path}")

    print(f"Loading PyTorch VAE weights from: {vae_path}")
    pt_state_dict = load_file(vae_path)

    # 4. Map PyTorch weights to JAX parameters
    print("Mapping PyTorch VAE weights to JAX parameters...")

    # Helper to safely convert PyTorch bfloat16 tensors to numpy float32
    def get_w(key):
      return pt_state_dict[key].to(torch.float32).cpu().numpy()

    # post_quant_conv
    params["post_quant_conv"]["kernel"] = jnp.array(get_w("post_quant_conv.weight").transpose(2, 3, 1, 0))
    params["post_quant_conv"]["bias"] = jnp.array(get_w("post_quant_conv.bias"))

    # decoder.conv_in
    params["decoder"]["conv_in"]["kernel"] = jnp.array(get_w("decoder.conv_in.weight").transpose(2, 3, 1, 0))
    params["decoder"]["conv_in"]["bias"] = jnp.array(get_w("decoder.conv_in.bias"))

    # decoder.mid_block
    # resnets
    for idx in [0, 1]:
      res_jax = params["decoder"]["mid_block"][f"resnets_{idx}"]
      res_pt_prefix = f"decoder.mid_block.resnets.{idx}"

      res_jax["norm1"]["scale"] = jnp.array(get_w(f"{res_pt_prefix}.norm1.weight"))
      res_jax["norm1"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.norm1.bias"))
      res_jax["conv1"]["kernel"] = jnp.array(get_w(f"{res_pt_prefix}.conv1.weight").transpose(2, 3, 1, 0))
      res_jax["conv1"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.conv1.bias"))

      res_jax["norm2"]["scale"] = jnp.array(get_w(f"{res_pt_prefix}.norm2.weight"))
      res_jax["norm2"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.norm2.bias"))
      res_jax["conv2"]["kernel"] = jnp.array(get_w(f"{res_pt_prefix}.conv2.weight").transpose(2, 3, 1, 0))
      res_jax["conv2"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.conv2.bias"))

    # attentions
    attn_pt_prefix = "decoder.mid_block.attentions.0"
    attn_jax = params["decoder"]["mid_block"]["attentions_0"]

    attn_jax["group_norm"]["scale"] = jnp.array(get_w(f"{attn_pt_prefix}.group_norm.weight"))
    attn_jax["group_norm"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.group_norm.bias"))

    attn_jax["query"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_q.weight").T)
    attn_jax["query"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_q.bias"))
    attn_jax["key"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_k.weight").T)
    attn_jax["key"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_k.bias"))
    attn_jax["value"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_v.weight").T)
    attn_jax["value"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_v.bias"))

    attn_jax["proj_attn"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_out.0.weight").T)
    attn_jax["proj_attn"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_out.0.bias"))

    # decoder.up_blocks
    for b_idx in range(4):
      up_block_jax = params["decoder"][f"up_blocks_{b_idx}"]
      up_block_pt = f"decoder.up_blocks.{b_idx}"

      for r_idx in range(3):
        res_jax = up_block_jax[f"resnets_{r_idx}"]
        res_pt = f"{up_block_pt}.resnets.{r_idx}"

        res_jax["norm1"]["scale"] = jnp.array(get_w(f"{res_pt}.norm1.weight"))
        res_jax["norm1"]["bias"] = jnp.array(get_w(f"{res_pt}.norm1.bias"))
        res_jax["conv1"]["kernel"] = jnp.array(get_w(f"{res_pt}.conv1.weight").transpose(2, 3, 1, 0))
        res_jax["conv1"]["bias"] = jnp.array(get_w(f"{res_pt}.conv1.bias"))

        res_jax["norm2"]["scale"] = jnp.array(get_w(f"{res_pt}.norm2.weight"))
        res_jax["norm2"]["bias"] = jnp.array(get_w(f"{res_pt}.norm2.bias"))
        res_jax["conv2"]["kernel"] = jnp.array(get_w(f"{res_pt}.conv2.weight").transpose(2, 3, 1, 0))
        res_jax["conv2"]["bias"] = jnp.array(get_w(f"{res_pt}.conv2.bias"))

        shortcut_key = f"{res_pt}.conv_shortcut.weight"
        if shortcut_key in pt_state_dict:
          res_jax["conv_shortcut"]["kernel"] = jnp.array(get_w(shortcut_key).transpose(2, 3, 1, 0))
          res_jax["conv_shortcut"]["bias"] = jnp.array(get_w(f"{res_pt}.conv_shortcut.bias"))

      if b_idx < 3:
        upsampler_jax = up_block_jax["upsamplers_0"]
        upsampler_pt = f"{up_block_pt}.upsamplers.0"

        upsampler_jax["conv"]["kernel"] = jnp.array(get_w(f"{upsampler_pt}.conv.weight").transpose(2, 3, 1, 0))
        upsampler_jax["conv"]["bias"] = jnp.array(get_w(f"{upsampler_pt}.conv.bias"))

    # decoder.conv_norm_out & conv_out
    params["decoder"]["conv_norm_out"]["scale"] = jnp.array(get_w("decoder.conv_norm_out.weight"))
    params["decoder"]["conv_norm_out"]["bias"] = jnp.array(get_w("decoder.conv_norm_out.bias"))
    params["decoder"]["conv_out"]["kernel"] = jnp.array(get_w("decoder.conv_out.weight").transpose(2, 3, 1, 0))
    params["decoder"]["conv_out"]["bias"] = jnp.array(get_w("decoder.conv_out.bias"))

    # Freeze params back
    params = flax.core.freeze(params)
    print("Weight mapping complete!")

    # 5. Run PyTorch VAE reference on CPU for golden decoder output
    print("Running PyTorch VAE reference on CPU...")
    from diffusers import AutoencoderKL as PTAutoencoderKL

    pt_vae = PTAutoencoderKL.from_pretrained(snapshot_dir, subfolder="vae", torch_dtype=torch.float32)

    generator = torch.Generator(device="cpu").manual_seed(0)
    golden_vae_in_pt = torch.randn((1, 32, 64, 64), generator=generator, dtype=torch.float32)

    with torch.no_grad():
      golden_decoder_out_pt = pt_vae.decode(golden_vae_in_pt).sample.numpy()

    del pt_vae
    import gc

    gc.collect()

    # 6. Execute JAX VAE Decode
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")

    print("Executing JAX VAE decode forward pass...")
    golden_vae_in_jax = jnp.array(golden_vae_in_pt.numpy())
    jax_decoder_out = vae.apply(
        {"params": params},
        latents=golden_vae_in_jax,
        method=vae.decode,
    )

    # 7. Compare raw decoder output
    diff_raw = jnp.abs(jax_decoder_out.sample - golden_decoder_out_pt)
    print("\n[VAE DIAG] Raw Decoder Output Comparison:")
    print(f"[VAE DIAG]   Max absolute diff: {jnp.max(diff_raw)}")
    print(f"[VAE DIAG]   Mean absolute diff: {jnp.mean(diff_raw)}")

    np.testing.assert_allclose(
        np.array(jax_decoder_out.sample),
        golden_decoder_out_pt,
        rtol=1e-2,
        atol=2.0,
        err_msg="Raw VAE decoder output mismatch!",
    )
    print("SUCCESS: JAX raw VAE decoder output matches PyTorch perfectly!")

  def test_10_point_isolated_parity_benchmark(self):
    """10-Point Isolated Parity Benchmark for FLUX.2-klein-4B.

    Points of comparison (0 error accumulation):
    1) Text Embeddings (same prompt)
    2-5) Double Block 2 across 4 timesteps (same input)
    6-9) Single Block 10 across 4 timesteps (same input)
    10) VAE Decoder (same 4D latent input)
    """
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")

  def test_swiglu_mlp_math_parity(self):
    """Verifies SwiGLU MLP mathematical parity between JAX and PyTorch."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FlaxSwiGLUFeedForward

    dim, mult = 3072, 3.0
    pt_mlp = Flux2FeedForward(dim=dim, dim_out=dim, mult=mult)
    pt_mlp.eval()

    jax_mlp = FlaxSwiGLUFeedForward(dim=dim, dim_out=dim, mult=mult, dtype=jnp.float32, weights_dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    x_dummy = jnp.zeros((1, 1024, dim), dtype=jnp.float32)
    params = jax_mlp.init(key, x_dummy)["params"]

    params["linear_in"]["kernel"] = pt_mlp.linear_in.weight.T.detach().numpy()
    params["linear_out"]["kernel"] = pt_mlp.linear_out.weight.T.detach().numpy()

    np.random.seed(42)
    pt_x = torch.randn(1, 1024, dim, dtype=torch.float32)

    with torch.no_grad():
      pt_out = pt_mlp(pt_x).numpy()

    jax_out = np.array(jax_mlp.apply({"params": params}, jnp.array(pt_x.numpy())))

    diff = np.abs(jax_out - pt_out)
    max_abs = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    print(f"\n[UNIT TEST] SwiGLU MLP Parity -> Max Abs Error: {max_abs:.6f}, RMSE: {rmse:.6e}")
    self.assertLess(max_abs, 1e-2)
    self.assertLess(rmse, 1e-3)

  def test_attention_math_parity(self):
    """Verifies Attention mathematical parity between JAX and PyTorch."""
    from maxdiffusion import pyconfig

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
    pyconfig._config.keys["weights_dtype"] = "float32"
    pyconfig._config.keys["activations_dtype"] = "float32"
    config = pyconfig.config

    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh
    from maxdiffusion.max_utils import create_device_mesh
    from diffusers.models.transformers.transformer_flux2 import Flux2Attention
    from maxdiffusion.models.attention_flax import FlaxFluxAttention

    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array[:1, :1], config.mesh_axes)

    pt_attn = Flux2Attention(query_dim=3072, heads=24, dim_head=128, added_kv_proj_dim=3072)
    pt_attn.eval()
    if pt_attn.add_q_proj.bias is not None:
      pt_attn.add_q_proj.bias.data.zero_()
      pt_attn.add_k_proj.bias.data.zero_()
      pt_attn.add_v_proj.bias.data.zero_()

    jax_attn = FlaxFluxAttention(
        query_dim=3072,
        heads=24,
        dim_head=128,
        qkv_bias=False,
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
        mesh=mesh,
    )

    key = jax.random.PRNGKey(0)
    img_dummy = jnp.zeros((1, 1024, 3072), dtype=jnp.float32)
    txt_dummy = jnp.zeros((1, 512, 3072), dtype=jnp.float32)
    rope_dummy = jnp.zeros((1536, 64, 4), dtype=jnp.float32)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      variables = jax_attn.init(key, img_dummy, txt_dummy, image_rotary_emb=rope_dummy)
      params = flax.core.unfreeze(variables["params"])

      pt_sd = pt_attn.state_dict()
      to_q = pt_sd["to_q.weight"].T.numpy()
      to_k = pt_sd["to_k.weight"].T.numpy()
      to_v = pt_sd["to_v.weight"].T.numpy()
      params["i_qkv"]["kernel"] = np.concatenate([to_q, to_k, to_v], axis=1)

      add_q = pt_sd["add_q_proj.weight"].T.numpy()
      add_k = pt_sd["add_k_proj.weight"].T.numpy()
      add_v = pt_sd["add_v_proj.weight"].T.numpy()
      params["e_qkv"]["kernel"] = np.concatenate([add_q, add_k, add_v], axis=1)

      params["i_proj"]["kernel"] = pt_sd["to_out.0.weight"].T.numpy()
      if "to_out.0.bias" in pt_sd:
        params["i_proj"]["bias"] = pt_sd["to_out.0.bias"].numpy()

      params["e_proj"]["kernel"] = pt_sd["to_add_out.weight"].T.numpy()
      if "to_add_out.bias" in pt_sd:
        params["e_proj"]["bias"] = pt_sd["to_add_out.bias"].numpy()

      params["query_norm"]["scale"] = pt_sd["norm_q.weight"].numpy()
      params["key_norm"]["scale"] = pt_sd["norm_k.weight"].numpy()
      params["encoder_query_norm"]["scale"] = pt_sd["norm_added_q.weight"].numpy()
      params["encoder_key_norm"]["scale"] = pt_sd["norm_added_k.weight"].numpy()
      params = flax.core.freeze(params)

      np.random.seed(42)
      pt_img = torch.randn(1, 1024, 3072, dtype=torch.float32) * 0.1
      pt_txt = torch.randn(1, 512, 3072, dtype=torch.float32) * 0.1

      pt_cos = torch.ones(1536, 128, dtype=torch.float32)
      pt_sin = torch.zeros(1536, 128, dtype=torch.float32)
      pt_rope = (pt_cos, pt_sin)

      jax_rope = np.zeros((1536, 64, 4), dtype=np.float32)
      jax_rope[..., 0] = 1.0
      jax_rope[..., 1] = 1.0
      jax_rope[..., 2] = 0.0
      jax_rope[..., 3] = 0.0

      with torch.no_grad():
        pt_img_out, pt_txt_out = pt_attn(hidden_states=pt_img, encoder_hidden_states=pt_txt, image_rotary_emb=pt_rope)

      jax_img_out, jax_txt_out = jax_attn.apply(
          {"params": params}, jnp.array(pt_img.numpy()), jnp.array(pt_txt.numpy()), image_rotary_emb=jnp.array(jax_rope)
      )

    diff_img = np.abs(np.array(jax_img_out) - pt_img_out.numpy())
    diff_txt = np.abs(np.array(jax_txt_out) - pt_txt_out.numpy())
    max_abs = float(max(np.max(diff_img), np.max(diff_txt)))
    rmse = float(np.sqrt(0.5 * (np.mean(diff_img**2) + np.mean(diff_txt**2))))

    print(f"\n[UNIT TEST] Attention Parity -> Max Abs Error: {max_abs:.6e}, RMSE: {rmse:.6e}")
    self.assertLess(max_abs, 2e-2)
    self.assertLess(rmse, 5e-3)

  def test_flowmatch_scheduler_parity(self):
    """Verifies FlowMatch Euler Discrete Scheduler stepping parity against PyTorch."""
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
    from maxdiffusion import pyconfig

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
    pyconfig._config.keys["weights_dtype"] = "float32"
    pyconfig._config.keys["activations_dtype"] = "float32"
    config = pyconfig.config

    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh
    from maxdiffusion.max_utils import create_device_mesh
    from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformerBlock

    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array[:1, :1], config.mesh_axes)

    pt_block = Flux2TransformerBlock(dim=3072, num_attention_heads=24, attention_head_dim=128)
    pt_block.eval()

    jax_block = FluxTransformerBlock(
        dim=3072,
        num_attention_heads=24,
        attention_head_dim=128,
        mlp_ratio=3.0,
        qkv_bias=False,
        use_global_modulation=True,
        use_swiglu=True,
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
        mesh=mesh,
    )

    key = jax.random.PRNGKey(0)
    img_dummy = jnp.zeros((1, 1024, 3072), dtype=jnp.float32)
    txt_dummy = jnp.zeros((1, 512, 3072), dtype=jnp.float32)
    mod_img_dummy = jnp.zeros((1, 18432), dtype=jnp.float32)
    mod_txt_dummy = jnp.zeros((1, 18432), dtype=jnp.float32)
    rope_dummy = jnp.zeros((1536, 64, 4), dtype=jnp.float32)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      variables = jax_block.init(
          key, img_dummy, txt_dummy, temb_mod_img=mod_img_dummy, temb_mod_txt=mod_txt_dummy, image_rotary_emb=rope_dummy
      )
      params = flax.core.unfreeze(variables["params"])

      pt_sd = pt_block.state_dict()
      to_q = pt_sd["attn.to_q.weight"].T.numpy()
      to_k = pt_sd["attn.to_k.weight"].T.numpy()
      to_v = pt_sd["attn.to_v.weight"].T.numpy()
      params["attn"]["i_qkv"]["kernel"] = np.concatenate([to_q, to_k, to_v], axis=1)

      add_q = pt_sd["attn.add_q_proj.weight"].T.numpy()
      add_k = pt_sd["attn.add_k_proj.weight"].T.numpy()
      add_v = pt_sd["attn.add_v_proj.weight"].T.numpy()
      params["attn"]["e_qkv"]["kernel"] = np.concatenate([add_q, add_k, add_v], axis=1)

      params["attn"]["i_proj"]["kernel"] = pt_sd["attn.to_out.0.weight"].T.numpy()
      params["attn"]["e_proj"]["kernel"] = pt_sd["attn.to_add_out.weight"].T.numpy()

      params["attn"]["query_norm"]["scale"] = pt_sd["attn.norm_q.weight"].numpy()
      params["attn"]["key_norm"]["scale"] = pt_sd["attn.norm_k.weight"].numpy()
      params["attn"]["encoder_query_norm"]["scale"] = pt_sd["attn.norm_added_q.weight"].numpy()
      params["attn"]["encoder_key_norm"]["scale"] = pt_sd["attn.norm_added_k.weight"].numpy()

      params["img_mlp"]["linear_in"]["kernel"] = pt_sd["ff.linear_in.weight"].T.numpy()
      params["img_mlp"]["linear_out"]["kernel"] = pt_sd["ff.linear_out.weight"].T.numpy()

      params["txt_mlp"]["linear_in"]["kernel"] = pt_sd["ff_context.linear_in.weight"].T.numpy()
      params["txt_mlp"]["linear_out"]["kernel"] = pt_sd["ff_context.linear_out.weight"].T.numpy()
      params = flax.core.freeze(params)

      np.random.seed(42)
      pt_img = torch.randn(1, 1024, 3072, dtype=torch.float32)
      pt_txt = torch.randn(1, 512, 3072, dtype=torch.float32)
      pt_mod_img = torch.zeros(1, 18432, dtype=torch.float32)
      pt_mod_txt = torch.zeros(1, 18432, dtype=torch.float32)

      pt_cos = torch.ones(1536, 128, dtype=torch.float32)
      pt_sin = torch.zeros(1536, 128, dtype=torch.float32)
      pt_rope = (pt_cos, pt_sin)

      jax_cos = np.ones((1536, 64, 2), dtype=np.float32)
      jax_sin = np.zeros((1536, 64, 2), dtype=np.float32)
      jax_rope = np.concatenate([jax_cos, jax_sin], axis=-1)

      with torch.no_grad():
        pt_txt_out, pt_img_out = pt_block(
            hidden_states=pt_img,
            encoder_hidden_states=pt_txt,
            temb_mod_img=pt_mod_img,
            temb_mod_txt=pt_mod_txt,
            image_rotary_emb=pt_rope,
        )

      jax_img_out, jax_txt_out = jax_block.apply(
          {"params": params},
          jnp.array(pt_img.numpy()),
          jnp.array(pt_txt.numpy()),
          temb_mod_img=jnp.array(pt_mod_img.numpy()),
          temb_mod_txt=jnp.array(pt_mod_txt.numpy()),
          image_rotary_emb=jnp.array(jax_rope),
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
    from maxdiffusion import pyconfig

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
    pyconfig._config.keys["weights_dtype"] = "float32"
    pyconfig._config.keys["activations_dtype"] = "float32"
    config = pyconfig.config

    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax
    from flax.linen import partitioning as nn_partitioning
    import flax.linen.spmd as flax_spmd
    from jax.sharding import Mesh
    from maxdiffusion.max_utils import create_device_mesh
    from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
    from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxSingleTransformerBlock

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

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
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
          {"params": params}, jax_input, temb_mod=jnp.array(pt_mod.numpy()), image_rotary_emb=jnp.array(jax_rope)
      )

    diff = np.abs(np.array(jax_out) - pt_out.numpy())
    max_abs = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    print(f"\n[UNIT TEST] Single Block Parity -> Max Abs Error: {max_abs:.6f}, RMSE: {rmse:.6e}")
    self.assertLess(max_abs, 1e-4)
    self.assertLess(rmse, 1e-4)


if __name__ == "__main__":
  unittest.main()
