import jax
from flax import nnx
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.ltx2_utils import load_transformer_weights
from flax.traverse_util import flatten_dict

class LTX2VideoConfig:
  def __init__(self):
    self.in_channels = 128
    self.out_channels = 128
    self.patch_size = 1
    self.patch_size_t = 1
    self.num_attention_heads = 32
    self.attention_head_dim = 128
    self.cross_attention_dim = 4096
    self.audio_in_channels = 128
    self.audio_out_channels = 128
    self.audio_patch_size = 1
    self.audio_patch_size_t = 1
    self.audio_num_attention_heads = 32
    self.audio_attention_head_dim = 64
    self.audio_cross_attention_dim = 2048
    self.num_layers = 48

config = LTX2VideoConfig()

print("Initializing model structure...", flush=True)
with jax.default_device(jax.devices("cpu")[0]):
  transformer = LTX2VideoTransformer3DModel(
      in_channels=config.in_channels,
      out_channels=config.out_channels,
      patch_size=config.patch_size,
      patch_size_t=config.patch_size_t,
      num_attention_heads=config.num_attention_heads,
      attention_head_dim=config.attention_head_dim,
      cross_attention_dim=4096,
      audio_in_channels=config.audio_in_channels,
      audio_out_channels=config.out_channels,
      audio_patch_size=config.audio_patch_size,
      audio_patch_size_t=config.audio_patch_size_t,
      audio_num_attention_heads=config.audio_num_attention_heads,
      audio_attention_head_dim=64,
      audio_cross_attention_dim=config.audio_cross_attention_dim,
      num_layers=config.num_layers,
      scan_layers=True,
      rngs=nnx.Rngs(0),
  )

state = nnx.state(transformer)
eval_shapes = state.to_pure_dict()

print("Loading converted checkpoint weights (dg845/LTX-2.3-Diffusers)...", flush=True)
loaded_weights = load_transformer_weights(
    pretrained_model_name_or_path="dg845/LTX-2.3-Diffusers",
    eval_shapes=eval_shapes,
    device="cpu",
    hf_download=True,
    num_layers=48,
    scan_layers=True,
)

flat_weights = flatten_dict(loaded_weights)

for key, val in flat_weights.items():
  key_names = [str(k) for k in key]
  if "proj_in" in key_names:
    print(f"📊 JAX Key: {key} | Shape: {val.shape} | Mean: {float(val.mean()):.8f} | Std: {float(val.std()):.8f}", flush=True)
