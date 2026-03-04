import jax.numpy as jnp
from flax import nnx
import torch
import torch.nn as nn
import numpy as np

# A tiny test to check vanilla Conv2D parity between PyTorch and JAX
in_c = 2
out_c = 3
k = 2

# PyTorch
pt_conv = nn.Conv2d(in_c, out_c, k, bias=False)
x_pt = torch.arange(1 * in_c * 4 * 4, dtype=torch.float32).reshape(1, in_c, 4, 4)
w_pt = torch.arange(out_c * in_c * k * k, dtype=torch.float32).reshape(out_c, in_c, k, k)
pt_conv.weight.data = w_pt

out_pt = pt_conv(x_pt)
out_pt_np = out_pt.permute(0, 2, 3, 1).detach().numpy() 

# JAX
jax_conv = nnx.Conv(in_c, out_c, (k, k), use_bias=False, padding="VALID", rngs=nnx.Rngs(0))
x_jax = jnp.array(x_pt.permute(0, 2, 3, 1).numpy()) # (1, 4, 4, 2)
w_jax = jnp.array(w_pt.permute(2, 3, 1, 0).numpy()) # (2, 2, 2, 3) 
jax_conv.kernel.value = w_jax

out_jax = jax_conv(x_jax)
out_jax_np = np.array(out_jax)

print("Shapes:", out_pt_np.shape, out_jax_np.shape)
print("Max Diff with permute(2,3,1,0):", np.abs(out_pt_np - out_jax_np).max())

# Let's see if another transposition fixes it?
# In regular maxdiffusion ltx2_utils.py `load_audio_vae_weights`:
# `if tensor.ndim == 4: tensor = tensor.transpose(2, 3, 1, 0)` -> which is permute(2, 3, 1, 0)
