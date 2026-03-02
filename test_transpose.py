import numpy as np

# Simulate hidden_states at the end of VAE decoder
B, C_out, p_t, p1, p2, f, h, w = 1, 3, 1, 4, 4, 2, 3, 5
total_channels = C_out * p_t * p1 * p2
seq_len = f * h * w

# Create a deterministic array
arr = np.arange(B * total_channels * f * h * w).reshape(B, total_channels, f, h, w)

# --- PyTorch Diffusers Trace ---
# PyTorch treats the channel dimension C as (C_out, p_t, p1, p2)
# Here p1 is width patch (4), p2 is height patch (4) according to diffusers permute
arr_pt = arr.reshape(B, C_out, p_t, p1, p2, f, h, w)
# PyTorch permutes to: B, C_out, f, p_t, h, p2, w, p1
arr_pt_permuted = arr_pt.transpose(0, 1, 5, 2, 6, 4, 7, 3)
arr_pt_flat = arr_pt_permuted.reshape(B, C_out, f*p_t, h*p2, w*p1)

# --- JAX MaxDiffusion Trace ---
# JAX tensor has shape (B, f, h, w, total_channels)
# The channels are identical, so we just move them to the end
arr_jax = arr.transpose(0, 2, 3, 4, 1)

# Apply my fix
arr_jax_reshaped = arr_jax.reshape(B, f, h, w, C_out, p_t, p1, p2)
arr_jax_permuted = arr_jax_reshaped.transpose(0, 1, 5, 2, 7, 3, 6, 4)
arr_jax_flat = arr_jax_permuted.reshape(B, f * p_t, h * p2, w * p1, C_out)

# Compare
arr_pt_flat_to_jax = arr_pt_flat.transpose(0, 2, 3, 4, 1)
print("MaxDiffusion matches Diffusers exactly:", np.array_equal(arr_pt_flat_to_jax, arr_jax_flat))

# Now let's try the ORIGINAL MaxDiffusion code
arr_jax_orig_reshaped = arr_jax.reshape(B, f, h, w, C_out, p_t, p1, p2)
arr_jax_orig_permuted = arr_jax_orig_reshaped.transpose(0, 1, 5, 2, 6, 3, 7, 4)
try:
    arr_jax_orig_flat = arr_jax_orig_permuted.reshape(B, f * p_t, h * p2, w * p1, C_out)
    print("Original MaxDiffusion matches Diffusers exactly:", np.array_equal(arr_pt_flat_to_jax, arr_jax_orig_flat))
except Exception as e:
    print("Original MaxDiffusion failed to reshape:", str(e))
