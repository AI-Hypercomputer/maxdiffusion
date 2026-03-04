import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

def test_conv_transpose():
    # PyTorch setup
    in_channels = 4
    out_channels = 8
    kernel_size = 4
    stride = 2
    padding = 1
    
    pt_conv = nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )
    
    # Initialize with simple values
    nn.init.constant_(pt_conv.weight, 0.1)
    nn.init.constant_(pt_conv.bias, 0.05)
    
    # Make weight more unique to catch transposition errors
    with torch.no_grad():
        pt_conv.weight += torch.randn_like(pt_conv.weight) * 0.1
    
    x_pt = torch.randn(1, in_channels, 10)
    out_pt = pt_conv(x_pt)
    
    # Flax setup
    fx_conv = nnx.ConvTranspose(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(kernel_size,),
        strides=(stride,),
        padding="SAME",
        rngs=nnx.Rngs(0)
    )
    
    # Convert weights
    # pt weight: (in_channels, out_channels, kernel_size)
    # fx weight: (kernel_size, in_features, out_features)
    pt_weight_np = pt_conv.weight.detach().numpy()
    fx_weight_np = np.transpose(pt_weight_np, (2, 0, 1))
    
    # Try different transposes/flips 
    fx_weight_flip = np.flip(fx_weight_np, axis=0) # Flip spatial dimension 
    
    # We will test two cases: standard transpose, and flipped
    
    fx_conv.kernel.value = jnp.array(fx_weight_np)
    fx_conv.bias.value = jnp.array(pt_conv.bias.detach().numpy())
    
    # Flax expects (B, L, C)
    x_fx = jnp.array(x_pt.detach().numpy()).transpose(0, 2, 1)
    
    out_fx = fx_conv(x_fx)
    # Convert out_fx to matching shape (B, C, L)
    out_fx_np = out_fx.transpose(0, 2, 1)
    
    out_pt_np = out_pt.detach().numpy()
    
    print(f"Output shapes: PT {out_pt_np.shape}, FX {out_fx_np.shape}")
    if out_pt_np.shape != out_fx_np.shape:
        print("Shapes mismatch!")
        return
        
    diff = np.max(np.abs(out_pt_np - out_fx_np))
    print(f"Max Diff (Standard Transpose): {diff}")
    
    # Now try flipped
    fx_conv.kernel.value = jnp.array(fx_weight_flip)
    out_fx_flip = fx_conv(x_fx)
    out_fx_flip_np = np.array(out_fx_flip).transpose(0, 2, 1)
    
    diff_flip = np.max(np.abs(out_pt_np - out_fx_flip_np))
    print(f"Max Diff (Flipped Spatial): {diff_flip}")

    # Now print out details about padding if standard transpose does NOT match perfectly
    
if __name__ == '__main__':
    test_conv_transpose()
