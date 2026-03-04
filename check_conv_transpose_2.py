import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

def test_vocoder_upsample():
    # PyTorch setup based on what is in LTX2Vocoder config
    in_channels = 1024 # first upsample input
    out_channels = 512
    kernel_size = 16
    stride = 6
    padding = (kernel_size - stride) // 2 # 5
    
    pt_conv = nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )
    
    with torch.no_grad():
        nn.init.normal_(pt_conv.weight)
        nn.init.normal_(pt_conv.bias)
        
    seq_len = 100
    x_pt = torch.randn(1, in_channels, seq_len)
    out_pt = pt_conv(x_pt)
    
    # Flax setup using current implementation
    fx_conv = nnx.ConvTranspose(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(kernel_size,),
        strides=(stride,),
        padding="SAME",
        rngs=nnx.Rngs(0)
    )
    
    # Convert weights with and without spatial flip
    pt_weight_np = pt_conv.weight.detach().numpy()
    # (in_channels, out_channels/groups, kernel_size) -> (kernel_size, in_channels, out_channels/groups)
    # PyTorch: in, out/g, k -> transposed to (k, in, out)
    fx_weight_np = np.transpose(pt_weight_np, (2, 0, 1))
    fx_weight_flip = np.flip(fx_weight_np, axis=0) # Flip spatial dimension 
    
    fx_conv.kernel.value = jnp.array(fx_weight_flip)
    fx_conv.bias.value = jnp.array(pt_conv.bias.detach().numpy())
    
    x_fx = jnp.array(x_pt.detach().numpy()).transpose(0, 2, 1)
    
    out_fx_flip = fx_conv(x_fx)
    out_fx_flip_np = np.array(out_fx_flip).transpose(0, 2, 1)
    
    print(f"Shapes: PT {out_pt.shape}, FX {out_fx_flip_np.shape}")
    
    if out_pt.shape != out_fx_flip_np.shape:
        print("Mismatched shapes!")
        # Print actual values
        seq_len_pt = out_pt.shape[2]
        seq_len_fx = out_fx_flip_np.shape[2]
        print(f"PT sequence length: {seq_len_pt}")
        print(f"FX sequence length: {seq_len_fx}")
        return
        
    diff = np.max(np.abs(out_pt.detach().numpy() - out_fx_flip_np))
    print(f"Max Diff with flip & SAME padding: {diff}")
    
    # Check if a custom padding matches
    # PyTorch formulas for output length: (L - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # For L=100, stride=6, padding=5, kernel=16:
    # 99 * 6 - 10 + 1 * 15 + 0 + 1 = 594 - 10 + 16 = 600
    
    fx_conv_custom = nnx.ConvTranspose(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(kernel_size,),
        strides=(stride,),
        padding=((padding, padding),), # Custom padding equivalent to padding
        rngs=nnx.Rngs(0)
    )
    fx_conv_custom.kernel.value = jnp.array(fx_weight_flip)
    fx_conv_custom.bias.value = jnp.array(pt_conv.bias.detach().numpy())
    out_fx_custom = np.array(fx_conv_custom(x_fx)).transpose(0, 2, 1)
    
    if out_pt.shape == out_fx_custom.shape:
         diff2 = np.max(np.abs(out_pt.detach().numpy() - out_fx_custom))
         print(f"Max Diff with flip & Custom padding ({padding}): {diff2}")
    else:
         print(f"Custom padding Shape Mismatch: PT {out_pt.shape}, FX {out_fx_custom.shape}")

if __name__ == '__main__':
    test_vocoder_upsample()
