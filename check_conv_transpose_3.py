import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

def test_vocoder_upsample():
    in_channels = 2
    out_channels = 1
    kernel_size = 16
    stride = 6
    padding = (kernel_size - stride) // 2 # 5
    
    pt_conv = nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )
    
    with torch.no_grad():
        # Set weights to a simple linear sequence to easily track shifts
        w = torch.arange(in_channels * out_channels * kernel_size).float()
        pt_conv.weight.copy_(w.view(in_channels, out_channels, kernel_size))
        
    seq_len = 10
    # Create input that is mostly zeros except one spike
    x_pt = torch.zeros(1, in_channels, seq_len)
    x_pt[0, 0, 5] = 1.0 # spike at index 5 for channel 0
    x_pt[0, 1, 6] = 2.0 # spike at index 6 for channel 1
    
    out_pt = pt_conv(x_pt)
    out_pt_np = out_pt.detach().numpy()
    print(f"PT shape: {out_pt.shape}")
    
    # Try multiple paddings for Flax
    pt_weight_np = pt_conv.weight.detach().numpy()
    fx_weight_np = np.transpose(pt_weight_np, (2, 0, 1))
    fx_weight_flip = np.flip(fx_weight_np, axis=0) # Flip spatial dimension 
    x_fx = jnp.array(x_pt.detach().numpy()).transpose(0, 2, 1)

    print("Testing padding tuples:")
    found_match = False
    best_diff = 1e9
    best_pad = None
    
    for pad_left in range(0, 17):
        for pad_right in range(0, 17):
            try:
                fx_conv_custom = nnx.ConvTranspose(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(kernel_size,),
                    strides=(stride,),
                    padding=((pad_left, pad_right),),
                    rngs=nnx.Rngs(0)
                )
                fx_conv_custom.kernel.value = jnp.array(fx_weight_flip)
                if hasattr(fx_conv_custom, 'bias') and fx_conv_custom.bias is not None:
                    fx_conv_custom.bias.value = jnp.zeros_like(fx_conv_custom.bias.value)
                    
                out_fx_custom = np.array(fx_conv_custom(x_fx)).transpose(0, 2, 1)
                
                if out_pt_np.shape == out_fx_custom.shape:
                    diff = np.max(np.abs(out_pt_np - out_fx_custom))
                    if diff < best_diff:
                        best_diff = diff
                        best_pad = (pad_left, pad_right)
                    if diff < 1e-4:
                        print(f"Match found! Padding: {(pad_left, pad_right)}")
                        found_match = True
                        break
            except Exception as e:
                pass
        if found_match:
            break
            
    if not found_match:
        print(f"No exact match. Best diff {best_diff} with pad {best_pad}")
        
    # Also test 'SAME' correctly
    try:
        fx_conv_same = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(stride,),
            padding="SAME",
            rngs=nnx.Rngs(0)
        )
        fx_conv_same.kernel.value = jnp.array(fx_weight_flip)
        if hasattr(fx_conv_same, 'bias') and fx_conv_same.bias is not None:
             fx_conv_same.bias.value = jnp.zeros_like(fx_conv_same.bias.value)
        out_fx_same = np.array(fx_conv_same(x_fx)).transpose(0, 2, 1)
        if out_pt_np.shape == out_fx_same.shape:
            diff_same = np.max(np.abs(out_pt_np - out_fx_same))
            print(f"Diff with SAME: {diff_same}")
    except Exception as e:
         print("SAME failed")


if __name__ == '__main__':
    test_vocoder_upsample()
