import jax.numpy as jnp
from flax import nnx
import jax
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Diffusers PyTorch Implementation
class LTX2AudioCausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        causality_axis: str = "width",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        pad_h = (kernel_size - 1) * dilation
        pad_w = (kernel_size - 1) * dilation

        if self.causality_axis == "none":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in {"width", "width-compatibility"}:
            padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        
        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=0,
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)

# MaxDiffusion JAX Implementation
class FlaxLTX2AudioCausalConv(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        strides: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        causality_axis: str = "width",
        *,
        rngs: nnx.Rngs,
    ):
        self.causality_axis = causality_axis
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.strides = strides
        
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            padding="VALID",
            kernel_dilation=(dilation, dilation),
            feature_group_count=groups,
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(self, x):
        dilation = self.dilation
        kernel_size = self.kernel_size
        
        pad_h = (kernel_size - 1) * dilation
        pad_w = (kernel_size - 1) * dilation
        
        if self.causality_axis == "none":
             padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        elif self.causality_axis in ("width", "width-compatibility"):
             padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w, 0))
        elif self.causality_axis == "height":
             padding = ((pad_h, 0), (pad_w // 2, pad_w - pad_w // 2))

        # Apply padding (Batch, Time, Freq, Channels)
        x = jnp.pad(x, ((0, 0), padding[0], padding[1], (0, 0)))
        return self.conv(x)

# Test
# (Batch=1, Time/Height=86, Freq/Width=64, Channels=4)
in_channels = 4
out_channels = 8
kernel_size = 3
causality = "width" # We test width because that is what audio VAE uses for `causality_axis`

x_pt = torch.arange(1 * in_channels * 86 * 64, dtype=torch.float32).reshape(1, in_channels, 86, 64)
x_jax = jnp.array(x_pt.permute(0, 2, 3, 1).numpy())

pt_model = LTX2AudioCausalConv2d(in_channels, out_channels, kernel_size, causality_axis=causality)

# Assign deterministic weights
# PyTorch Conv2d weight: (out_channels, in_channels, kH, kW)
# JAX Conv kernel: (kH, kW, in_channels, out_channels)
w_pt = torch.arange(out_channels * in_channels * kernel_size * kernel_size, dtype=torch.float32).reshape(out_channels, in_channels, kernel_size, kernel_size)
pt_model.conv.weight.data = w_pt

jax_model = FlaxLTX2AudioCausalConv(in_channels, out_channels, kernel_size, causality_axis=causality, rngs=nnx.Rngs(0))
w_jax = jnp.array(w_pt.permute(2, 3, 1, 0).numpy())
jax_model.conv.kernel.value = w_jax

out_pt = pt_model(x_pt)
out_jax = jax_model(x_jax)

# Diffusers is (Batch, C, Time, Freq). JAX is (Batch, Time, Freq, C)
out_pt_np = out_pt.permute(0, 2, 3, 1).detach().numpy()
out_jax_np = np.array(out_jax)
print(f"Shapes match: PT {out_pt_np.shape} vs JAX {out_jax_np.shape}")
print(f"Max Diff: {np.abs(out_pt_np - out_jax_np).max()}")
