from typing import List, Tuple
import jax
import jax.numpy as jnp
from flax import nnx

class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: List[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 32

def swish(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)

class AttnBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_channels: int):
        self.in_channels = in_channels
        self.norm = nnx.GroupNorm(
            num_groups=32, num_channels=in_channels, epsilon=1e-6, use_bias=True, use_scale=True, rngs=rngs
        )
        self.q = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), rngs=rngs)
        self.k = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), rngs=rngs)
        self.v = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), rngs=rngs)
        self.proj_out = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), rngs=rngs)

    def attention(self, h_: jax.Array) -> jax.Array:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, h_dim, w_dim, c = q.shape
        q = q.reshape((b, h_dim * w_dim, c))
        k = k.reshape((b, h_dim * w_dim, c))
        v = v.reshape((b, h_dim * w_dim, c))

        attn_weights = jnp.einsum('bqc,bkc->bqk', q, k) * (c ** -0.5)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        h_ = jnp.einsum('bqk,bkc->bqc', attn_weights, v)

        return h_.reshape((b, h_dim, w_dim, c))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.proj_out(self.attention(x))

class ResnetBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nnx.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-6, rngs=rngs)
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)
        
        self.norm2 = nnx.GroupNorm(num_groups=32, num_channels=out_channels, epsilon=1e-6, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), strides=(1, 1), padding=0, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_channels: int):
        self.conv = nnx.Conv(in_channels, in_channels, kernel_size=(3, 3), strides=(2, 2), padding=0, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='constant')
        return self.conv(x)

class Upsample(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_channels: int):
        self.conv = nnx.Conv(in_channels, in_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method='nearest')
        return self.conv(x)

class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, resolution: int, in_channels: int, ch: int, ch_mult: List[int], num_res_blocks: int, z_channels: int):
        self.quant_conv = nnx.Conv(2 * z_channels, 2 * z_channels, kernel_size=(1, 1), rngs=rngs)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nnx.Conv(in_channels, self.ch, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        
        self.down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(rngs, in_channels=block_in, out_channels=block_out))
                block_in = block_out
            
            downsample = None
            if i_level != self.num_resolutions - 1:
                downsample = Downsample(rngs, block_in)
                curr_res = curr_res // 2
                
            self.down.append((block, attn, downsample))

        self.mid_block_1 = ResnetBlock(rngs, in_channels=block_in, out_channels=block_in)
        self.mid_attn_1 = AttnBlock(rngs, block_in)
        self.mid_block_2 = ResnetBlock(rngs, in_channels=block_in, out_channels=block_in)

        self.norm_out = nnx.GroupNorm(num_groups=32, num_channels=block_in, epsilon=1e-6, rngs=rngs)
        self.conv_out = nnx.Conv(block_in, 2 * z_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            block, attn, downsample = self.down[i_level]
            for i_block in range(self.num_res_blocks):
                h = block[i_block](hs[-1])
                if len(attn) > 0:
                    h = attn[i_block](h)
                hs.append(h)
            if downsample is not None:
                hs.append(downsample(hs[-1]))

        h = hs[-1]
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h

class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, ch: int, out_ch: int, ch_mult: List[int], num_res_blocks: int, in_channels: int, resolution: int, z_channels: int):
        self.post_quant_conv = nnx.Conv(z_channels, z_channels, kernel_size=(1, 1), rngs=rngs)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, curr_res, curr_res, z_channels)

        self.conv_in = nnx.Conv(z_channels, block_in, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)

        self.mid_block_1 = ResnetBlock(rngs, in_channels=block_in, out_channels=block_in)
        self.mid_attn_1 = AttnBlock(rngs, block_in)
        self.mid_block_2 = ResnetBlock(rngs, in_channels=block_in, out_channels=block_in)

        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(rngs, in_channels=block_in, out_channels=block_out))
                block_in = block_out
            
            upsample = None
            if i_level != 0:
                upsample = Upsample(rngs, block_in)
                curr_res = curr_res * 2
                
            self.up.insert(0, (block, attn, upsample))

        self.norm_out = nnx.GroupNorm(num_groups=32, num_channels=block_in, epsilon=1e-6, rngs=rngs)
        self.conv_out = nnx.Conv(block_in, out_ch, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        z = self.post_quant_conv(z)
        h = self.conv_in(z)

        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            block, attn, upsample = self.up[i_level]
            for i_block in range(self.num_res_blocks + 1):
                h = block[i_block](h)
                if len(attn) > 0:
                    h = attn[i_block](h)
            if upsample is not None:
                h = upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class AutoEncoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, params: AutoEncoderParams):
        self.params = params
        self.encoder = Encoder(
            rngs,
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            rngs,
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

    def encode(self, x: jax.Array) -> jax.Array:
        # Note: input x should be in NHWC for flax.
        return self.encoder(x)

    def decode(self, z: jax.Array) -> jax.Array:
        # Note: z should be in NHWC for flax.
        return self.decoder(z)
