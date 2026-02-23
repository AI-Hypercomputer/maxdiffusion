
import jax
import jax.numpy as jnp
from flax import nnx
from flax.traverse_util import flatten_dict
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL

def inspect_structure():
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(8, 16),
        decoder_block_out_channels=(8, 16),
        layers_per_block=(1, 1, 1),
        decoder_layers_per_block=(1, 1, 1), 
        spatio_temporal_scaling=(True, True),
        decoder_spatio_temporal_scaling=(True, True),
        decoder_inject_noise=(False, False, False),
        downsample_type=("spatial", "temporal", "spatial"),
        upsample_residual=(True, True, True),
        upsample_factor=(2, 2, 2),
        rngs=nnx.Rngs(0)
    )
    
    state = nnx.state(model)
    eval_shapes = state.to_pure_dict()
    flat_shapes = flatten_dict(eval_shapes)
    
    print(f"Total keys: {len(flat_shapes)}")
    
    # Check for resnets keys
    resnet_keys = [k for k in flat_shapes.keys() if "resnets" in [str(x) for x in k]]
    print("\nResnet keys sample:")
    for k in resnet_keys[:5]:
        print(f"{k}: {flat_shapes[k].shape}")
        
    # Check for conv_in keys
    conv_in_keys = [k for k in flat_shapes.keys() if "conv_in" in [str(x) for x in k]]
    print("\nConv_in keys sample:")
    for k in conv_in_keys[:5]:
        print(f"{k}")

    # Check for conv_out keys
    conv_out_keys = [k for k in flat_shapes.keys() if "conv_out" in [str(x) for x in k]]
    print("\nConv_out keys sample:")
    for k in conv_out_keys[:5]:
        print(f"{k}")

    # Check for conv1 keys inside resnets
    conv1_keys = [k for k in resnet_keys if "conv1" in [str(x) for x in k]]
    print("\nConv1 keys inside resnets sample:")
    for k in conv1_keys[:5]:
        print(f"{k}")

if __name__ == "__main__":
    inspect_structure()
