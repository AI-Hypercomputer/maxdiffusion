import jax
import jax.numpy as jnp
from flax import nnx

def load_ideogram_transformer_weights(model: nnx.Module, pt_state_dict: dict):
    """
    Update the nnx model state in place using the PyTorch state dict.
    """
    # model is an instance of Ideogram4Transformer
    nnx_state = nnx.state(model)
    flat_nnx_state = nnx_state.flat_state()

    # Map keys
    updated_flat_state = {}
    
    # We will do a manual mapping loop based on the keys
    for pt_key, pt_tensor in pt_state_dict.items():
        # pt_key is like "layers.0.attention.qkv.weight"
        # We need to map it to nnx_state path tuple
        
        # Replace numbers in 'layers.0' to 'layers', 0
        parts = pt_key.split('.')
        flax_path = []
        for p in parts:
            if p.isdigit():
                flax_path.append(int(p))
            else:
                flax_path.append(p)
                
        # PyTorch to Flax naming:
        # weight -> kernel or embedding (for Embed) or value (for Variable)
        if flax_path[-1] == 'weight':
            # Could be Conv, Linear, LayerNorm, Embed
            # We'll check the shape to infer transpose
            parent_path = tuple(flax_path[:-1])
            if parent_path + ('kernel',) in flat_nnx_state:
                flax_path[-1] = 'kernel'
            elif parent_path + ('embedding',) in flat_nnx_state:
                flax_path[-1] = 'embedding'
            elif parent_path + ('scale',) in flat_nnx_state:
                flax_path[-1] = 'scale'
                
        # For our custom Ideogram4RMSNorm, we named it 'weight' but it's a Param wrapped value.
        # Wait, nnx.Param in flat_state has a path ending in `value` by default!
        # E.g., `layers`, 0, `attention_norm1`, `weight`, `value`
        
        # Try to resolve to the exact flat_nnx_state key
        best_match = None
        for nnx_key in flat_nnx_state.keys():
            # Convert nnx_key to string representation
            # e.g., ('layers', 0, 'attention', 'qkv', 'kernel', 'value')
            # Let's see if pt_key maps naturally
            nnx_key_str = ".".join([str(k) for k in nnx_key])
            if pt_key.replace("weight", "kernel") in nnx_key_str or \
               pt_key.replace("weight", "embedding") in nnx_key_str or \
               pt_key.replace("weight", "weight") in nnx_key_str: # for custom RMSNorm
               
               # Check suffix
               if nnx_key_str.endswith("value") and nnx_key_str.replace(".value", "") == ".".join([str(p) for p in flax_path]):
                   best_match = nnx_key
                   break
                   
        if best_match is not None:
            v = jnp.array(pt_tensor)
            if best_match[-2] == 'kernel' and v.ndim == 2:
                v = v.T
            updated_flat_state[best_match] = v
        else:
            print(f"Warning: Could not map PyTorch key {pt_key} to Flax.")

    # Update state
    new_state = nnx.State(updated_flat_state)
    nnx.update(model, new_state)

def load_ideogram_autoencoder_weights(model: nnx.Module, pt_state_dict: dict):
    # Autoencoder weight loading logic
    from maxdiffusion.models.ideogram.autoencoder_ideogram import convert_diffusers_state_dict
    
    # Assuming pt_state_dict is the diffusers formatted state dict, 
    # we convert it using the logic from PyTorch first
    pt_state_dict = convert_diffusers_state_dict(pt_state_dict)
    
    nnx_state = nnx.state(model)
    flat_nnx_state = nnx_state.flat_state()
    updated_flat_state = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        parts = pt_key.split('.')
        flax_path = []
        for p in parts:
            if p.isdigit():
                flax_path.append(int(p))
            else:
                flax_path.append(p)
                
        if flax_path[-1] == 'weight':
            parent_path = tuple(flax_path[:-1])
            if parent_path + ('kernel',) in flat_nnx_state:
                flax_path[-1] = 'kernel'
            elif parent_path + ('scale',) in flat_nnx_state:
                flax_path[-1] = 'scale'
                
        best_match = None
        for nnx_key in flat_nnx_state.keys():
            nnx_key_str = ".".join([str(k) for k in nnx_key])
            if nnx_key_str.endswith("value") and nnx_key_str.replace(".value", "") == ".".join([str(p) for p in flax_path]):
                best_match = nnx_key
                break
                
        if best_match is not None:
            v = jnp.array(pt_tensor)
            # Conv weights in Flax: (H, W, in_C, out_C)
            # Conv weights in PyTorch: (out_C, in_C, H, W)
            if best_match[-2] == 'kernel' and v.ndim == 4:
                v = jnp.transpose(v, (2, 3, 1, 0))
            elif best_match[-2] == 'kernel' and v.ndim == 2:
                v = v.T
            updated_flat_state[best_match] = v
        else:
            print(f"Warning: Could not map PyTorch key {pt_key} to Flax.")

    new_state = nnx.State(updated_flat_state)
    nnx.update(model, new_state)
