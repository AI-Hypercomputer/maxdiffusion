import jax.numpy as jnp
from maxdiffusion.models.ltx2.ltx2_utils import rename_for_ltx2_transformer, get_key_and_value

keys = [
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.attn1.to_out.0.weight",
    "transformer_blocks.0.norm1.weight",
]

# We need to mock random_flax_state_dict and rename_key 
# Actually just running pytest with coverage is easier.
