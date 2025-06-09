import os
import jax
import jax.numpy as jnp
import json


from models.transformers.transformer3d import Transformer3DModel

# Load JSON config
base_dir = os.path.dirname(__file__)
config_path = os.path.join(base_dir, "xora_v1.2-13B-balanced-128.json")
with open(config_path, "r") as f:
    model_config = json.load(f)

key = jax.random.PRNGKey(0)
model = Transformer3DModel(**model_config, dtype=jnp.bfloat16, gradient_checkpointing="matmul_without_batch")

batch_size, text_tokens, num_tokens, features = 4, 256, 2048, 128
prompt_embeds = jax.random.normal(key, shape=(batch_size, text_tokens, features), dtype=jnp.bfloat16)
fractional_coords = jax.random.normal(key, shape=(batch_size, 3, num_tokens), dtype=jnp.bfloat16)
latents = jax.random.normal(key, shape=(batch_size, num_tokens, features), dtype=jnp.bfloat16)
noise_cond = jax.random.normal(key, shape=(batch_size, 1), dtype=jnp.bfloat16)

model_params = model.init(
    hidden_states=latents,
    indices_grid=fractional_coords,
    encoder_hidden_states=prompt_embeds,
    timestep=noise_cond,
    rngs={"params": key}
)

output = model.apply(
    model_params,
    hidden_states=latents,
    indices_grid=fractional_coords,
    encoder_hidden_states=prompt_embeds,
    timestep=noise_cond,
)

print("done!")
