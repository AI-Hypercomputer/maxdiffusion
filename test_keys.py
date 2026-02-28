import jax
import jax.numpy as jnp
from flax import nnx

def fix_struct(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            key = jax.random.key(0)
            if x.shape == ():
                return key
            else:
                return jax.random.split(key, x.shape[0])
        else:
            return jnp.zeros(x.shape, x.dtype)
    return x

struct_key = jax.ShapeDtypeStruct((5,), jax.dtypes.prng_key)
struct_count = jax.ShapeDtypeStruct((5,), jnp.uint32)

fixed_key = fix_struct(struct_key)
fixed_count = fix_struct(struct_count)

print("Key dtype:", fixed_key.dtype, "shape:", fixed_key.shape)
print("Count dtype:", fixed_count.dtype, "shape:", fixed_count.shape)
