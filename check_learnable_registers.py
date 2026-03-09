import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
from flax import nnx
from flax.traverse_util import flatten_dict
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder

def main():
    rngs = nnx.Rngs(0)
    encoder = LTX2AudioVideoGemmaTextEncoder(rngs=rngs)
    _, state = nnx.split(encoder)
    
    # Convert nnx State to dict
    flat_state = flatten_dict(state.to_pure_dict())
    
    for k in flat_state.keys():
        if "learnable_registers" in k:
            print(k)

if __name__ == "__main__":
    main()
