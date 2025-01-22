
# copied from https://github.com/ml-gde/jflux/blob/main/jflux/util.py
import os
from dataclasses import dataclass

import jax
from jax.typing import DTypeLike
import torch  # need for torch 2 jax
from chex import Array
from flax import nnx
from huggingface_hub import hf_hub_download
from jax import numpy as jnp
from safetensors import safe_open

# from jflux.model import Flux, FluxParams
from .port import port_flux

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    rngs: Array
    param_dtype: DTypeLike

def torch2jax(torch_tensor: torch.Tensor) -> Array:
    is_bfloat16 = torch_tensor.dtype == torch.bfloat16
    if is_bfloat16:
        # upcast the tensor to fp32
        torch_tensor = torch_tensor.to(dtype=torch.float32)

    if torch.device.type != "cpu":
        torch_tensor = torch_tensor.to("cpu")

    numpy_value = torch_tensor.numpy()
    jax_array = jnp.array(numpy_value, dtype=jnp.bfloat16 if is_bfloat16 else None)
    return jax_array


@dataclass
class ModelSpec:
    params: FluxParams
    ckpt_path: str | None
    repo_id: str | None
    repo_flow: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
            #rngs=nnx.Rngs(default=42),
            rngs=jax.random.PRNGKey(42),
            param_dtype=jnp.bfloat16,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
            #rngs=nnx.Rngs(default=42),
            rngs=jax.random.PRNGKey(47),
            param_dtype=jnp.bfloat16,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str, hf_download: bool = True): # -> Flux:
    device = jax.devices(device)[0]
    with jax.default_device(device):
        ckpt_path = configs[name].ckpt_path
        if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_flow is not None
            and hf_download
        ):
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

        print(f"Load and port flux on {device}")

        #model = Flux(params=configs[name].params)
        if ckpt_path is not None:
            tensors = {}
            with safe_open(ckpt_path, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = torch2jax(f.get_tensor(k))
            breakpoint()
            model = port_flux(flux=model, tensors=tensors)

            del tensors
            jax.clear_caches()
    return model
