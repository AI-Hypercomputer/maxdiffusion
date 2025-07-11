from dataclasses import field
from typing import Any, Callable, Dict, List, Tuple, Optional

import jax
from flax import linen as nn
import jax.numpy as jnp
from flax.linen import partitioning


class RepeatableCarryBlock(nn.Module):
    """
    Integrates an input module in a jax carry format

    ergo, the module assumes the role of a building block
    and returns both input and output across all blocks
    """

    module: Callable[[Any], nn.Module]
    module_init_args: List[Any]
    module_init_kwargs: Dict[str, Any]

    @nn.compact
    def __call__(self, carry: Tuple[jax.Array, jax.Array], *block_args) -> Tuple[Tuple[jax.Array, jax.Array], None]:
        data_input, index_input = carry

        mod = self.module(*self.module_init_args, **self.module_init_kwargs)

        # block_args are the static arguments passed to each individual block
        output_data = mod(index_input, data_input, *block_args) # Pass block_args to the module

        next_index = index_input + 1
        new_carry = (output_data, next_index)
    

        return new_carry, None

class RepeatableLayer(nn.Module):
    """
    RepeatableLayer will assume a similar role to torch.nn.ModuleList
    with the condition that each block has the same graph, and only the parameters differ

    The compilation in RepeatableLayer will happen only once, in contrast to repeat-graph compilation
    """

    module: Callable[[Any], nn.Module]
    """
    A Callable function for single block construction
    """

    num_layers: int
    """
    The amount of blocks to build
    """

    module_init_args: List[Any] = field(default_factory=list)
    """
    args passed to RepeatableLayer.module callable, to support block construction
    """

    module_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    kwargs passed to RepeatableLayer.module callable, to support block construction
    """

    pspec_name: Optional[str] = None
    """
    Partition spec metadata
    """

    param_scan_axis: int = 0
    """
    The axis that the "layers" will be aggragated on
    eg: if a kernel is shaped (8, 16)
    N layers will be (N, 8, 16) if param_scan_axis=0
    and (8, N, 16) if param_scan_axis=1
    """

    @nn.compact
    def __call__(self, *args): # args is now the full input to RepeatableLayer
        if not args:
            raise ValueError("RepeatableLayer expects at least one argument for initial data input.")

        initial_data_input = args[0] # The first element is your main data input
        static_block_args = args[1:] # Any subsequent elements are static args for each block

        initial_index = jnp.array(0, dtype=jnp.int32)

        scan_kwargs = {}
        if self.pspec_name is not None:
            scan_kwargs["metadata_params"] = {nn.PARTITION_NAME: self.pspec_name}

        initializing = self.is_mutable_collection("params")
        params_spec = self.param_scan_axis if initializing else partitioning.ScanIn(self.param_scan_axis)

        # in_axes for the scanned function (RepeatableCarryBlock.__call__):
        # 1. The 'carry' tuple ((0, 0))
        # 2. Then, nn.broadcast for each of the `static_block_args`
        in_axes_for_scan = (nn.broadcast,) * (len(args)-1)

        scan_fn = nn.scan(
            RepeatableCarryBlock,
            variable_axes={
                "params": params_spec,
                "cache": 0,
                "intermediates": 0,
                "aqt": 0,
                "_overwrite_with_gradient": 0,
            },
            split_rngs={"params": True},
            in_axes=in_axes_for_scan,
            length=self.num_layers,
            **scan_kwargs,
        )

        wrapped_function = scan_fn(self.module, self.module_init_args, self.module_init_kwargs)

        # Call wrapped_function with the initial carry tuple and the static_block_args
        (final_data, final_index), _ = wrapped_function((initial_data_input, initial_index), *static_block_args)

        # Typically, you only want the final data output from the sequence of layers
        return final_data