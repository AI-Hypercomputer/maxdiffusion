import re
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import flax
import jax
import torch
import torch.utils._pytree as pytree
from flax.traverse_util import flatten_dict


AnyTensor = Union[jax.Array, torch.Tensor]
StateDict = Dict[str, AnyTensor]

ScanRepeatableCarryBlock = "ScanRepeatableCarryBlock"

JaxParams = Dict[str, Union[Dict[str, jax.Array], jax.Array]]


def unbox_logically_partioned(statedict: JaxParams) -> JaxParams:
  return jax.tree_util.tree_map(
      lambda t: t.unbox() if isinstance(t, flax.linen.spmd.LogicallyPartitioned) else t,
      statedict,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def torch_tensor_to_jax_array(data: torch.Tensor) -> jax.Array:
  match data.dtype:
    case torch.bfloat16:
      return jax.numpy.from_dlpack(data)
    case _:
      return jax.numpy.array(data)


def is_stack_or_tensor(param: Any) -> bool:
  """
  Returns True if param is of type tensor or list/tuple of tensors (stack of tensors)

  Used for mapping utils
  """
  return isinstance(param, (torch.Tensor, list, tuple))


def convert_tensor_stack_to_tensor(param: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
  """
  Converts a list of torch tensors to a single torch tensor.
  Args:
      param (Union[List[torch.Tensor], torch.Tensor]): The parameter to convert.

  Returns:
      torch.Tensor: The converted tensor.
  """
  if isinstance(param, list):
    return torch.stack(param)
  return param


@dataclass
class ConvertAction:
  """
  Defines a set of actions to be done on a given parameter.

  The definition must be commutative, i.e. the order of the actions should not matter.
  also we should strive for actions to be reversible (so the same action can be used for both directions).
  """

  transpose: Optional[Tuple[int, int]] = None
  """
    If defined, transposes the tensor with the given indices.
    Example: (1, 0) transposes a (at least 2D tensor) from (..., a, b) to (..., b, a).
    """

  rename: Optional[Dict[str, str]] = None
  """
    If defined, renames the parameter according to the given mapping.
    Example: {"torch": "weight", "jax": "kernel"}
        * renames "torch.weight" to "jax.kernel" when converting from torch to jax.
        * renames "jax.kernel" to "torch.weight" when converting from jax to torch.
    """

  split_by: Optional[str] = None
  """
    If defined, splits the parameter by the given delimiter.
    Example: "ScanRepeatableCarryBlock.k1" assumes the parameter is a concatenation of multiple tensors (shaped: (n, ...)).
    and splits them into individual tensors named as "ScanRepeatableCarryBlock.0.k1", "ScanRepeatableCarryBlock.n.k1".
    """

  group_by: Optional[str] = None
  """
    If defined, groups the parameter by the given delimiter.
    Example: "ScanRepeatableCarryBlock.0.k1", "ScanRepeatableCarryBlock.1.k1", "ScanRepeatableCarryBlock.2.k1"
    will be grouped into a single tensor named "ScanRepeatableCarryBlock.k1" shaped (n, ...).

    *** Note:
    this is kind of the reverse of split_by, only a different behavior.
    it's easy to define "actions" that are reversible in base of context (jax->torch, torch->jax).
    but it's very wrong to do so, since it blocks modular behavior and makes the code harder to maintain.

    """

  jax_groups: Optional[List[str]] = None
  """
    Generally used in group_by, this is a list of all possible keys that can be used to group the parameters.
    This must be defined if group_by is defined.

    It's due to the un-reversibility nature of the group_by action.
    """

  def apply_transpose(self, mini_statedict: StateDict) -> StateDict:
    """
    Applies the transpose action if defined
    Args:
        mini_statedict (StateDict): Local context of the state dict

    Returns:
        StateDict: Output local context of the state dict
    """

    if self.transpose is None:
      return mini_statedict
    index0, index1 = self.transpose
    return {param_name: param.swapaxes(index0, index1) for param_name, param in mini_statedict.items()}

  def apply_rename(self, mini_statedict: StateDict, delim: str) -> StateDict:
    """
    Applies the rename action if defined

    Args:
        mini_statedict (StateDict): Local context of the state dict
        delim (str): delimiter used for parsing (usually "."), kept as parameter for flexibility.

    Returns:
        StateDict: Output local context of the state dict
    """
    if self.rename is None:
      return mini_statedict

    param_names = list(mini_statedict.keys())
    for param_name in param_names:
      param = mini_statedict.pop(param_name)
      parts = param_name.split(delim)
      rename_source = "torch" if isinstance(param, torch.Tensor) else "jax"
      rename_target = "jax" if isinstance(param, torch.Tensor) else "torch"
      source_name = self.rename[rename_source]
      dest_name = self.rename[rename_target]
      if source_name == param_name:
        new_param_name = dest_name
      else:
        # There is always ```self.rename[rename_source]``` in parts
        index = parts.index(self.rename[rename_source])
        parts[index] = self.rename[rename_target]
        new_param_name = delim.join(parts)
      mini_statedict[new_param_name] = param

    return mini_statedict

  def apply_split_by(self, mini_statedict: StateDict, new_params: List, delim: str) -> Tuple[StateDict, List[str]]:
    """
    Applies the split_by action if defined

    Args:
        mini_statedict (StateDict): Local state dict
        new_params (List): State containing list of new params that were created during the process (if any)
        delim (str): Output local context of the state dict

    Returns:
        Tuple[StateDict, List[str]]: Output local context of the state dict and list of new keys to add to the global state dict.
    """
    if self.split_by is None:
      return mini_statedict, new_params

    param_names = list(mini_statedict.keys())
    for param_name in param_names:
      parts = param_name.split(delim)
      indices = [i for i, p in enumerate(parts) if self.split_by in p]
      if len(indices) != 1:
        raise ValueError(f"Expected exactly one split_by in param_name: {param_name}")
      index = indices[0]
      params = mini_statedict.pop(param_name)
      for i, param in enumerate(params):
        new_parts = parts[:index] + [f"{i}"] + parts[index + 2 :]
        new_param_name = delim.join(new_parts)
        mini_statedict[new_param_name] = param
        new_params.append(new_param_name)

    return mini_statedict, new_params

  def apply_group_by(
      self, mini_statedict: StateDict, new_params: List, full_statedict: StateDict, delim: str
  ) -> Tuple[StateDict, List[str]]:
    """
    Applies the group_by action if defined

    Args:
        mini_statedict (StateDict): Local state dict
        new_params (List): State containing list of new params that were created during the process (if any)
        full_statedict (StateDict): Global context of the state dict
        delim (str): delimiter used for parsing (usually "."), kept as parameter for flexibility.

    Returns:
        Tuple[StateDict, List[str]]: Output local context of the state dict and list of new keys to add to the global state dict.
    """
    if self.group_by is None:
      return mini_statedict, new_params

    param_names = list(mini_statedict.keys())
    for param_name in param_names:
      param = mini_statedict.pop(param_name)
      jax_keywords = extract_scan_keywords(param_name, self.jax_groups, delim)
      block_index = re.findall(r"\.\d+\.", param_name)[0][1:-1]
      parts = param_name.split(delim)
      index = parts.index(block_index)
      prefix = delim.join(parts[:index])
      suffix = delim.join(parts[index + 1 :])

      new_param_name = f"{prefix}.{delim.join(jax_keywords)}.{suffix}"

      if new_param_name not in full_statedict:
        full_statedict[new_param_name] = [param]
      else:
        full_statedict[new_param_name] = full_statedict[new_param_name] + [param]

    return mini_statedict, new_params

  def __call__(
      self,
      mini_statedict: StateDict,
      new_params: List,
      full_statedict: StateDict,
      delim: str,
  ) -> Tuple[StateDict, List[str]]:
    """
    Given a state dict, applies the transformations defined in the ConvertAction.

    Args:
        mini_statedict (StateDict): Local context of the state dict
        new_params (List): new params that were created during the process (if any)
        full_statedict (StateDict): Global context of the state dict
        delim (str): delimiter used for parsing (usually "."), kept as parameter for flexibility.

    Returns:
        Tuple[StateDict, List[str]]: Updated local state dict and list of new keys to add to the global state dict.
    """
    mini_statedict = self.apply_transpose(mini_statedict)
    mini_statedict = self.apply_rename(mini_statedict, delim)
    mini_statedict, new_params = self.apply_split_by(mini_statedict, new_params, delim)
    mini_statedict, new_params = self.apply_group_by(mini_statedict, new_params, full_statedict, delim)
    return mini_statedict, new_params


def is_kernel_2d(param_name: str, param: AnyTensor) -> bool:
  """
  Checks if the parameter is a 2D kernel (weight) or not.
  usually applies to linear layers or convolutions.
  Args:
      param_name (str): Name of the parameter
      param (AnyTensor): The parameter itself (could be either jax or torch Tensor)

  Returns:
      bool: True if the parameter is a weight for linear/convolutional layer or not.
  """
  expected_name = "weight" if isinstance(param, torch.Tensor) else "kernel"
  return expected_name in param_name and param.ndim == 2


def is_scan_repeatable(param_name: str, _) -> bool:
  """
  Checks if the parameter is a scan repeatable carry block parameter.

  Args:
      param_name (str): Parameter name
      _ (_type_): Unused, will contain the parameter itself

  Returns:
      bool: True if the parameter is a scan repeatable carry block parameter or not.
  """
  return ScanRepeatableCarryBlock in param_name


def is_scale_shift_table(param_name: str, _) -> bool:
  """
  Checks if the parameter is a scale shift table parameter.

  Args:
      param_name (str): Parameter name
      _ (_type_): Unused, will contain the parameter itself

  Returns:
      bool: True if the parameter is a scale shift table parameter or not.
  """
  return "scale_shift_table" in param_name


def is_affine_scale_param(param_name: str, parameter: AnyTensor, jax_flattened_keys: List[str]) -> bool:
  """
  Checks if the parameter is an affine scale parameter.

  Args:
      param_name (str): Parameter name
      parameter (AnyTensor): The parameter itself
      jax_flattened_keys (List[str]): Flattened list of the keys use in jax (for reference and keys search)


  Returns:
      bool: True if the parameter is an affine scale parameter or not.
  """
  if isinstance(parameter, torch.Tensor):
    return "weight" in param_name and parameter.ndim == 1 and param_name not in jax_flattened_keys
  else:
    return "scale" in param_name and parameter.ndim == 1


def extract_scan_keywords(param_name: str, jax_flattened_keys: List[str], delim: str) -> Optional[Tuple[str, str]]:
  """
  Extracts the keywords from the scan repeatable carry block parameter (if exists)

  If the parameter is a scan repeatable carry block, it will return the keywords that are used to group the parameters.
  otherwise it will return None.

  Args:
      param_name (str): Name of the parameter
      jax_flattened_keys (List[str]): Flattened list of the keys use in jax (for reference and keys search)
      delim (str): The delimiter used in the parameter name (in torch)

  Returns:
      Optional[Tuple[str, str]]: Tuple of the keywords used to group the parameters (or None if it is not a scan repeatable carry block)
  """
  block_indices = re.findall(r"\.\d+\.", param_name)

  if len(block_indices) == 0:
    return None
  block_indices = [block_indices[0]]
  block_index = block_indices[0][1:-1]
  parts = param_name.split(delim)
  index = parts.index(block_index)
  prefix = delim.join(parts[:index])
  suffix = delim.join(parts[index + 1 :])

  for flat_key in jax_flattened_keys:
    if flat_key.startswith(prefix) and flat_key.endswith(suffix):
      mid_layer = flat_key[len(prefix) + 1 : -len(suffix) - 1]
      mid_parts = mid_layer.split(delim)
      if not any(ScanRepeatableCarryBlock in mid_part for mid_part in mid_parts):
        continue
      return mid_parts

  return None


def should_be_scan_repeatable(param_name: str, param: AnyTensor, jax_flattened_keys: List[str], delim: str) -> bool:
  """
  Checks if the parameter should be a scan repeatable carry block or not.
  Args:
      param_name (str): The name of the parameter
      param (AnyTensor): the Parameter itself
      jax_flattened_keys (List[str]): Flattened list of the keys use in jax (for reference and keys search)
      delim (str): The delimiter used in the parameter name (in torch)

  Returns:
      bool: True if the paramter should be treated scan repeatable block parameter.
  """
  if not isinstance(param, torch.Tensor):
    return False

  keywords = extract_scan_keywords(param_name, jax_flattened_keys, delim)
  return keywords is not None


def jax_statedict_to_torch(
    jax_params: JaxParams, rulebook: Optional[Dict[Callable[[str, AnyTensor], bool], ConvertAction]] = None
) -> Dict[str, torch.Tensor]:
  """
  Converts a JAX state dict to a torch state dict.

  Args:
      jax_params (JaxParams): The current params in JAX format, to ease parsing and conversion.
      rulebook (Optional[Dict[Callable[[str, AnyTensor], bool], ConvertAction]], optional): Defines a rulebook stating how to convert state dict from jax to torch.
                                                                                            Defaults to None.


  Returns:
      Dict[str, torch.Tensor]: The converted state dict in torch format (Pytorch state dict).
  """

  affine_scale_search = partial(is_affine_scale_param, jax_flattened_keys=[])

  if rulebook is None:
    rulebook = {
        is_scan_repeatable: ConvertAction(split_by=ScanRepeatableCarryBlock),
        is_kernel_2d: ConvertAction(transpose=(1, 0), rename=dict(torch="weight", jax="kernel")), #noqa: C408
        affine_scale_search: ConvertAction(rename=dict(torch="weight", jax="scale")), #noqa: C408
    }
  if "params" not in jax_params:
    raise ValueError('Expected "params" key in jax_params, are you sure you are passing the correct object?')

  jax_params = copy(jax_params["params"])  # Non reference copy
  jax_params = unbox_logically_partioned(jax_params)

  delim = "."
  # Move to flattened dict to match torch state dict convention
  flattened_params = flatten_dict(jax_params, sep=delim)

  param_names = list(flattened_params.keys())
  for param_name in param_names:
    param = flattened_params.pop(param_name)
    mini_statedict = {param_name: param}
    new_params = []
    for condition, rule in rulebook.items():
      if condition(param_name, param):
        mini_statedict, new_params = rule(mini_statedict, new_params, flattened_params, delim)
        if len(mini_statedict) == 1:
          param_name = list(mini_statedict.keys())[0]

    flattened_params.update(mini_statedict)
    param_names.extend(new_params)

  flattened_params = pytree.tree_map(convert_tensor_stack_to_tensor, flattened_params, is_leaf=is_stack_or_tensor)

  to_cpu = pytree.tree_map(lambda t: jax.device_put(t, jax.devices("cpu")[0]), flattened_params)
  to_torch = pytree.tree_map(torch.from_dlpack, to_cpu)
  return to_torch


def torch_statedict_to_jax(
    jax_params: JaxParams,
    torch_params: Dict[str, torch.Tensor],
) -> JaxParams:
  """
  Converts a torch state dict to a JAX state dict.

  Args:
      jax_params (JaxParams): The current params in JAX format, to ease parsing and conversion.
      torch_params (Dict[str, torch.Tensor]): The current params in torch format, to load parameters from.

  Returns:
      JaxParams: The state dict in JAX format.
  """
  with jax.default_device("cpu"):
    jax_params = copy(jax_params)
    jax_params = unbox_logically_partioned(jax_params)
    torch_params = copy(torch_params)

    if "params" not in jax_params:
      raise ValueError('Expected "params" key in jax_params, are you sure you are passing the correct object?')

    delim = "."
    flattened_keys = list(flatten_dict(jax_params["params"], sep=".").keys())
    scan_repeatable_cond = partial(should_be_scan_repeatable, jax_flattened_keys=flattened_keys, delim=delim)
    affine_scale_search = partial(is_affine_scale_param, jax_flattened_keys=flattened_keys)

    rulebook = {
        is_kernel_2d: ConvertAction(transpose=(1, 0), rename=dict(torch="weight", jax="kernel")), #noqa: C408
        affine_scale_search: ConvertAction(rename=dict(torch="weight", jax="scale")), #noqa: C408
        scan_repeatable_cond: ConvertAction(group_by=ScanRepeatableCarryBlock, jax_groups=flattened_keys),
    }

    # First pass - Rulebook
    param_names = list(torch_params.keys())
    for param_name in param_names:
      param = torch_params.pop(param_name)
      mini_statedict = {param_name: param}
      new_params = []
      for condition, rule in rulebook.items():
        if condition(param_name, param):
          mini_statedict, new_params = rule(mini_statedict, new_params, torch_params, delim=delim)
          if len(mini_statedict) == 1:
            param_name = list(mini_statedict.keys())[0]

      torch_params.update(mini_statedict)
      param_names.extend(new_params)

    # Ensures any list of tensors are converted to a single tensor
    # This is due to the fact that the scan repeatable block is a list of tensors
    torch_params = pytree.tree_map(convert_tensor_stack_to_tensor, torch_params, is_leaf=is_stack_or_tensor)

    to_jax: Dict = pytree.tree_map(torch_tensor_to_jax_array, torch_params)

    def nested_insert(param_name: str, param: torch.Tensor, nested_dict: Dict):
      """
      Inserts a parameter into a nested dictionary. (to fit Jax format)
      The keys in torch are split into groups by a delimiter of choice (usually "." to fit torch schema)
      and then inserted into a nested dictionary.

      in case the parameter is of the form of "a.b" and "a.b" is a layer type in jax -
      the parameter will be inserted as "a.b": {...: param}. this ensures compatibility between jax layers and torch layers.

      Args:
          param_name (str): Parameter name
          param (torch.Tensor): Parameter itself
          nested_dict (Dict): Current nested dict state
      """
      if delim not in param_name:
        nested_dict[param_name] = param
        return

      parts = param_name.split(delim)
      if len(parts) == 1:
        return nested_insert(parts[0], param, nested_dict)
      else:
        key = parts[0]
        # May be either complex key or nested key
        if len(parts) > 2 and re.fullmatch(r"\d+", parts[1]) is not None:
          key = delim.join(parts[:2])
          new_param_name = delim.join(parts[2:])
        else:
          new_param_name = delim.join(parts[1:])
        new_nested_dict = nested_dict.get(key, {})
        nested_dict[key] = new_nested_dict
        return nested_insert(new_param_name, param, new_nested_dict)

    params = {}
    for param_name, param in to_jax.items():
      nested_insert(param_name, param, params)

    # Jax state dict is usually held as dict containings "parmas" keys which contains
    # dict of dict containing all the params
    return {"params": params}
