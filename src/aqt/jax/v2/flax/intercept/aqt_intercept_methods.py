# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AQT intercept methods."""

import abc
from typing import Any, Callable, Mapping, Sequence

from aqt.jax.v2.flax import aqt_flax
import flax
from flax import linen as nn
import jax
from jax import lax


_Args = Sequence[Any]
_Kwargs = Mapping[str, Any]
_NextGetter = Callable[..., Any]
_Interceptor = Callable[
    [_NextGetter, _Args, _Kwargs, flax.linen.module.InterceptorContext], Any
]


class _DotGeneralScope:
  def __init__(self, dot_general):
    self.dot_general = dot_general

  def __enter__(self):
    self.orignal_dot_general = lax.dot_general
    lax.dot_general = self.dot_general
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    lax.dot_general = self.orignal_dot_general


class DotGeneralGeneratorBase(metaclass=abc.ABCMeta):
  """Abstract class to generate dot general.

  Attributes:
    dot_general: Saved the dot general for the generator.
  """

  @abc.abstractmethod
  def __call__(
      self,
      next_f: _NextGetter,
      args: _Args,
      kwargs: _Kwargs,
      context: flax.linen.module.InterceptorContext):
    """Generate dot general method by intercepted arguments.

    Args:
      next_f: A function wrapped by the interceptor.
      args: Arguments to the function call.
      kwargs: Keyword arguments to the function call.
      context: Context in which the interceptor is applied.

    Returns:
      The dot general method for the intercepted context.
    """
    raise NotImplementedError


class DotGeneralGenerator(DotGeneralGeneratorBase):
  """Generate saved dot general."""

  def __init__(self, dot_general: Any):
    """Initializes dot general generator with given dot general method.

    Args:
      dot_general: Saved the dot general.
    """
    self.dot_general = dot_general

  def __call__(
      self,
      next_f: _NextGetter,
      args: _Args,
      kwargs: _Kwargs,
      context: flax.linen.module.InterceptorContext) -> Any:
    """Return the dot general method passed on initializes.

    Args:
      next_f: A function wrapped by the interceptor.
      args: Arguments to the function call.
      kwargs: Keyword arguments to the function call.
      context: Context in which the interceptor is applied.

    Returns:
      The saved dot general method.
    """
    del next_f, args, kwargs, context
    return self.dot_general


# TODO(kimjaehong): This generator filtered intercept state by module class to
# support flax examples. To support enhanced transformer based LLMs, we need to
# add filters by module name to distinguish between attention and feedforward.
class DotGeneralGeneratorByModule(DotGeneralGenerator):
  """Abstract class to generate dot general by module."""

  def __call__(
      self,
      next_f: _NextGetter,
      args: _Args,
      kwargs: _Kwargs,
      context: flax.linen.module.InterceptorContext):
    """Return the dot general method by module on the context.

    Args:
      next_f: A function wrapped by the interceptor.
      args: Arguments to the function call.
      kwargs: Keyword arguments to the function call.
      context: Context in which the interceptor is applied.

    Returns:
      The dot general method for the module on the context.
    """
    ret = self.generate_by_module(context.module)
    if ret is None:
      ret = super().__call__(next_f, args, kwargs, context)
    return ret

  @abc.abstractmethod
  def generate_by_module(self, module: nn.Module):
    """Generate dot general method by the intercepted module object.

    Args:
      module: An intercepted module from the intercept methods.

    Returns:
      The dot general method for the intercepted module.
    """
    raise NotImplementedError


def intercept_methods_replace_dot_general(
    dot_general_generator: DotGeneralGeneratorBase = DotGeneralGenerator(
        lax.dot_general
    ),
):
  # pylint: disable=g-doc-return-or-yield
  """Flax intercept method wrapper to replace dot general.

  Args:
    dot_general_generator: The dot general generator by the intercepted args.
  """
  dot_general_cache = {}

  def dot_general_interceptor(
      next_f: _NextGetter,
      args: _Args,
      kwargs: _Kwargs,
      context: flax.linen.module.InterceptorContext,
  ) -> Any:
    """Dot general interceptor that replace dot general by module.

    Args:
      next_f: A function wrapped by the interceptor.
      args: Arguments to the function call.
      kwargs: Keyword arguments to the function call.
      context: Context in which the interceptor is applied.

    Returns:
      The result of original function, with dot general scope.
    """
    layer_path = '/'.join(context.module.scope.path)
    if layer_path not in dot_general_cache:
      if not context.module._initialization_allowed:  # pylint: disable=protected-access
        raise ValueError(
            f'The tracing function {context.method_name} is not compact, and'
            f' the module {layer_path} does not have setup. Cannot initialize'
            ' dot general module...')
      dot_general_cache[layer_path] = dot_general_generator(
          next_f, args, kwargs, context)

    with _DotGeneralScope(dot_general_cache[layer_path]):
      out = next_f(*args, **kwargs)
    return out

  return nn.intercept_methods(dot_general_interceptor)


class AqtDotGeneralGenerator(DotGeneralGeneratorByModule):
  """Generate AQT dot general by module."""

  def __init__(self, *args, **kwargs):
    super().__init__(jax.lax.dot_general)
    self.aqt_args = args
    self.aqt_kwargs = kwargs

  def generate_by_module(self, module: nn.Module):
    """Generate AQT dot general method by the intercepted module object.

    Args:
      module: An intercepted module from the intercept methods.

    Returns:
      The AQT dot general method for the intercepted module.
    """
    # Replace `jax.lax.dot_general` under the all dense layer methods.
    if isinstance(module, nn.Dense):
      return aqt_flax.AqtDotGeneral(*self.aqt_args, **self.aqt_kwargs)
    return None


def intercept_methods(*args, **kwargs):
  return intercept_methods_replace_dot_general(
      AqtDotGeneralGenerator(*args, **kwargs))


def intercept_wrapper(func, *aqt_args, **aqt_kwargs):
  def wrapper(*args, **kwargs):
    with intercept_methods(*aqt_args, **aqt_kwargs):
      return func(*args, **kwargs)
  return wrapper

