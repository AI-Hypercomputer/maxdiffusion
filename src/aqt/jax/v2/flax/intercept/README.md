# AQT Intercept methods API

Intercept methods API designed to make better usability for users.

It is based on `flax.linen.intercept_methods` which we can intercept flax module
methods and apply AQT during runtime.

## Usage

Internally, It should work exactly the same as AQT injection API.

Only difference is API. The injection API puts configs on module code, but the intercept method puts configs on execution by scope.

Below usage code was modified from aqt v2 [example colab](https://github.com/google/aqt/blob/main/aqt/jax/v2/examples/examples.ipynb).


```python
class MlpBlockOriginal(nn.Module):
  @nn.compact
  def __call__(self, inputs):
    x = nn.Dense(features=inputs.shape[-1] * 4)(inputs)
    x = nn.relu(x)
    x = nn.Dense(features=inputs.shape[-1])(x)
    return x

# create a config that quantizes both forward and backward passes to int8
int8_config = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)

# run and print results
mlp_original = MlpBlockOriginal()
init_and_eval('mlp_fp16', mlp_original)

mlp_aqt = MlpBlockOriginal()

# apply aqt by intercept methods.
from aqt.jax.v2.flax.intercept import aqt_intercept_methods
with aqt_intercept_methods.intercept_methods(int8_config):
  init_and_eval('mlp_aqt_intercept', mlp_aqt)
```

Expected result would be exact same as injection API result.

```
mlp_fp16:
 [[ 0.7207441   1.5375544  -2.6456933  -1.7605033 ]
 [-0.01541615  0.09728501 -1.5742414  -0.37375218]
 [ 0.40717596  1.1941448  -0.6982092  -0.48336375]]
mlp_aqt_intercept:
 [[ 0.7030779   1.5099456  -2.6334763  -1.7550919 ]
 [-0.00901393  0.08774488 -1.5644912  -0.3728472 ]
 [ 0.40121436  1.189411   -0.6939187  -0.48000643]]
```

