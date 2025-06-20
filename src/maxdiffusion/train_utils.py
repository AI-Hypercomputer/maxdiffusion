"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import numpy as np
import jax
import jax.numpy as jnp
import queue

from maxdiffusion import max_utils, max_logging


def get_first_step(state):
  return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons"""
  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return next(train_iter)


def validate_train_config(config):
  """Validates the configuration is set correctly for train.py"""

  def _validate_gcs_bucket_name(bucket_name, config_var):
    assert bucket_name, f"Please set {config_var}."
    if "gs://" not in bucket_name:
      max_logging.log(
          f"***WARNING : It is highly recommended that your output_dir uses a gcs directory, currently your output dir is set to {bucket_name}"
      )

  assert config.run_name, "Erroring out, need a real run_name"
  _validate_gcs_bucket_name(config.output_dir, "output_dir")

  assert (
      config.max_train_steps > 0 or config.num_train_epochs > 0
  ), "You must set steps or learning_rate_schedule_steps to a positive interger."

  if config.checkpoint_every > 0 and len(config.checkpoint_dir) <= 0:
    raise AssertionError("Need to set checkpoint_dir when checkpoint_every is set.")

  if config.train_text_encoder and config.cache_latents_text_encoder_outputs:
    raise AssertionError(
        "Cannot train text encoder and cache text encoder outputs."
        " Set either train_text_encoder, or cache_latents_text_encoder_outputs to False"
    )


def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})


_metrics_queue = queue.Queue()
_buffered_step = None
_buffered_metrics = None


def _tensorboard_writer_worker(writer, config):
  """
  A worker function that runs in a separate thread.
  It waits for metrics to appear in the queue and writes them to TensorBoard.
  """
  while True:
    data = _metrics_queue.get()
    if data is None:
      break
    metrics, step = data
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

      if step % config.log_period == 0:
        writer.flush()


def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config):
  """Entry point for all metrics writing in Train's Main.
  TODO: would be better as a Class in the future (that initialized all state!)

  To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
  onto the last metrics and step and only publish when we receive a new metrics and step.
  The logic is that this ensures that Jax is able to queues train_steps and we
  don't block when turning "lazy" Jax arrays into real Python numbers.
  """
  global _buffered_step, _buffered_metrics, _metrics_queue

  if metrics:
    _metrics_queue.put((metrics, step))
  if _buffered_metrics is not None:
    if config.metrics_file:
      max_utils.write_metrics_locally(_buffered_metrics, _buffered_step, config, local_metrics_file)

    if _buffered_step is None:
      raise ValueError(f"When writing metrics, {_buffered_step=} was none")
    write_metrics_to_tensorboard(writer, _buffered_metrics, _buffered_step, config)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(_buffered_metrics, _buffered_step, config, running_gcs_metrics)

  _buffered_step = step
  _buffered_metrics = metrics


def write_metrics_to_tensorboard(writer, metrics, step, config):
  """Writes metrics to tensorboard"""
  if jax.process_index() == 0:
    max_logging.log(
        "completed step: {}, seconds: {:.3f}, TFLOP/s/device: {:.3f}, loss: {:.3f}".format(
            step,
            metrics["scalar"]["perf/step_time_seconds"],
            metrics["scalar"]["perf/per_device_tflops_per_sec"],
            float(metrics["scalar"]["learning/loss"]),
        )
    )
  if jax.process_index() == 0:
    for metric_name in metrics.get("scalar", []):
      writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
    for metric_name in metrics.get("scalars", []):
      writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

  full_log = step % config.log_period == 0

  if full_log and jax.process_index() == 0:
    max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
    writer.flush()


def get_params_to_save(params):
  """Retrieves params from host"""
  return jax.device_get(jax.tree_util.tree_map(lambda x: x, params))


def compute_snr(timesteps, noise_scheduler_state):
  """
  Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
  """
  alphas_cumprod = noise_scheduler_state.common.alphas_cumprod
  sqrt_alphas_cumprod = alphas_cumprod**0.5
  sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

  alpha = sqrt_alphas_cumprod[timesteps]
  sigma = sqrt_one_minus_alphas_cumprod[timesteps]
  # Compute SNR.
  snr = (alpha / sigma) ** 2
  return snr


def generate_timestep_weights(config, num_timesteps):
  timestep_bias_config = config.timestep_bias
  weights = np.ones(num_timesteps)

  # Determine the indices to bias
  num_to_bias = int(timestep_bias_config["portion"] * num_timesteps)
  strategy = timestep_bias_config["strategy"]
  if strategy == "later":
    bias_indices = slice(-num_to_bias, None)
  elif strategy == "earlier":
    bias_indices = slice(0, -num_to_bias)
  elif strategy == "range":
    # Out of possible 1000 steps, we might want to focus on eg. 200-500.
    range_begin = timestep_bias_config["begin"]
    range_end = timestep_bias_config["end"]
    if range_begin < 0:
      raise ValueError(
          "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
      )
    if range_end > num_timesteps:
      raise ValueError(
          "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
      )
    bias_indices = slice(range_begin, range_end)
  else:
    raise ValueError(f"strategy {strategy} is not supported.")

  weights[bias_indices] *= timestep_bias_config["multiplier"]
  weights /= weights.sum()
  return jnp.array(weights)
