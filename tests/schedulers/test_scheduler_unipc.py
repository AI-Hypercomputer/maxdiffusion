# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: check https://arxiv.org/abs/2302.04867 and https://github.com/wl-zhao/UniPC for more info
# The codebase is modified based on https://github.com/huggingface/diffusers/blob/main/tests/schedulers/test_scheduler_unipc.py

import tempfile

import torch
import jax.numpy as jnp
from typing import Dict, List, Tuple

from maxdiffusion.schedulers.scheduling_unipc_multistep_flax import (
    FlaxUniPCMultistepScheduler,
)
from maxdiffusion import FlaxDPMSolverMultistepScheduler

from .test_scheduler_flax import FlaxSchedulerCommonTest


class FlaxUniPCMultistepSchedulerTest(FlaxSchedulerCommonTest):
    scheduler_classes = (FlaxUniPCMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        sample = torch.rand((batch_size, num_channels, height, width))
        jax_sample= jnp.asarray(sample)
        return jax_sample

    @property
    def dummy_noise_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = torch.arange(num_elems).flip(-1)
        sample = sample.reshape(num_channels, height, width, batch_size)
        sample = sample / num_elems
        sample = sample.permute(3, 0, 1, 2)

        jax_sample= jnp.asarray(sample)
        return jax_sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = torch.arange(num_elems)
        sample = sample.reshape(num_channels, height, width, batch_size)
        sample = sample / num_elems
        sample = sample.permute(3, 0, 1, 2)

        jax_sample= jnp.asarray(sample)
        return jax_sample

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "solver_order": 2,
            "solver_type": "bh2",
            "final_sigmas_type": "sigma_min",
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_model_outputs = [residual + 0.2, residual + 0.15, residual + 0.10]
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            state = scheduler.set_timesteps(
                state, num_inference_steps, sample.shape
            )
            new_state = new_scheduler.set_timesteps(
                new_state, num_inference_steps, sample.shape
            )
            # copy over dummy past residuals
            initial_model_outputs = jnp.stack(dummy_past_model_outputs[
                : scheduler.config.solver_order
            ])
            state = state.replace(model_outputs=initial_model_outputs)
            # Copy over dummy past residuals to new_state as well
            new_state = new_state.replace(model_outputs=initial_model_outputs)


            output_sample, output_state = sample, state
            new_output_sample, new_output_state = sample, new_state
            # Need to iterate through the steps as UniPC maintains history over steps
            # The loop for solver_order + 1 steps is crucial for UniPC's history logic.
            for i in range(time_step, time_step + scheduler.config.solver_order + 1):
                # Ensure time_step + i is within the bounds of timesteps
                if i >= len(output_state.timesteps):
                    break
                t = output_state.timesteps[i]
                step_output = scheduler.step(
                    state=output_state,
                    model_output=residual,
                    timestep=t,  # Pass the current timestep from the scheduler's sequence
                    sample=output_sample,
                    return_dict=True,  # Return a SchedulerOutput dataclass
                )
                output_sample = step_output.prev_sample
                output_state = step_output.state

                new_step_output = new_scheduler.step(
                    state=new_output_state,
                    model_output=residual,
                    timestep=t,  # Pass the current timestep from the scheduler's sequence
                    sample=new_output_sample,
                    return_dict=True,  # Return a SchedulerOutput dataclass
                )
                new_output_sample = new_step_output.prev_sample
                new_output_state = new_step_output.state

            self.assertTrue(
                jnp.allclose(output_sample, new_output_sample, atol=1e-5),
                "Scheduler outputs are not identical",
            )
            # Also assert that states are identical
            self.assertEqual(output_state.step_index, new_output_state.step_index)
            self.assertTrue(jnp.allclose(output_state.timesteps, new_output_state.timesteps))
            self.assertTrue(jnp.allclose(output_state.sigmas, new_output_state.sigmas, atol=1e-5))
            # Comparing model_outputs (history) directly:
            if output_state.model_outputs is not None and new_output_state.model_outputs is not None:
                for out1, out2 in zip(output_state.model_outputs, new_output_state.model_outputs):
                    self.assertTrue(jnp.allclose(out1, out2, atol=1e-5), "Model outputs history not identical")

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_model_outputs = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            state = scheduler.set_timesteps(
                state, num_inference_steps, sample.shape
            )

            # copy over dummy past residuals
            initial_model_outputs = jnp.stack(dummy_past_model_outputs[
                : scheduler.config.solver_order
            ])
            state = state.replace(model_outputs=initial_model_outputs)

            # What is this doing?
            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(new_scheduler, "set_timesteps"):
                new_state = new_scheduler.set_timesteps(
                    new_state, num_inference_steps, sample.shape
                )
            # Copy over dummy past residuals to new_state as well
            new_state = new_state.replace(model_outputs=initial_model_outputs)


            output_sample, output_state = sample, state
            new_output_sample, new_output_state = sample, new_state

            # Need to iterate through the steps as UniPC maintains history over steps
            # The loop for solver_order + 1 steps is crucial for UniPC's history logic.
            for i in range(time_step, time_step + scheduler.config.solver_order + 1):
                # Ensure time_step + i is within the bounds of timesteps
                if i >= len(output_state.timesteps):
                    break

                t = output_state.timesteps[i]

                step_output = scheduler.step(
                    state=output_state,
                    model_output=residual,
                    timestep=t,  # Pass the current timestep from the scheduler's sequence
                    sample=output_sample,
                    return_dict=True,  # Return a SchedulerOutput dataclass
                    **kwargs,
                )
                output_sample = step_output.prev_sample
                output_state = step_output.state

                new_step_output = new_scheduler.step(
                    state=new_output_state,
                    model_output=residual,
                    timestep=t,  # Pass the current timestep from the scheduler's sequence
                    sample=new_output_sample,
                    return_dict=True,  # Return a SchedulerOutput dataclass
                    **kwargs,
                )
                new_output_sample = new_step_output.prev_sample
                new_output_state = new_step_output.state

            self.assertTrue(
                jnp.allclose(output_sample, new_output_sample, atol=1e-5),
                "Scheduler outputs are not identical",
            )
            # Also assert that states are identical
            self.assertEqual(output_state.step_index, new_output_state.step_index)
            self.assertTrue(jnp.allclose(output_state.timesteps, new_output_state.timesteps))
            self.assertTrue(jnp.allclose(output_state.sigmas, new_output_state.sigmas, atol=1e-5))
            # Comparing model_outputs (history) directly:
            if output_state.model_outputs is not None and new_output_state.model_outputs is not None:
                for out1, out2 in zip(output_state.model_outputs, new_output_state.model_outputs):
                    self.assertTrue(jnp.allclose(out1, out2, atol=1e-5), "Model outputs history not identical")


    def full_loop(self, scheduler=None, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        if scheduler is None:
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()
        else:
            state = scheduler.create_state() # Ensure state is fresh for the loop

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)

        for i, t in enumerate(state.timesteps):
            residual = model(sample, t)

            # scheduler.step in common test receives state, residual, t, sample
            step_output = scheduler.step(
                    state=state,
                    model_output=residual,
                    timestep=t,  # Pass the current timestep from the scheduler's sequence
                    sample=sample,
                    return_dict=True,  # Return a SchedulerOutput dataclass
            )
            sample = step_output.prev_sample
            state = step_output.state # Update state for next iteration

        return sample

    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps, sample.shape)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, 1, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, 1, sample, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state() # Create initial state

            sample = self.dummy_sample  # Get sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)
            elif (
                num_inference_steps is not None
                and not hasattr(scheduler, "set_timesteps")
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            # Copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_model_outputs = [
                0.2 * sample,
                0.15 * sample,
                0.10 * sample,
            ]
            initial_model_outputs = jnp.stack(dummy_past_model_outputs[
                : scheduler.config.solver_order
            ])
            state = state.replace(model_outputs=initial_model_outputs)

            time_step_0 = state.timesteps[5]
            time_step_1 = state.timesteps[6]

            output_0 = scheduler.step(state, residual, time_step_0, sample).prev_sample
            output_1 = scheduler.step(state, residual, time_step_1, sample).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            return t.at[t != t].set(0)

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    jnp.allclose(set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {jnp.max(jnp.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {jnp.isnan(tuple_object).any()} and `inf`: {jnp.isinf(tuple_object)}. Dict has"
                        f" `nan`: {jnp.isnan(dict_object).any()} and `inf`: {jnp.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.create_state()
                state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(state, residual, 0, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.create_state()
                state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(state, residual, 0, sample, return_dict=False, **kwargs)
            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)

    def test_switch(self):
        # make sure that iterating over schedulers with same config names gives same results
        # for defaults
        scheduler_config = self.get_scheduler_config()
        scheduler_1 = FlaxUniPCMultistepScheduler(**scheduler_config)
        sample_1 = self.full_loop(scheduler=scheduler_1)
        result_mean_1 = jnp.mean(jnp.abs(sample_1))

        assert abs(result_mean_1.item() - 0.2464) < 1e-3

        scheduler_2 = FlaxUniPCMultistepScheduler(**scheduler_config) # New instance
        sample_2 = self.full_loop(scheduler=scheduler_2)
        result_mean_2 = jnp.mean(jnp.abs(sample_2))

        self.assertTrue(jnp.allclose(result_mean_1, result_mean_2, atol=1e-3)) # Check consistency

        assert abs(result_mean_2.item() - 0.2464) < 1e-3

    def test_timesteps(self):
        for timesteps in [25, 50, 100, 999, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for order in [1, 2, 3]:
            for solver_type in ["bh1", "bh2"]:
                for threshold in [0.5, 1.0, 2.0]:
                    for prediction_type in ["epsilon", "sample"]:
                        with self.assertRaises(NotImplementedError):
                            self.check_over_configs(
                                thresholding=True,
                                prediction_type=prediction_type,
                                sample_max_value=threshold,
                                solver_order=order,
                                solver_type=solver_type,
                            )

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_rescale_betas_zero_snr(self):
        for rescale_zero_terminal_snr in [True, False]:
            self.check_over_configs(rescale_zero_terminal_snr=rescale_zero_terminal_snr)

    def test_solver_order_and_type(self):
        for solver_type in ["bh1", "bh2"]:
            for order in [1, 2, 3]:
                for prediction_type in ["epsilon", "sample"]:
                    self.check_over_configs(
                        solver_order=order,
                        solver_type=solver_type,
                        prediction_type=prediction_type,
                    )
                    sample = self.full_loop(
                        solver_order=order,
                        solver_type=solver_type,
                        prediction_type=prediction_type,
                    )
                    assert not jnp.any(jnp.isnan(sample)), "Samples have nan numbers"


    def test_lower_order_final(self):
        self.check_over_configs(lower_order_final=True)
        self.check_over_configs(lower_order_final=False)

    def test_inference_steps(self):
        for num_inference_steps in [1, 2, 3, 5, 10, 50, 100, 999, 1000]:
            self.check_over_forward(time_step = 0, num_inference_steps=num_inference_steps)

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_mean.item() - 0.2464) < 1e-3

    def test_full_loop_with_karras(self):
        # sample = self.full_loop(use_karras_sigmas=True)
        # result_mean = jnp.mean(jnp.abs(sample))

        # assert abs(result_mean.item() - 0.2925) < 1e-3
        with self.assertRaises(NotImplementedError):
            self.full_loop(use_karras_sigmas=True)

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_mean.item() - 0.1014) < 1e-3

    def test_full_loop_with_karras_and_v_prediction(self):
        # sample = self.full_loop(prediction_type="v_prediction", use_karras_sigmas=True)
        # result_mean = jnp.mean(jnp.abs(sample))

        # assert abs(result_mean.item() - 0.1966) < 1e-3
        with self.assertRaises(NotImplementedError):
            self.full_loop(prediction_type="v_prediction", use_karras_sigmas=True)

    def test_fp16_support(self):
        scheduler_class = self.scheduler_classes[0]
        for order in [1, 2, 3]:
            for solver_type in ["bh1", "bh2"]:
                for prediction_type in ["epsilon", "sample", "v_prediction"]:
                    scheduler_config = self.get_scheduler_config(
                        thresholding=False,
                        dynamic_thresholding_ratio=0,
                        prediction_type=prediction_type,
                        solver_order=order,
                        solver_type=solver_type,
                    )
                    scheduler = scheduler_class(**scheduler_config)
                    state = scheduler.create_state()

                    num_inference_steps = 10
                    model = self.dummy_model()
                    sample = self.dummy_sample_deter.astype(jnp.bfloat16)
                    state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)

                    for i, t in enumerate(state.timesteps):
                        residual = model(sample, t)
                        step_output = scheduler.step(state, residual, t, sample)
                        sample = step_output.prev_sample
                        state = step_output.state
                    # sample is casted to fp32 inside step and output should be fp32.
                    self.assertEqual(sample.dtype, jnp.float32)

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()

        num_inference_steps = 10
        t_start_index = 8

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)

        # add noise
        noise = self.dummy_noise_deter
        timesteps_for_noise = state.timesteps[t_start_index :]
        sample = scheduler.add_noise(state, sample, noise, timesteps_for_noise[:1])

        for i, t in enumerate(timesteps_for_noise):
            residual = model(sample, t)
            step_output = scheduler.step(state, residual, t, sample)
            sample = step_output.prev_sample
            state = step_output.state

        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_sum.item() - 315.5757) < 1e-2, f" expected result sum 315.5757, but get {result_sum}"
        assert abs(result_mean.item() - 0.4109) < 1e-3, f" expected result mean 0.4109, but get {result_mean}"


class FlaxUniPCMultistepScheduler1DTest(FlaxUniPCMultistepSchedulerTest):
    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        width = 8

        torch_sample = torch.rand((batch_size, num_channels, width))
        jax_sample= jnp.asarray(torch_sample)
        return jax_sample

    @property
    def dummy_noise_deter(self):
        batch_size = 4
        num_channels = 3
        width = 8

        num_elems = batch_size * num_channels * width
        sample = torch.arange(num_elems).flip(-1)
        sample = sample.reshape(num_channels, width, batch_size)
        sample = sample / num_elems
        sample = sample.permute(2, 0, 1)

        jax_sample= jnp.asarray(sample)
        return jax_sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        width = 8

        num_elems = batch_size * num_channels * width
        sample = torch.arange(num_elems)
        sample = sample.reshape(num_channels, width, batch_size)
        sample = sample / num_elems
        sample = sample.permute(2, 0, 1)
        jax_sample= jnp.asarray(sample)
        return jax_sample

    def test_switch(self):
        # make sure that iterating over schedulers with same config names gives same results
        # for defaults
        scheduler = FlaxUniPCMultistepScheduler(**self.get_scheduler_config())
        sample = self.full_loop(scheduler=scheduler)
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_mean.item() - 0.2441) < 1e-3

        scheduler = FlaxDPMSolverMultistepScheduler.from_config(scheduler.config)
        scheduler = FlaxUniPCMultistepScheduler.from_config(scheduler.config)

        sample = self.full_loop(scheduler=scheduler)
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_mean.item() - 0.2441) < 1e-3

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_mean.item() - 0.2441) < 1e-3

    def test_full_loop_with_karras(self):
        # sample = self.full_loop(use_karras_sigmas=True)
        # result_mean = jnp.mean(jnp.abs(sample))

        # assert abs(result_mean.item() - 0.2898) < 1e-3
        with self.assertRaises(NotImplementedError):
            self.full_loop(use_karras_sigmas=True)


    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_mean.item() - 0.1014) < 1e-3

    def test_full_loop_with_karras_and_v_prediction(self):
        # sample = self.full_loop(prediction_type="v_prediction", use_karras_sigmas=True)
        # result_mean = jnp.mean(jnp.abs(sample))

        # assert abs(result_mean.item() - 0.1944) < 1e-3
        with self.assertRaises(NotImplementedError):
            self.full_loop(prediction_type="v_prediction", use_karras_sigmas=True)

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()

        num_inference_steps = 10
        t_start_index = 8

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        state = scheduler.set_timesteps(state, num_inference_steps, sample.shape)

        # add noise
        noise = self.dummy_noise_deter
        timesteps_for_noise = state.timesteps[t_start_index :]
        sample = scheduler.add_noise(state, sample, noise, timesteps_for_noise[:1])

        for i, t in enumerate(timesteps_for_noise):
            residual = model(sample, t)
            step_output = scheduler.step(state, residual, t, sample)
            sample = step_output.prev_sample
            state = step_output.state


        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_sum.item() - 39.0870) < 1e-2, f" expected result sum 39.0870, but get {result_sum}"
        assert abs(result_mean.item() - 0.4072) < 1e-3, f" expected result mean 0.4072, but get {result_mean}"

    def test_beta_sigmas(self):
        # self.check_over_configs(use_beta_sigmas=True)
        with self.assertRaises(NotImplementedError):
            self.full_loop(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        #self.check_over_configs(use_exponential_sigmas=True)
        with self.assertRaises(NotImplementedError):
            self.full_loop(use_exponential_sigmas=True)
