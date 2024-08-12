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

from abc import abstractmethod
import jax
from maxdiffusion import (
    max_utils,
    maxdiffusion_utils,
    max_logging
)

from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (
    BaseStableDiffusionCheckpointer
)

class BaseStableDiffusionTrainer(BaseStableDiffusionCheckpointer):
    def __init__(self, config, checkpoint_type):
        BaseStableDiffusionCheckpointer.__init__(self, config, checkpoint_type)

        # sharding
        self.data_sharding = None

        self.per_device_tflops = None

        self.writer = max_utils.initialize_summary_writer(config)

        self.p_train_step = None

    @abstractmethod
    def get_shaped_batch(self, config, pipeline):
       pass

    @abstractmethod
    def compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings):
       pass

    @abstractmethod
    def pre_training_steps(self):
        pass

    @abstractmethod
    def post_training_steps(self, pipeline, params, train_states):
        pass

    @abstractmethod
    def load_dataset(self, pipeline, params, train_states):
        pass

    @abstractmethod
    def training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler):
        pass

    @abstractmethod
    def get_data_shardings(self):
        pass

    @abstractmethod
    def create_scheduler(self, pipeline, params):
        pass

    def calculate_tflops(self, pipeline, params):
        per_device_tflops = maxdiffusion_utils.calculate_unet_tflops(
            self.config, pipeline,
            (2 * self.config.per_device_batch_size * jax.local_device_count()),
            self.rng,
            train=True
        )
        max_logging.log(f"UNET per device TFLOPS: {per_device_tflops}")
        return per_device_tflops

    def start_training(self):

        # Hook
        self.pre_training_steps()
        # Load checkpoint - will load or create states
        pipeline, params = self.load_checkpoint()
        # create train states
        train_states = {}
        state_shardings = {}
        unet_state, unet_state_mesh_shardings, unet_learning_rate_scheduler = self.create_unet_state(
            # ambiguous here, but if self.params.get("unet") doesn't exist
            # Then its 1 of 2 scenarios:
            # 1. unet state will be loaded directly from orbax
            # 2. a new unet is being trained from scratch.
            pipeline=pipeline,
            params=params,
            checkpoint_item_name="unet_state",
            is_training=True,
        )
        train_states["unet_state"] = unet_state
        state_shardings["unet_state_shardings"] = unet_state_mesh_shardings
        vae_state, vae_state_mesh_shardings = self.create_vae_state(
            pipeline=pipeline,
            params=params,
            checkpoint_item_name="vae_state",
            is_training=False
        )

        train_states["vae_state"] = vae_state
        state_shardings["vae_state_shardings"] = vae_state_mesh_shardings

        text_encoder_state, text_encoder_state_mesh_shardings = self.create_text_encoder_state(
            pipeline=pipeline,
            params=params,
            checkpoint_item_name="text_encoder_state",
            is_training=self.config.train_text_encoder
            )
        train_states["text_encoder_state"] = text_encoder_state
        state_shardings["text_encoder_state_shardings"] = text_encoder_state_mesh_shardings
        if hasattr(pipeline, "text_encoder_2"):
            text_encoder_2_state, text_encoder_2_state_mesh_shardings = self.create_text_encoder_2_state(
                pipeline,
                params,
                "text_encoder_2_state",
                is_training=self.config.train_text_encoder
            )
            train_states["text_encoder_2_state"] = text_encoder_2_state
            state_shardings["text_encoder_2_state_shardings"] = text_encoder_2_state_mesh_shardings

        # Create scheduler
        noise_scheduler, noise_scheduler_state =self.create_scheduler(pipeline, params)
        pipeline.scheduler = noise_scheduler
        params["scheduler"] = noise_scheduler_state

        # Calculate tflops
        per_device_tflops = self.calculate_tflops(pipeline, params)
        self.per_device_tflops = per_device_tflops

        # Load dataset
        data_iterator = self.load_dataset(pipeline, params, train_states)

        data_shardings = self.get_data_shardings()
        # Compile train_step
        p_train_step = self.compile_train_step(
            pipeline,
            params,
            train_states,
            state_shardings,
            data_shardings
        )
        # Start training
        train_states = self.training_loop(p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler)
        # 6. save final checkpoint
        # Hook
        self.post_training_steps(pipeline, params, train_states)
