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
    maxdiffusion_utils
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

        #Optimizer params
        self.learning_rate_scheduler = None

        self.p_train_step = None

    @abstractmethod
    def get_shaped_batch(self, config, pipeline):
       pass

    @abstractmethod
    def compile_train_step(self):
       pass

    @abstractmethod
    def pre_training_steps(self):
        pass

    @abstractmethod
    def post_training_steps(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def training_loop(self):
        pass

    @abstractmethod
    def get_data_shardings(self):
        pass

    @abstractmethod
    def create_scheduler(self):
        pass

    def calculate_tflops(self):
        self.per_device_tflops = maxdiffusion_utils.calculate_unet_tflops(
            self.config, self.pipeline,
            (2 * self.config.per_device_batch_size * jax.local_device_count()),
            self.rng,
            train=True
        )

    def start_training(self):

        # Hook
        self.pre_training_steps()
        # Load checkpoint - will load or create states
        self.load_checkpoint()
        # Create scheduler
        self.create_scheduler()
        # Calculate tflops
        self.calculate_tflops()
        # Load dataset
        self.load_dataset()
        self.get_data_shardings()
        # Compile train_step
        self.compile_train_step()
        # Start training
        self.training_loop()
        # 6. save final checkpoint
        # Hook
        self.post_training_steps()
