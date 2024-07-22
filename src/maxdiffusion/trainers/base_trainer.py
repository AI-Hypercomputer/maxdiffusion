
from abc import abstractmethod
import jax
from maxdiffusion import (
    max_utils,
    maxdiffusion_utils
)

from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (
    BaseStableDiffusionCheckpointer
)

class BaseTrainer(BaseStableDiffusionCheckpointer):
    def __init__(self, config, checkpoint_type):
        BaseStableDiffusionCheckpointer.__init__(self, config, checkpoint_type)

        # sharding
        self.data_sharding = None

        self.per_device_tflops = None

        self.writer = max_utils.initialize_summary_writer(config)

        #Optimizer params
        self.learning_rate_scheduler = None
        self.optimizer = None

    def _create_optimizer(self):
        self.learning_rate_scheduler = max_utils.create_learning_rate_schedule(self.config)
        tx = max_utils.create_optimizer(self.config, self.learning_rate_scheduler)
        return tx

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        return self.optimizer

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
