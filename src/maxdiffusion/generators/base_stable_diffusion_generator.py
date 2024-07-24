
import time
from abc import abstractmethod

from maxdiffusion import max_logging

from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (
    BaseStableDiffusionCheckpointer
)

class BaseStableDiffusionGenerator(BaseStableDiffusionCheckpointer):
    def __init__(self, config, checkpoint_type):
        BaseStableDiffusionCheckpointer.__init__(self, config, checkpoint_type)
        self.p_run_inference = None

    @abstractmethod
    def get_inputs(self):
        pass

    @abstractmethod
    def create_scheduler(self):
        pass

    @abstractmethod
    def compile_gen_step(self):
        pass

    @abstractmethod
    def vae_decode(self):
        pass

    @abstractmethod
    def pre_generating_steps(self):
        pass

    @abstractmethod
    def post_generation_steps(self, numpy_images):
        pass

    @abstractmethod
    def generation_loop(self):
        pass

    def initialize(self):
        # Hook
        self.pre_generating_steps()
        # Load checkpoint - will load or create states
        self.load_checkpoint()
        # Create scheduler
        self.create_scheduler()
        # Compile
        self.compile_gen_step()
    
    def generate(self):
        # Start generating
        s = time.time()
        numpy_images = self.generation_loop()
        max_logging.log(f"Generation time: {(time.time() - s)}")
        # Hook
        self.post_generation_steps(numpy_images)