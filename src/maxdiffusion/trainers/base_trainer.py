
from abc import ABC, abstractmethod
from functools import partial
import jax
from jax.sharding import Mesh, PartitionSpec as P, PositionalSharding
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

        if len(config.cache_dir) > 0:
            jax.config.update("jax_compilation_cache_dir", config.cache_dir)
        
        self.config = config
        self.rng = jax.random.PRNGKey(self.config.seed)

        devices_array = max_utils.create_device_mesh(config)
        self.mesh = Mesh(devices_array, self.config.mesh_axes)

        # sharding
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, P(*config.data_sharding))

        self.total_train_batch_size = max_utils.get_global_batch_size(self.config)

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
        if self.optimizer == None:
            self.optimizer = self._create_optimizer()
        return self.optimizer
    
    @abstractmethod
    def get_shaped_batch(self, config, pipeline):
       pass

    @abstractmethod
    def compile_train_step(self, *args, **kwargs):
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
    def post_create_states_and_shard(self):
        """
        Hook to create any other states or additional shardings. 
        Called last in create_states().
        """
        pass

    def calculate_tflops(self):
        self.per_device_tflops = maxdiffusion_utils.calculate_unet_flops(
            self.config, self.pipeline, 
            (2 * self.config.per_device_batch_size * jax.local_device_count()),
            self.rng,
            train=True
        )

    def create_unet_state(self):
        if self.config.train_new_unet:
            unet_variables = jax.jit(self.pipeline.unet.init_weights, static_argnames=["eval_only"])(self.rng, eval_only=False)
        else:
            unet_variables = self.pipeline.unet.init_weights(self.rng, eval_only=True)
        
        if self.config.train_new_unet:
            self.params["unet"] = unet_variables
        else:
            del unet_variables

        unet_state, unet_state_mesh_shardings = max_utils.setup_initial_state(
            model=self.pipeline.unet,
            tx=self.get_optimizer(),
            config=self.config,
            mesh=self.mesh,
            model_params=self.params.get("unet", None),
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_item="unet_state",
            training=True
        )
        self.train_states["unet_state"] = unet_state
        self.state_shardings["unet_state_shardings"] = unet_state_mesh_shardings
    
    def create_vae_state(self):
        vae_state, vae_state_mesh_shardings = max_utils.setup_initial_state(
            model=self.pipeline.vae,
            tx=self.get_optimizer(),
            config=self.config,
            mesh=self.mesh,
            model_params=self.params.get("vae", None),
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_item="vae_state",
            training=True
        )
        self.train_states["vae_state"] = vae_state
        self.state_shardings["vae_state_shardings"] = vae_state_mesh_shardings

    def create_text_encoder_state(self):
        text_encoder_state, text_encoder_mesh_shardings = max_utils.setup_initial_state(
            model=self.pipeline.text_encoder,
            tx=self.get_optimizer(),
            config=self.config,
            mesh=self.mesh,
            model_params=self.params.get("text_encoder", None),
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_item="text_encoder_state",
            training=True
        )
        self.train_states["text_encoder_state"] = text_encoder_state
        self.state_shardings["text_encoder_state_shardings"] = text_encoder_mesh_shardings

    def create_states_and_shard(self):
        """
        Creates train states and shards models accordingly.
        """
        # pipeline should already be initialized here
        # but check anyway.
        if self.pipeline is None:
            self.load_checkpoint()
        
        self.create_unet_state()
        self.create_vae_state()
        self.create_text_encoder_state()

        # # replicate text_encoder params
        # text_encoder_sharding = PositionalSharding(self.mesh.devices).replicate()
        # partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=text_encoder_sharding)
        # self.params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, self.params["text_encoder"])

        self.post_create_states_and_shard()

    def start_trainining(self):
        
        # Hook
        self.pre_training_steps()
        # Load checkpoint
        self.load_checkpoint()
        # Create or load train states and shard models
        self.create_states_and_shard()
        # Calculate tflops
        self.calculate_tflops()
        # Load dataset
        self.load_dataset()
        # Compile train_step
        self.compile_train_step()
        # Start training
        self.training_loop()
        # 6. save final checkpoint
        # Hook
        self.post_training_steps()
