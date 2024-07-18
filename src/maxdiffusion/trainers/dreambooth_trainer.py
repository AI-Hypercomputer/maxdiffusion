
from pathlib import Path
import time
from maxdiffusion.trainers.base_trainer import BaseTrainer
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, PositionalSharding
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from flax.linen import partitioning as nn_partitioning
import optax
import torch
from torch.utils.data import Dataset
from huggingface_hub.utils import insecure_hashlib
from tqdm import tqdm

from maxdiffusion import (
    FlaxStableDiffusionPipeline,
    max_logging,
    max_utils
)
from maxdiffusion.maxdiffusion_utils import (
    encode
)

from maxdiffusion.input_pipeline.input_pipeline_interface import (
    make_dreambooth_train_iterator
)

from maxdiffusion.train_utils import (
    generate_timestep_weights
)

from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (
    STABLE_DIFFUSION_CHECKPOINT
)

from maxdiffusion.dreambooth.dreambooth_constants import (
    INSTANCE_IMAGE_LATENTS,
    INSTANCE_PROMPT_INPUT_IDS,
    CLASS_IMAGE_LATENTS,
    CLASS_PROMPT_INPUT_IDS
)

class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class DreamboothTrainer(BaseTrainer):
    def __init__(self, config):
        BaseTrainer.__init__(self, config, STABLE_DIFFUSION_CHECKPOINT)

    def get_shaped_batch(self, config, pipeline):
        vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
        total_train_batch_size = config.per_device_batch_size * jax.device_count()
        batch_image_shape = (total_train_batch_size, 4,
                    config.resolution // vae_scale_factor,
                    config.resolution // vae_scale_factor)
        batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings)
        shaped_batch = (
            {
                INSTANCE_IMAGE_LATENTS : jax.ShapeDtypeStruct(batch_image_shape, config.weights_dtype),
                INSTANCE_PROMPT_INPUT_IDS : jax.ShapeDtypeStruct(batch_ids_shape, jnp.int32)
            },{
                CLASS_IMAGE_LATENTS : jax.ShapeDtypeStruct(batch_image_shape, config.weights_dtype),
                CLASS_PROMPT_INPUT_IDS : jax.ShapeDtypeStruct(batch_ids_shape, jnp.int32)
            }
        )
        return shaped_batch
    
    def pre_training_steps(self):
        self.prepare_w_prior_preservation()
    
    def post_training_steps(self):
        return super().post_training_steps()

    def create_other_states(self):

        # text_encoder states have already been replicated
        # in BaseTrainer.create_states_and_shard
        text_encoder_state = train_state.TrainState.create(
            apply_fn=self.pipeline.text_encoder.__call__,
            params=self.__str__params["text_encoder"],
            tx=self.get_optimizer()
    )

    def load_dataset(self):
        self.data_iterator = make_dreambooth_train_iterator(
            self.config,
            self.mesh,
            self.total_train_batch_size,
            self.pipeline.tokenizer,
            self.pipeline.vae,
            self.params["vae"]
        )

    def prepare_w_prior_preservation(self):
        class_images_dir = Path(self.config.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        # just use pmap here
        if cur_class_images < self.config.num_class_images:
            pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
                self.config.pretrained_model_name_or_path,
                safety_checker=None,
                revision=self.config.revision,
                split_head_dim=self.config.split_head_dim
            )
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = self.config.num_class_images - cur_class_images
            max_logging.log(f"Number of class images to sample: {num_new_images}.")
            sample_dataset = PromptDataset(self.config.class_prompt, num_new_images)
            total_sample_batch_size = self.config.per_device_batch_size * jax.local_device_count()
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=total_sample_batch_size)
            for example in tqdm(
                sample_dataloader, desc="Generating class images",
                disable=not jax.process_index() == 0
            ):
                prompt_ids = pipeline.prepare_inputs(example["prompt"])
                prompt_ids = shard(prompt_ids)
                p_params = jax_utils.replicate(params)
                rng = jax.random.split(rng)[0]
                sample_rng = jax.random.split(rng, jax.device_count())
                images = pipeline(prompt_ids, p_params, sample_rng, jit=True).images
                images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
                images = pipeline.numpy_to_pil(np.array(images))

                for i, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

            max_utils.delete_pytree(params)
            del pipeline
    
    def compile_train_step(self):
        self.rng, train_rngs = jax.random.split(self.rng)
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            p_train_step = jax.jit(
                _train_step,
                in_shardings=(self.unet_state_mesh_shardings, None, self.get_my_data_shardings(), None),
                out_shardings=(self.unet_state_mesh_shardings, None, None, None),
                donate_argnums=(0,)
            )
            max_logging.log("Precompiling...")
            s = time.time()
            dummy_batch = self.get_shaped_batch(self.config, self.pipeline)
            p_train_step = p_train_step.lower(self.unet_state,
                                            self.text_encoder_state,
                                            dummy_batch,
                                            train_rngs)
            self.p_train_step = p_train_step.compile()
            max_logging.log(f"Compile time: {(time.time() - s )}")
    
    def training_loop(self):
        num_model_parameters = max_utils.calculate_num_params_from_pytree(self.unet_state.params)
        max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), self.writer)
        max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], self.writer)
        max_utils.add_config_to_summary_writer(self.config, self.writer)

def _train_step(unet_state, text_encoder_state, batch, train_rng, config, pipeline):
    _, gen_dummy_rng = jax.random.split(train_rng)
    sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)
    instance_batch = batch[0]
    class_batch = batch[1]

    instance_latents = instance_batch[INSTANCE_IMAGE_LATENTS]
    instance_input_ids = instance_batch[INSTANCE_PROMPT_INPUT_IDS]
    class_latents = class_batch[CLASS_IMAGE_LATENTS]
    class_input_ids = class_batch[CLASS_PROMPT_INPUT_IDS]

    latents = jnp.concatenate((instance_latents, class_latents), axis=0)
    input_ids = jnp.concatenate((instance_input_ids, class_input_ids), axis=0)
    params = {"text_encoder" : text_encoder_state.params, "unet" : unet_state.params}

    def compute_loss(params):
        encoder_hidden_states = encode(input_ids, pipeline.text_encoder, params["text_encoder"])

        # Sample noise that we'll add to the latents
        noise_rng, timestep_rng = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        if config.timestep_bias["strategy"] == "none":
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                pipeline.scheduler.config.num_train_timesteps,
            )
        else:
            weights = generate_timestep_weights(config, pipeline.scheduler.config.num_train_timesteps)
            timesteps = jax.random.categorical(timestep_bias_rng, logits=jnp.log(weights), shape=(bsz,))

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = pipeline.scheduler.add_noise(params["scheduler"], latents, noise, timesteps)

        # Predict the noise residual and compute loss
        model_pred = pipeline.unet.apply(
            {"params": params["unet"]}, noisy_latents, timesteps, encoder_hidden_states, train=True
        ).sample

        # Get the target for loss depending on the prediction type
        if pipeline.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif pipeline.scheduler.config.prediction_type == "v_prediction":
            target = pipeline.scheduler.get_velocity(params["scheduler"], latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {pipeline.scheduler.config.prediction_type}")

        # This script always uses prior preservation.
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
        target, target_prior = jnp.split(target, 2, axis=0)

        # Compute instance loss
        loss = (target - model_pred) ** 2
        loss = loss.mean()

        # Compute prior loss
        prior_loss = (target_prior - model_pred_prior) ** 2
        prior_loss = prior_loss.mean()

        # Add the prior loss to the instance loss.
        loss = loss + config.prior_loss_weight * prior_loss

        return loss
    
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(params)

    if config.max_grad_norm > 0:
        grad, _ = optax.clip_by_global_norm(config.max_grad_norm).update(grad, unet_state, None)

    new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
    new_text_encoder_state = text_encoder_state.apply_gradients(grads=grad["text_encoder"])
    metrics = {'scalar' : {'learning/loss' : loss}, 'scalars': {}}

    return new_unet_state, new_text_encoder_state, metrics, new_train_rng
