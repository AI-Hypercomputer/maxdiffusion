
import jax
from base_trainer import BaseTrainer

class StableDiffusionTrainer(BaseTrainer):
    checkpoint_manager: None

    def __init__(self, checkpoint_manager):
        print("init")
    
    def get_shaped_batch(config, pipeline):
        """Return the shape of the batch - this is what eval_shape would return for the
        output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078.
        This function works with sd1.x and 2.x.
        """
        vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
        total_train_batch_size = config.per_device_batch_size * jax.device_count()
        if config.cache_latents_text_encoder_outputs:
            batch_image_shape = (total_train_batch_size, 4,
                    config.resolution // vae_scale_factor,
                    config.resolution // vae_scale_factor)
            #bs, encoder_input, seq_length
            batch_ids_shape = (total_train_batch_size,
                            pipeline.text_encoder.config.max_position_embeddings,
                            pipeline.text_encoder.config.hidden_size)
        else:
            batch_image_shape = (total_train_batch_size, 3, config.resolution, config.resolution)
            batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings)
        shaped_batch = {}
        shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, jnp.float32)
        shaped_batch["input_ids"] = jax.ShapeDtypeStruct(batch_ids_shape, jnp.float32)
        return shaped_batch
    
    