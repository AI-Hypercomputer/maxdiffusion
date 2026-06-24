"""Generate an image with Z-Image or Z-Image-Turbo."""

from absl import app
import jax

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh, get_flash_block_sizes
from maxdiffusion.pipelines.z_image import ZImagePipeline
from jax.sharding import Mesh


def main(argv):
  pyconfig.initialize(argv)
  config = pyconfig.config
  mesh = Mesh(create_device_mesh(config), config.mesh_axes)
  pipeline = ZImagePipeline.from_pretrained(
      config.pretrained_model_name_or_path,
      jax.random.key(config.seed),
      attention_kernel=config.attention,
      mesh=mesh,
      flash_block_sizes=get_flash_block_sizes(config),
      dtype=config.activations_dtype,
      weights_dtype=config.weights_dtype,
      logical_axis_rules=config.logical_axis_rules,
  )
  image = pipeline(
      config.prompt,
      height=config.resolution,
      width=config.resolution,
      num_inference_steps=config.num_inference_steps,
      seed=config.seed,
  )
  image.save(config.output_file)


if __name__ == "__main__":
  app.run(main)
