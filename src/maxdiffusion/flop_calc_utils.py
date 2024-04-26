
def get_conv_flops(input_shape, kernel_size, strides, output_channels):
  output_shape = (input_shape[0], input_shape[1], input_shape[2], output_channels)
  num_params = (kernel_size**2 * input_shape[3] + 1) * output_channels
  flops = ((input_shape[1] * input_shape[2]) / strides**2) * num_params
  return flops, output_shape

def get_timestep_embedding_flops(input_dim, time_embed_dim):
  # there are two dense layers inside FlaxTimestepEmbedding
  # so multiply by 2
  num_params = 2 * (input_dim + 1) * time_embed_dim
  output_shape = (1, time_embed_dim)
  return num_params, output_shape

def get_crossattn_downblocks_flops():
  return 0

def get_downblocks_flops():
  return 0

def get_crossattn_upblocks_flops():
  return 0

def get_upblocks_flops():
  return 0

def calculate_unet_flops(config,
                         sample_shape,
                         encoder_hidden_states_shape,
                         unet_config,
                         train=True):
  multiplier = 6 if train else 2
  print("WARNING: SDXL not currently supported!")
  per_device_batch_size = config.per_device_batch_size
  down_block_types = unet_config["down_block_types"]
  # up_block_types = unet_config["up_block_types"]
  block_out_channels = unet_config["block_out_channels"]

  total_flops = 0
  # simulate a transpose: jnp.transpose(sample, (0, 2, 3, 1))
  sample_shape = (sample_shape[0], sample_shape[2], sample_shape[3], sample_shape[1])
  conv_in_flops, sample_shape = get_conv_flops(sample_shape, 3, 1,
                                   block_out_channels[0])
  conv_in_flops = conv_in_flops * per_device_batch_size * multiplier
  total_flops+=conv_in_flops

  timestep_embedding_flops, t_emb_shape = get_timestep_embedding_flops(block_out_channels[0],
                                                            block_out_channels[0]*4)
  timestep_embedding_flops = timestep_embedding_flops * per_device_batch_size * multiplier
  total_flops+=timestep_embedding_flops

  # TODO (@jfacevedo) : add addition_embed_type calculations for SDXL

  total_down_blocks_flops = 0
  for down_block in down_block_types:
    if down_block == "CrossAttnDownBlock2D":
      down_blocks_flops = get_crossattn_downblocks_flops(sample_shape,
                                                         t_emb_shape,
                                                         encoder_hidden_states_shape)
      down_blocks_flops = down_blocks_flops * per_device_batch_size * multiplier
      total_down_blocks_flops += down_blocks_flops
    elif down_block == "DownBlock2D":
      down_blocks_flops = get_downblocks_flops()
      down_blocks_flops = down_blocks_flops * per_device_batch_size * multiplier
      total_down_blocks_flops += down_blocks_flops
    else:
      raise ValueError(f"{down_block} cannot be found.")

    # TODO - midblock

    # for up_block in up_block_types:
    #   if up_block == "CrossAttnUpBlock2D":
    #     up_block_flops = get_crossattn_upblocks_flops()
    #   elif up_block == "UpBlock2D":
    #     up_blocks_flops = get_upblocks_flops()
    #   else:
    #     raise ValueError(f"{up_block} cannot be found")

  breakpoint()
  return conv_in_flops
