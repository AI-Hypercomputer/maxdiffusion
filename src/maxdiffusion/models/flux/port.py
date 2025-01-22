from einops import rearrange

##############################################################################################
# FLUX MODEL PORTING
##############################################################################################


def port_linear(linear, tensors, prefix):
    linear.kernel.value = rearrange(tensors[f"{prefix}.weight"], "i o -> o i")
    linear.bias.value = tensors[f"{prefix}.bias"]
    return linear


def port_modulation(modulation, tensors, prefix):
    modulation.lin = port_linear(
        linear=modulation.lin, tensors=tensors, prefix=f"{prefix}.lin"
    )
    return modulation


def port_rms_norm(rms_norm, tensors, prefix):
    rms_norm.scale.value = tensors[f"{prefix}.scale"]
    return rms_norm


def port_qk_norm(qk_norm, tensors, prefix):
    qk_norm.query_norm = port_rms_norm(
        rms_norm=qk_norm.query_norm,
        tensors=tensors,
        prefix=f"{prefix}.query_norm",
    )
    qk_norm.key_norm = port_rms_norm(
        rms_norm=qk_norm.key_norm,
        tensors=tensors,
        prefix=f"{prefix}.key_norm",
    )
    return qk_norm


def port_self_attention(self_attention, tensors, prefix):
    self_attention.qkv = port_linear(
        linear=self_attention.qkv,
        tensors=tensors,
        prefix=f"{prefix}.qkv",
    )

    self_attention.norm = port_qk_norm(
        qk_norm=self_attention.norm,
        tensors=tensors,
        prefix=f"{prefix}.norm",
    )

    self_attention.proj = port_linear(
        linear=self_attention.proj,
        tensors=tensors,
        prefix=f"{prefix}.proj",
    )

    return self_attention


def port_double_stream_block(double_stream_block, tensors, prefix):
    double_stream_block.img_mod = port_modulation(
        modulation=double_stream_block.img_mod,
        tensors=tensors,
        prefix=f"{prefix}.img_mod",
    )

    # double_stream_block.img_norm1 has no params

    double_stream_block.img_attn = port_self_attention(
        self_attention=double_stream_block.img_attn,
        tensors=tensors,
        prefix=f"{prefix}.img_attn",
    )

    # double_stream_block.img_norm2 has no params

    double_stream_block.img_mlp.layers[0] = port_linear(
        linear=double_stream_block.img_mlp.layers[0],
        tensors=tensors,
        prefix=f"{prefix}.img_mlp.0",
    )
    double_stream_block.img_mlp.layers[2] = port_linear(
        linear=double_stream_block.img_mlp.layers[2],
        tensors=tensors,
        prefix=f"{prefix}.img_mlp.2",
    )

    double_stream_block.txt_mod = port_modulation(
        modulation=double_stream_block.txt_mod,
        tensors=tensors,
        prefix=f"{prefix}.txt_mod",
    )

    # double_stream_block.txt_norm1 has no params

    double_stream_block.txt_attn = port_self_attention(
        self_attention=double_stream_block.txt_attn,
        tensors=tensors,
        prefix=f"{prefix}.txt_attn",
    )

    # double_stream_block.txt_norm2 has no params

    double_stream_block.txt_mlp.layers[0] = port_linear(
        linear=double_stream_block.txt_mlp.layers[0],
        tensors=tensors,
        prefix=f"{prefix}.txt_mlp.0",
    )
    double_stream_block.txt_mlp.layers[2] = port_linear(
        linear=double_stream_block.txt_mlp.layers[2],
        tensors=tensors,
        prefix=f"{prefix}.txt_mlp.2",
    )

    return double_stream_block


def port_single_stream_block(single_stream_block, tensors, prefix):
    single_stream_block.linear1 = port_linear(
        linear=single_stream_block.linear1, tensors=tensors, prefix=f"{prefix}.linear1"
    )
    single_stream_block.linear2 = port_linear(
        linear=single_stream_block.linear2, tensors=tensors, prefix=f"{prefix}.linear2"
    )

    single_stream_block.norm = port_qk_norm(
        qk_norm=single_stream_block.norm, tensors=tensors, prefix=f"{prefix}.norm"
    )

    # single_stream_block.pre_norm has no params

    single_stream_block.modulation = port_modulation(
        modulation=single_stream_block.modulation,
        tensors=tensors,
        prefix=f"{prefix}.modulation",
    )

    return single_stream_block


def port_mlp_embedder(mlp_embedder, tensors, prefix):
    mlp_embedder.in_layer = port_linear(
        linear=mlp_embedder.in_layer, tensors=tensors, prefix=f"{prefix}.in_layer"
    )

    mlp_embedder.out_layer = port_linear(
        linear=mlp_embedder.out_layer, tensors=tensors, prefix=f"{prefix}.out_layer"
    )
    return mlp_embedder


def port_final_layer(final_layer, tensors, prefix):
    # last_layer.norm_final has no params
    final_layer.linear = port_linear(
        linear=final_layer.linear,
        tensors=tensors,
        prefix=f"{prefix}.linear",
    )

    final_layer.adaLN_modulation.layers[1] = port_linear(
        linear=final_layer.adaLN_modulation.layers[1],
        tensors=tensors,
        prefix=f"{prefix}.adaLN_modulation.1",
    )

    return final_layer


def port_flux(flux, tensors):
    flux.img_in = port_linear(
        linear=flux.img_in,
        tensors=tensors,
        prefix="img_in",
    )

    flux.time_in = port_mlp_embedder(
        mlp_embedder=flux.time_in,
        tensors=tensors,
        prefix="time_in",
    )

    flux.vector_in = port_mlp_embedder(
        mlp_embedder=flux.vector_in,
        tensors=tensors,
        prefix="vector_in",
    )

    if flux.params.guidance_embed:
        flux.guidance_in = port_mlp_embedder(
            mlp_embedder=flux.guidance_in,
            tensors=tensors,
            prefix="guidance_in",
        )

    flux.txt_in = port_linear(
        linear=flux.txt_in,
        tensors=tensors,
        prefix="txt_in",
    )

    for i, layer in enumerate(flux.double_blocks.layers):
        layer = port_double_stream_block(
            double_stream_block=layer,
            tensors=tensors,
            prefix=f"double_blocks.{i}",
        )

    for i, layer in enumerate(flux.single_blocks.layers):
        layer = port_single_stream_block(
            single_stream_block=layer,
            tensors=tensors,
            prefix=f"single_blocks.{i}",
        )

    flux.final_layer = port_final_layer(
        final_layer=flux.final_layer,
        tensors=tensors,
        prefix="final_layer",
    )

    return flux
