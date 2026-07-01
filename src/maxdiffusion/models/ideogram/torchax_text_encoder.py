import jax
from torchax import interop

class TorchaxQwen3VLTextEncoder(interop.JittableModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        pos_2d: jax.Array,
    ) -> jax.Array:
        with interop.default_env():
            input_ids = interop.torch_view(input_ids)
            attention_mask = interop.torch_view(attention_mask)
            pos_2d = interop.torch_view(pos_2d)
            
            output = self.functional_call(
                self._forward_inner,
                params=self.params,
                buffers=self.buffers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pos_2d=pos_2d,
            )
        return interop.jax_view(output)

    @staticmethod
    def _forward_inner(model, input_ids, attention_mask, pos_2d):
        from transformers.masking_utils import create_causal_mask
        import torch
        
        language_model = model.language_model
        inputs_embeds = language_model.embed_tokens(input_ids)

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        import inspect
        sig = inspect.signature(create_causal_mask)
        mask_kwargs = {
            "config": language_model.config,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        if "input_embeds" in sig.parameters:
            mask_kwargs["input_embeds"] = inputs_embeds
        else:
            mask_kwargs["inputs_embeds"] = inputs_embeds
        if "cache_position" in sig.parameters:
            mask_kwargs["cache_position"] = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )

        causal_mask = create_causal_mask(**mask_kwargs)
        if language_model.rotary_emb.inv_freq.device.type != "jax":
            language_model.rotary_emb.inv_freq = (
                language_model.rotary_emb.inv_freq.to(inputs_embeds.device)
            )
        position_embeddings = language_model.rotary_emb(
            inputs_embeds, mrope_position_ids
        )

        from maxdiffusion.models.ideogram.constants import QWEN3_VL_ACTIVATION_LAYERS
        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
        stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
        stacked = torch.permute(stacked, (1, 2, 3, 0))
        batch_size, seq_len = input_ids.shape
        stacked = stacked.reshape(batch_size, seq_len, -1)
        return stacked
