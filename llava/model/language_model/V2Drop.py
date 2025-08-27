import torch
from typing import Tuple, Callable
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa, Cache, DynamicCache
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from .. import globals

class V2DropLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        self.prev_hidden_states = None
        self.image_token_amount_after_selection = 0
        super().__init__(config)
        # self.model.layers = nn.ModuleList([IQDecoderLayyer(config, layer_idx) for layer_idx in range(LlamaConfig.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                current_hidden_states = hidden_states
                if current_hidden_states.size(1) > 576:
                    image_token_amount = 576
                else:
                    image_token_amount = 0
                if decoder_layer.self_attn.layer_idx == 0:
                    self.image_token_amount_after_selection = image_token_amount
                target_layer_indices = {3,17,22}
                if decoder_layer.self_attn.layer_idx in target_layer_indices and current_hidden_states.size(1) > 1 and self.image_token_amount_after_selection > 0:
                    device = hidden_states.device
                    candidate_indices = torch.arange(35, self.image_token_amount_after_selection + 35, device=device)
                    current_candidates = current_hidden_states[:, candidate_indices, :]
                    prev_candidates = self.prev_hidden_states[:, candidate_indices, :]

                    cos_sim_candidate = F.cosine_similarity(current_candidates, prev_candidates, dim=-1)[0]

                    current_layer = decoder_layer.self_attn.layer_idx
                    if current_layer == 3:
                        keep_ratio = 0.5
                    if current_layer == 17:
                        keep_ratio = 0.5
                    if current_layer == 22:
                        keep_ratio = 0

                    k = max(1, int(self.image_token_amount_after_selection * keep_ratio))
                    topk_values, topk_indices = torch.topk(cos_sim_candidate, k=k, largest=False)
                    selected_indices = candidate_indices[topk_indices]
                    current_seq_length = current_hidden_states.size(1)
                    keep_indexs = torch.cat((torch.arange(35, device=device), selected_indices, torch.arange(self.image_token_amount_after_selection + 35, current_seq_length, device=device)))
                    keep_indexs = keep_indexs.sort().values
                    current_hidden_states = current_hidden_states[:, keep_indexs, :]
                    position_ids = position_ids[:, keep_indexs]

                    if attention_mask is not None:
                        if attention_mask.ndim == 2:
                            attention_mask = attention_mask[:, keep_indexs]
                        else:
                            attention_mask = attention_mask[:, :, keep_indexs, :][..., keep_indexs]

                    if use_cache and past_key_values is not None:
                        layer_idx = decoder_layer.self_attn.layer_idx
                        if layer_idx < len(past_key_values):
                            past_key, past_value = past_key_values[layer_idx]
                        else:
                            past_key, past_value = None, None
                        if past_key is not None and past_value is not None:
                            past_key = past_key[:, :, keep_indexs, :]
                            past_value = past_value[:, :, keep_indexs, :]
                            past_key_values[layer_idx] = (past_key, past_value)

                    self.image_token_amount_after_selection = selected_indices.shape[0]
                self.prev_hidden_states = current_hidden_states
                hidden_states = current_hidden_states

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

