from typing import List, Optional

import torch
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.layer._operation import gather_forward_split_backward, split_forward_gather_backward


class RobertaPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Bert models
    under pipeline setting.
    """

    @staticmethod
    def roberta_model_forward(
        self: RobertaModel,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,  # this is from the previous stage
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        else:
            input_shape = hidden_states.size()[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        attention_mask = extended_attention_mask
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        hidden_states = hidden_states if hidden_states is not None else None

        if stage_manager.is_first_stage():
            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        # inherit from bert_layer,this should be changed when we add the feature to record hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.encoder.gradient_checkpointing and self.encoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        next_decoder_cache = () if use_cache else None

        start_idx, end_idx = stage_index[0], stage_index[1]
        # layer_outputs
        layer_outputs = hidden_states if hidden_states is not None else None

        # split the input tensor along sequence dimension
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len/TP_size, hidden_size]
        if shard_config is not None and shard_config.enable_sequence_parallelism:
            hidden_states = split_forward_gather_backward(
                hidden_states, dim=1, process_group=shard_config.tensor_parallel_process_group
            )
            if encoder_hidden_states is not None:
                encoder_hidden_states = split_forward_gather_backward(
                    encoder_hidden_states, dim=1, process_group=shard_config.tensor_parallel_process_group
                )

        for idx, encoder_layer in enumerate(self.encoder.layer[start_idx:end_idx], start=start_idx):
            if stage_manager.is_first_stage() and idx == 0:
                encoder_attention_mask = encoder_extended_attention_mask

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[idx] if head_mask is not None else None
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.encoder.gradient_checkpointing and self.encoder.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # When sequence parallelism done, gather the output tensor in forward and split it in backward
        if shard_config is not None and shard_config.enable_sequence_parallelism:
            hidden_states = gather_forward_split_backward(
                hidden_states, dim=1, process_group=shard_config.tensor_parallel_process_group
            )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # end of a stage loop
        sequence_output = hidden_states if hidden_states is not None else None

        if stage_manager.is_last_stage():
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            if not return_dict:
                return (sequence_output, pooled_output) + layer_outputs[1:]
            # return dict is not supported at this moment
            else:
                return BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=next_decoder_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attentions,
                    cross_attentions=all_cross_attentions,
                )

        # output of non-first and non-last stages: must be a dict
        else:
            # intermediate stage always return dict
            return {
                "hidden_states": hidden_states,
            }
