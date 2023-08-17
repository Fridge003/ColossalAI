from functools import partial
from typing import Callable, Dict, List, Tuple

from torch import Tensor, nn

import colossalai.shardformer.layer as col_nn

from ..modeling.blip2 import (
    forward_fn,
    get_blip2_flash_attention_forward,
    get_jit_fused_blip2_QFormer_output_forward,
    get_jit_fused_blip2_QFormer_self_output_forward,
)
from ..modeling.jit import get_jit_fused_dropout_add_func
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ['BlipPolicy', 'BlipModelPolicy', 'Blip2ForConditionalGenerationPolicy']


class BlipPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        from transformers import OPTForCausalLM
        if not isinstance(self.model.language_model, OPTForCausalLM):
            raise ValueError("Currently Shardformer only supports OPTForCausalLM as language model decoder for Blip2.")

        vocab_size = self.model.config.text_config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.language_model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.blip_2.modeling_blip_2 import (
            Blip2Attention,
            Blip2EncoderLayer,
            Blip2QFormerLayer,
            Blip2QFormerModel,
            Blip2QFormerOutput,
            Blip2QFormerSelfOutput,
            Blip2VisionModel,
        )
        from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTForCausalLM

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[Blip2EncoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.num_heads": self.model.config.vision_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.embed_dim": self.model.config.vision_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.qkv",
                        target_module=col_nn.FusedLinear1D_Col,
                        kwargs={
                            "n_fused": 3,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.projection",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.fc1",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.fc2",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )

            policy[Blip2QFormerModel] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ]
            )

            policy[Blip2QFormerLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.attention.num_attention_heads": self.model.config.qformer_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "attention.attention.all_head_size": self.model.config.qformer_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.attention.num_attention_heads": self.model.config.qformer_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.attention.all_head_size": self.model.config.qformer_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.attention.query",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.key",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.value",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.query",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.key",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.value",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate_query.dense",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output_query.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output_query.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

            policy[OPTDecoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.embed_dim": self.model.config.text_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.num_heads": self.model.config.text_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.out_proj",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc1",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc2",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )

            policy[OPTForCausalLM] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="model.decoder.embed_tokens",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="lm_head",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={"gather_output": True},
                    ),
                ]
            )

            policy[Blip2Attention] = ModulePolicyDescription(method_replacement={"forward": forward_fn()})

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # Handle Blip2EncoderLayer layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layer_norm1",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm2",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=Blip2EncoderLayer,
            )

            # handle Blip2VisionModel layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="post_layernorm",
                        target_module=col_nn.FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=Blip2VisionModel,
            )

            # handle Blip2VisionModel layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layernorm",
                        target_module=col_nn.FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=Blip2QFormerModel,
            )

            # handle Blip2QFormerLayer layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="attention.output.LayerNorm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.output.LayerNorm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output_query.LayerNorm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=Blip2QFormerLayer,
            )

            # handle OPTForCausalLM layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="model.decoder.final_layer_norm",
                        target_module=col_nn.FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=OPTForCausalLM,
            )

            # handle OPTDecoderLayer layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="self_attn_layer_norm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_layer_norm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=OPTDecoderLayer,
            )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_blip2_flash_attention_forward(),
                },
                policy=policy,
                target_key=Blip2Attention,
            )

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_blip2_QFormer_self_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=Blip2QFormerSelfOutput,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_blip2_QFormer_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=Blip2QFormerOutput,
            )

        return policy

    def postprocess(self):
        return self.model

    @staticmethod
    def distribute_blip2_layers(num_vision_layers: int, num_qformer_layers: int, num_language_layers: int,
                                num_stages: int) -> Tuple[List[int], int]:
        """
        Distribute blip2 layers into stages when pipeline parallel is used.
        Return the layer distribution as a list and the starting stage of QFormer model and language model.
        """

        if num_vision_layers <= 0 or num_qformer_layers <= 0 or num_language_layers <= 0:
            raise ValueError("The number of layers of vision/qformer/language models should be non-zero 0.")

        # number of layers should be large enough to fill in every stage
        if num_vision_layers + num_qformer_layers + num_language_layers < num_stages:
            raise ValueError("The total number of layers can't be smaller than number of stages.")

        # the number of stages distributed between vision, qformer and language model is optmized in this way:
        # num_vision_stages, num_qformer_stages, num_language_stages = argmin((x - ave) ^ 2 + (y - ave) ^ 2 + (z - ave) ^ 2)
        # s.t. x = num_vision_layers / num_vision_stages, y = num_qformer_layers / num_qformer_stages, z = num_language_layers / num_language_stages
        #      ave = (x + y + z) / 3
        #      num_vision_stages + num_qformer_stages + num_language_stages = num_stages, num_vision_stages >= 1, num_qformer_stages >= 1, num_language_stages >= 1

        def objective(num_vision_stages, num_qformer_stages):
            x = num_vision_layers / num_vision_stages
            y = num_qformer_layers / num_qformer_stages
            z = num_language_layers / (num_stages - num_vision_stages - num_qformer_stages)
            ave = (x + y + z) / 3
            return (x - ave)**2 + (y - ave)**2 + (z - ave)**2

        num_vision_stages, num_qformer_stages = 1, 1
        optimal_diff = 2**31 - 1
        for i in range(1, num_stages - 1):
            for j in range(1, num_stages - 1):
                if (i + j + 1) > num_stages:
                    break
            attempt = objective(i, j)
            if attempt < optimal_diff:
                num_vision_stages, num_qformer_stages = i, j
                optimal_diff = attempt
        num_language_stages = num_stages - num_vision_stages - num_qformer_stages

        vision_distribution = Policy.distribute_layers(num_vision_layers, num_vision_stages)
        qformer_distribution = Policy.distribute_layers(num_qformer_layers, num_qformer_stages)
        language_distribution = Policy.distribute_layers(num_language_layers, num_language_stages)

        qformer_starting_stage = num_vision_stages
        language_starting_stage = num_vision_stages + num_qformer_stages

        return vision_distribution + qformer_distribution + language_distribution, qformer_starting_stage, language_starting_stage

    @staticmethod
    def get_blip2_stage_index(layers_per_stage: List[int], stage: int, qformer_starting_stage: int,
                              language_starting_stage: int) -> Tuple[bool, int, int]:
        """
        Input the distribution of layers among stages, the current stage and the first stage of qformer/language model.
        Return the starting/ending idx of layers belonging to this stage.
        """
        if stage < qformer_starting_stage:
            return Policy.get_stage_index(layers_per_stage[:qformer_starting_stage], stage)
        elif stage < language_starting_stage:
            return Policy.get_stage_index(layers_per_stage[qformer_starting_stage:language_starting_stage],
                                          stage - qformer_starting_stage)
        else:
            return Policy.get_stage_index(layers_per_stage[language_starting_stage:], stage - language_starting_stage)

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None
        stage_manager = self.pipeline_stage_manager

        model = self.model
        vision_model = self.model.vision_model
        qformer_model = self.model.qformer
        language_model = self.model.language_model

        num_vision_layers = len(vision_model.encoder.layers)
        num_qformer_layers = len(qformer_model.encoder.layer)
        num_language_layers = len(language_model.model.decoder.layers)

        held_layers = []
        layers_per_stage, qformer_starting_stage, language_starting_stage = BlipPolicy.distribute_blip2_layers(
            num_vision_layers, num_qformer_layers, num_language_layers, stage_manager.num_stages)
        start_idx, end_idx = BlipPolicy.get_blip2_stage_index(layers_per_stage, stage_manager.stage,
                                                              qformer_starting_stage, language_starting_stage)

        if stage_manager.stage < qformer_starting_stage:
            # current stage is in blip2's vision model
            if stage_manager.is_first_stage():
                held_layers.append(vision_model.embeddings)
            if stage_manager.stage == qformer_starting_stage - 1:
                held_layers.append(vision_model.post_layernorm)
            held_layers.extend(vision_model.encoder.layers[start_idx:end_idx])
        elif stage_manager.stage < language_starting_stage:
            # current stage is in blip2's qformer model
            if stage_manager.stage == qformer_starting_stage:
                held_layers.append(qformer_model.layernorm)
                held_layers.append(qformer_model.dropout)
            if stage_manager.stage == language_starting_stage - 1:
                # put the projection from qformer output to language input at the last stage of qformer
                held_layers.append(model.language_projection)
            held_layers.extend(qformer_model.encoder.layer[start_idx:end_idx])
        else:
            # current stage is in blip2's language model
            if stage_manager.stage == language_starting_stage:
                held_layers.append(language_model.model.decoder.embed_tokens)
                held_layers.append(language_model.model.decoder.embed_positions)
                held_layers.append(language_model.model.decoder.final_layer_norm)
            if stage_manager.is_last_stage():
                held_layers.append(language_model.lm_head)
            held_layers.extend(language_model.model.decoder.layers[start_idx:end_idx])
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
           to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError("set_pipeline_forward method can only be called when pipeline parallel is enabled.")
        stage_manager = self.pipeline_stage_manager

        num_vision_layers = len(self.model.vision_model.encoder.layers)
        num_qformer_layers = len(self.model.qformer.encoder.layer)
        num_language_layers = len(self.model.language_model.model.decoder.layers)

        layers_per_stage, qformer_starting_stage, language_starting_stage = BlipPolicy.distribute_blip2_layers(
            num_vision_layers, num_qformer_layers, num_language_layers, stage_manager.num_stages)
        stage_index = BlipPolicy.get_blip2_stage_index(layers_per_stage, stage_manager.stage, qformer_starting_stage,
                                                       language_starting_stage)

        method_replacement = {
            'forward':
                partial(new_forward,
                        stage_manager=stage_manager,
                        stage_index=stage_index,
                        qformer_starting_stage=qformer_starting_stage,
                        language_starting_stage=language_starting_stage)
        }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)

    def get_shared_params(self) -> List[Dict[int, Tensor]]:

        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None and stage_manager.num_stages > 1:
            num_vision_layers = len(self.model.vision_model.encoder.layers)
            num_qformer_layers = len(self.model.qformer.encoder.layer)
            num_language_layers = len(self.model.language_model.model.decoder.layers)

            _, _, language_starting_stage = BlipPolicy.distribute_blip2_layers(num_vision_layers, num_qformer_layers,
                                                                               num_language_layers,
                                                                               stage_manager.num_stages)

            # embedding and lm_head in language_model should be shared
            language_embed_tokens_weight = self.model.language_model.model.decoder.embed_tokens.weight
            lm_head_weight = self.model.language_model.lm_head.weight
            if id(language_embed_tokens_weight) == id(lm_head_weight):
                return [{
                    language_starting_stage: language_embed_tokens_weight,
                    stage_manager.num_stages - 1: lm_head_weight
                }]

        return []


# Blip2Model
class Blip2ModelPolicy(BlipPolicy):
    def __init__(self) -> None:

        super().__init__()

    def module_policy(self):
        from transformers import Blip2Model
        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=Blip2Model,
                                      new_forward=Blip2PipelineForwards.blip2_model_forward,
                                      policy=policy)

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return super().get_shared_params()


# Blip2ForConditionalGeneration
class Blip2ForConditionalGenerationPolicy(BlipPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers import Blip2ForConditionalGeneration
        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=Blip2ForConditionalGeneration,
                                      new_forward=Blip2PipelineForwards.blip2_for_conditional_generation_forward,
                                      policy=policy)

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return super().get_shared_params()
