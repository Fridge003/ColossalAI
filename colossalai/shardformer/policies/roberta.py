from functools import partial
from typing import Callable, Dict, List

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

import colossalai.shardformer.layer as col_nn

from ..modeling.roberta import RobertaPipelineForwards
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "RobertaPolicy",
    # "RobertaModelPolicy",
    # "RobertaForCausalLMPolicy",
    # "RobertaForMaskedLMPolicy",
    # "RobertaForSequenceClassificationPolicy",
    # "RobertaForMultipleChoicePolicy",
    # "RobertaForTokenClassificationPolicy",
    # "RobertaForQuestionAnsweringPolicy",
]


class RobertaPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaLayer

        policy = {}
        self.shard_config.enable_sequence_parallelism
        self.shard_config.enable_sequence_overlap
        if self.shard_config.enable_tensor_parallelism:
            policy[RobertaLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.self.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.self.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "attention.self.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.self.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.self.query",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.key",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.value",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.dropout",
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
                        suffix="intermediate.dense",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

            policy[RobertaEmbeddings] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="position_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ]
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # Handle bert layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="attention.output.LayerNorm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.LayerNorm",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=RobertaLayer,
            )

            # handle embedding layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="LayerNorm",
                        target_module=col_nn.FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=RobertaEmbeddings,
            )

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "RobertaModel":
                module = self.model
            else:
                module = self.model.roberta

            layers_per_stage = Policy.distribute_layers(len(module.encoder.layer), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                "forward": partial(
                    new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
                )
            }
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )

        return

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "RobertaModel":
            module = self.model
        else:
            module = self.model.roberta
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage() and (module.pooler is not None):
            held_layers.append(module.pooler)

        return held_layers


class RobertaModelPolicy(RobertaPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.roberta.modeling_roberta import RobertaModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=RobertaModel, new_forward=RobertaPipelineForwards.roberta_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bert model"""
        return []
