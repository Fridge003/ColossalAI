import pytest
import torch
from torch import distributed as dist

import colossalai
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_grad,
    check_loss,
    check_output_hidden_state,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
)


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):

    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = \
        build_model_from_hybrid_plugin(model_fn, loss_fn, test_config)

    org_loss, org_output, sharded_loss, sharded_output = \
        run_forward_backward_with_hybrid_plugin(
            org_model,
            sharded_model,
            sharded_optimizer,
            data_gen_fn,
            output_transform_fn,
            criterion,
            booster)

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    check_loss(org_loss, sharded_loss, atol=1e-5, rtol=1e-3)

    # check grad
    blip2 = org_model
    sharded_blip2 = sharded_model.unwrap()

    # check grad
    row_layer_for_check = [
        'vision_model.encoder.layers[0].self_attn.qkv', 'qformer.encoder.layer[0].attention.attention.query',
        'language_model.model.decoder.layers[0].self_attn.k_proj'
    ]
    col_layer_for_check = [
        'vision_model.encoder.layers[0].self_attn.projection', 'qformer.encoder.layer[0].attention.output.dense',
        'language_model.model.decoder.layers[0].self_attn.out_proj'
    ]
    check_grad(blip2, sharded_blip2, row_layer_for_check, atol=1e-6, rtol=1e-5, dim=0, verbose=False)
    check_grad(blip2, sharded_blip2, col_layer_for_check, atol=1e-6, rtol=1e-5, dim=1, verbose=False)

    # check weights after optimizer.step()
    org_optimizer.step()
    sharded_optimizer.step()
    if stage_manager is None or stage_manager.is_first_stage():
        check_weight(blip2, sharded_blip2, col_layer_for_check, tp_group, atol=5e-3, rtol=1e-3, dim=1, verbose=False)

    torch.cuda.empty_cache()


@parameterize('test_config', [{
    'tp_size': 4,
    'pp_size': 1,
    'enable_fused_normalization': True,
    'use_lazy_init': False,
    'precision': 'fp32'
}])
def run_blip2_test(test_config):

    sub_model_zoo = model_zoo.get_sub_registry('transformers_blip2')

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


def check_blip2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_blip2_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_blip2():
    spawn(check_blip2, 4)


if __name__ == "__main__":
    test_blip2()
