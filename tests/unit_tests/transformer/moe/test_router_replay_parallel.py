# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Routing replay through a GPT MoE model across TP/SP, CP, EP, ETP, and PP meshes.

Every router must route exactly to the replayed experts in the forward pass and in the
MoE recompute during backward, with the fused and unfused routers agreeing.
"""

from collections import defaultdict, deque

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank
from tests.unit_tests.test_utilities import Utils

NUM_LAYERS = 4
NUM_EXPERTS = 8
TOPK = 2
SEQ_LENGTH = 64
MICRO_BATCH_SIZE = 2
NUM_MICROBATCHES = 4
VOCAB_SIZE = 128

# (tp, pp, cp, ep, etp) on 8 GPUs; sequence parallelism is on whenever tp > 1.
MESHES = [
    pytest.param(2, 2, 2, 2, 1, id="tp2_pp2_cp2_ep2"),
    pytest.param(4, 1, 2, 4, 1, id="tp4_cp2_ep4"),
    pytest.param(1, 1, 1, 8, 1, id="ep8"),
    pytest.param(2, 1, 1, 4, 2, id="tp2_ep4_etp2"),
    pytest.param(2, 2, 1, 2, 2, id="tp2_pp2_ep2_etp2"),
    pytest.param(1, 4, 1, 2, 1, id="pp4_ep2"),
    pytest.param(1, 2, 1, 4, 1, id="pp2_ep4"),
]

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 8, reason="requires 8 GPUs"
    ),
    pytest.mark.skipif(
        moe_utils.fused_topk_with_score_function is None
        or not moe_utils._fused_router_supports("precomputed_indices"),
        reason="requires a TE fused router that accepts precomputed_indices",
    ),
]


def _replay_targets(tokens, positions, layer_number):
    """Distinct experts per token that vary with token id, position, and layer."""
    base = (tokens * 31 + positions * 7 + layer_number * 13) % NUM_EXPERTS
    return (base.unsqueeze(-1) + torch.arange(TOPK, device=tokens.device)) % NUM_EXPERTS


def _router_view(batch_tensor, config):
    """Flatten a CP-local [b, s] tensor into the token order the router sees."""
    tensor = batch_tensor.t()
    if config.sequence_parallel:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tensor = tensor.chunk(tp_size, dim=0)[tp_rank]
    return tensor.reshape(-1)


def _build_model(tp, pp, cp, ep, etp, score_function):
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=64,
        num_attention_heads=8,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=64,
        add_bias_linear=False,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_score_function=score_function,
        moe_router_enable_expert_bias=score_function == "sigmoid",
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_enable_routing_replay=True,
        recompute_granularity="selective",
        recompute_modules=["moe"],
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        sequence_parallel=tp > 1,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=NUM_EXPERTS, moe_grouped_gemm=True
        ),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LENGTH,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    model = model.bfloat16().cuda()
    model.model_type = ModelType.encoder_or_decoder
    return model, config


def _microbatches():
    # Same data on every rank of a DP replica; vary it across DP replicas.
    generator = torch.Generator(device="cuda").manual_seed(
        1000 + parallel_state.get_data_parallel_rank()
    )
    for _ in range(NUM_MICROBATCHES):
        tokens = torch.randint(
            VOCAB_SIZE, (MICRO_BATCH_SIZE, SEQ_LENGTH + 1), device="cuda", generator=generator
        )
        positions = torch.arange(SEQ_LENGTH, device="cuda").expand(MICRO_BATCH_SIZE, -1)
        yield dict(tokens=tokens[:, :-1], labels=tokens[:, 1:], position_ids=positions)


class _ReplayChecker:
    """Sets per-microbatch replay targets and checks every routing decision against them."""

    def __init__(self, model, config):
        self.config = config
        self.forward_expected = {}
        self.backward_expected = defaultdict(deque)
        self.num_checks = defaultdict(int)
        self.routers = [m for m in model.modules() if isinstance(m, TopKRouter)]
        assert (
            len(self.routers)
            == NUM_LAYERS // parallel_state.get_pipeline_model_parallel_world_size()
        )
        self.handles = [r.register_forward_hook(self._check) for r in self.routers]

    def set_targets(self, batch):
        tokens = _router_view(batch["tokens"], self.config)
        positions = _router_view(batch["position_ids"], self.config)
        for router in self.routers:
            layer_number = router.layer_number
            targets = _replay_targets(tokens, positions, layer_number)
            router.router_replay.set_target_indices(targets)
            expected = torch.zeros(targets.shape[0], NUM_EXPERTS, dtype=torch.bool, device="cuda")
            expected.scatter_(1, targets, True)
            self.forward_expected[layer_number] = expected
            self.backward_expected[layer_number].append(expected)
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    def _check(self, router, inputs, outputs):
        action = router.router_replay.router_replay_action
        if action == RouterReplayAction.REPLAY_FORWARD:
            expected = self.forward_expected[router.layer_number]
        else:
            assert action == RouterReplayAction.REPLAY_BACKWARD
            expected = self.backward_expected[router.layer_number].popleft()
        routing_map = outputs[1]
        assert torch.equal(routing_map, expected), (
            f"layer {router.layer_number} {action}: "
            f"{(routing_map == expected).all(dim=1).float().mean().item():.4f} of tokens replayed"
        )
        self.num_checks[action] += 1

    def remove(self):
        for handle in self.handles:
            handle.remove()


def _loss_func(loss_mask, output_tensor):
    loss = (output_tensor.float() * loss_mask).sum()
    num_tokens = loss_mask.sum().to(torch.int)
    return loss, num_tokens, {"lm loss": torch.cat([loss.detach().view(1), num_tokens.view(1)])}


def _run_step(model, config, checker):
    data = _microbatches()

    def forward_step(data_iterator, model):
        batch = get_batch_on_this_cp_rank(
            dict(next(data_iterator), cu_seqlens=None),
            is_hybrid_cp=False,
            cp_group=parallel_state.get_context_parallel_group(),
        )
        checker.set_targets(batch)
        output = model(batch["tokens"], batch["position_ids"], None, labels=batch["labels"])
        # Backward of this microbatch recomputes its MoE layers from the replay FIFO.
        output.register_hook(
            lambda grad: RouterReplay.set_global_router_replay_action(
                RouterReplayAction.REPLAY_BACKWARD
            )
            or grad
        )
        loss_mask = torch.ones_like(batch["labels"], dtype=torch.float32)
        return output, lambda out: _loss_func(loss_mask, out)

    model.zero_grad(set_to_none=True)
    losses = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=data,
        model=[model],
        num_microbatches=NUM_MICROBATCHES,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
    )
    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return losses, grads


@pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
@pytest.mark.parametrize("tp,pp,cp,ep,etp", MESHES)
def test_fused_routing_replay_across_meshes(tp, pp, cp, ep, etp, score_function):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    RouterReplay.clear_global_router_replay_instances()
    try:
        model_parallel_cuda_manual_seed(123)
        model, config = _build_model(tp, pp, cp, ep, etp, score_function)
        checker = _ReplayChecker(model, config)
        results = {}
        for fused in (False, True):
            config.moe_router_fusion = fused
            checker.num_checks.clear()
            results[fused] = _run_step(model, config, checker)
            num_routers = len(checker.routers) * NUM_MICROBATCHES
            assert checker.num_checks[RouterReplayAction.REPLAY_FORWARD] == num_routers
            assert checker.num_checks[RouterReplayAction.REPLAY_BACKWARD] == num_routers
            assert not any(checker.backward_expected.values())
        checker.remove()

        (losses_ref, grads_ref), (losses_fused, grads_fused) = results[False], results[True]
        for ref, fused in zip(losses_ref, losses_fused):
            torch.testing.assert_close(fused["lm loss"], ref["lm loss"], rtol=1e-3, atol=1e-3)
        assert grads_ref.keys() == grads_fused.keys()
        for name, grad in grads_ref.items():
            torch.testing.assert_close(grads_fused[name], grad, rtol=5e-2, atol=5e-3, msg=name)
    finally:
        RouterReplay.clear_global_router_replay_instances()
        Utils.destroy_model_parallel()
