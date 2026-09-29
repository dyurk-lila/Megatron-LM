# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


def setup_function():
    RouterReplay.global_router_replay_instances.clear()
    torch.manual_seed(0)


def teardown_function():
    RouterReplay.global_router_replay_instances.clear()


def test_record_mode_with_topk_routing_softmax_post():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.RECORD)
    logits = torch.randn(4, 6)
    probs, routing_map = topk_routing_with_score_function(
        logits=logits, topk=2, use_pre_softmax=False, router_replay=rr, score_function="softmax"
    )
    recorded = rr.get_recorded_indices()
    expected_idx = torch.topk(logits, k=2, dim=1).indices
    assert recorded is not None
    assert torch.equal(recorded, expected_idx)
    assert probs.shape == (4, 6)
    assert routing_map.shape == (4, 6)
    assert routing_map.sum(dim=1).eq(2).all()


def test_replay_forward_with_topk_routing_softmax_pre():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logits = torch.randn(3, 5)
    target = torch.tensor([[1, 2], [0, 3], [2, 4]], dtype=torch.long)
    rr.set_target_indices(target)
    probs, routing_map = topk_routing_with_score_function(
        logits=logits, topk=2, use_pre_softmax=True, router_replay=rr, score_function="softmax"
    )
    assert routing_map.sum(dim=1).eq(2).all()
    scores = torch.softmax(logits, dim=-1)
    assert torch.equal(probs.gather(1, target), scores.gather(1, target))


def test_replay_forward_with_topk_routing_softmax_post():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logits = torch.randn(3, 6)
    target = torch.tensor([[1, 2], [0, 5], [3, 4]], dtype=torch.long)
    rr.set_target_indices(target)
    probs, routing_map = topk_routing_with_score_function(
        logits=logits, topk=2, use_pre_softmax=False, router_replay=rr, score_function="softmax"
    )
    selected = torch.softmax(logits.gather(1, target), dim=-1)
    assert torch.equal(probs.gather(1, target), selected)
    assert routing_map.sum(dim=1).eq(2).all()


def test_global_set_get_clear_indices():
    r1 = RouterReplay()
    r2 = RouterReplay()
    t1 = torch.tensor([[0, 1]], dtype=torch.long)
    t2 = torch.tensor([[1, 0]], dtype=torch.long)
    RouterReplay.set_replay_data([t1, t2])
    assert torch.equal(r1.target_topk_idx, t1)
    assert torch.equal(r2.target_topk_idx, t2)
    r1.record_indices(t1)
    r2.record_indices(t2)
    rec = RouterReplay.get_recorded_data()
    assert len(rec) == 2
    assert torch.equal(rec[0], t1)
    assert torch.equal(rec[1], t2)
    RouterReplay.clear_global_indices()
    assert r1.target_topk_idx is None and r2.target_topk_idx is None
    assert r1.get_recorded_indices() is None and r2.get_recorded_indices() is None


def test_global_action_set_and_clear():
    r1 = RouterReplay()
    r2 = RouterReplay()
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    assert r1.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    assert r2.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    RouterReplay.clear_global_router_replay_action()
    assert r1.router_replay_action is None and r2.router_replay_action is None


def test_set_replay_data_length_mismatch():
    _ = RouterReplay()
    with pytest.raises(ValueError):
        RouterReplay.set_replay_data(
            [torch.tensor([[0, 1]], dtype=torch.long), torch.tensor([[1, 0]], dtype=torch.long)]
        )


def test_replay_overrides_precomputed_indices():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logits = torch.randn(3, 6)
    target = torch.tensor([[1, 2], [0, 5], [3, 4]], dtype=torch.long)
    rr.set_target_indices(target)
    probs, _ = topk_routing_with_score_function(
        logits=logits,
        topk=2,
        score_function="sigmoid",
        router_replay=rr,
        precomputed_indices=torch.tensor([[0, 1], [0, 1], [0, 1]]),
    )
    assert torch.equal(probs.nonzero()[:, 1].view(3, 2).sort(dim=1).values, target)


def test_record_mode_records_precomputed_indices():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.RECORD)
    precomputed = torch.tensor([[4, 1], [2, 0]], dtype=torch.long)
    topk_routing_with_score_function(
        logits=torch.randn(2, 6),
        topk=2,
        score_function="sigmoid",
        router_replay=rr,
        precomputed_indices=precomputed,
    )
    assert torch.equal(rr.get_recorded_indices(), precomputed)


@pytest.mark.parametrize(
    "action", [RouterReplayAction.REPLAY_FORWARD, RouterReplayAction.REPLAY_BACKWARD]
)
def test_replay_without_indices_raises(action):
    rr = RouterReplay()
    rr.set_router_replay_action(action)
    with pytest.raises(RuntimeError):
        topk_routing_with_score_function(
            logits=torch.randn(2, 6), topk=2, score_function="softmax", router_replay=rr
        )


# Router configurations of recent MoE model families.
ROUTER_CONFIGS = {
    "deepseek_v3": dict(
        num_experts=256,
        topk=8,
        score_function="sigmoid",
        expert_bias=True,
        num_groups=8,
        group_topk=4,
        scaling_factor=2.5,
    ),
    "deepseek_v4": dict(
        num_experts=384, topk=6, score_function="sqrtsoftplus", expert_bias=True, scaling_factor=2.5
    ),
    "kimi_k2": dict(
        num_experts=384, topk=8, score_function="sigmoid", expert_bias=True, scaling_factor=2.827
    ),
    "kimi_k3": dict(num_experts=896, topk=16, score_function="sigmoid", expert_bias=True),
    "glm5": dict(
        num_experts=256, topk=8, score_function="sigmoid", expert_bias=True, scaling_factor=2.5
    ),
    "mimo_v2": dict(num_experts=256, topk=8, score_function="sigmoid", expert_bias=True),
    "qwen3": dict(num_experts=128, topk=8, score_function="softmax"),
    "qwen3_next": dict(num_experts=512, topk=10, score_function="softmax"),
    "qwen2_moe": dict(num_experts=64, topk=8, score_function="softmax", use_pre_softmax=True),
    "sigmoid_top1": dict(num_experts=16, topk=1, score_function="sigmoid"),
}


def _fused_router_supports(arg_name):
    return moe_utils.fused_topk_with_score_function is not None and (
        moe_utils._fused_router_supports(arg_name)
    )


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_fused_precomputed = pytest.mark.skipif(
    not torch.cuda.is_available() or not _fused_router_supports("precomputed_indices"),
    reason="requires a TE fused router that accepts precomputed_indices",
)
requires_fused_topk_indices = pytest.mark.skipif(
    not torch.cuda.is_available() or not _fused_router_supports("topk_indices"),
    reason="requires a TE fused router that outputs dense topk_indices",
)


def _routing_inputs(config, num_tokens, dtype):
    config = dict(config)
    num_experts = config.pop("num_experts")
    logits = (torch.randn(num_tokens, num_experts, device="cuda") * 2).to(dtype)
    if config.pop("expert_bias", False):
        config["expert_bias"] = torch.randn(num_experts, device="cuda")
    target = torch.rand(num_tokens, num_experts, device="cuda").argsort(dim=-1)[:, : config["topk"]]
    return logits, target, config


def _route(logits, config, fused, router_replay=None, dense_output=False):
    logits = logits.detach().clone().requires_grad_(True)
    probs, routing_output = topk_routing_with_score_function(
        logits=logits, fused=fused, router_replay=router_replay, dense_output=dense_output, **config
    )
    return logits, probs, routing_output


def _replaying(action, target):
    rr = RouterReplay()
    rr.set_target_indices(target)
    rr.set_router_replay_action(action)
    return rr


@requires_fused_precomputed
@pytest.mark.internal
@pytest.mark.parametrize("config_name", sorted(ROUTER_CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "action", [RouterReplayAction.REPLAY_FORWARD, RouterReplayAction.REPLAY_BACKWARD]
)
def test_fused_replay_matches_unfused(config_name, dtype, action):
    logits, target, config = _routing_inputs(ROUTER_CONFIGS[config_name], 4096, dtype)
    logits_ref, probs_ref, map_ref = _route(logits, config, False, _replaying(action, target))
    logits_fused, probs_fused, map_fused = _route(logits, config, True, _replaying(action, target))

    expected_map = torch.zeros_like(map_ref).scatter(1, target, True)
    assert torch.equal(map_ref, expected_map)
    assert torch.equal(map_fused, expected_map)
    torch.testing.assert_close(probs_fused, probs_ref)

    grad = torch.randn_like(probs_ref)
    probs_ref.backward(grad)
    probs_fused.backward(grad)
    torch.testing.assert_close(logits_fused.grad, logits_ref.grad)


@requires_fused_precomputed
@pytest.mark.internal
def test_fused_replay_backward_consumes_queue_in_order():
    logits, first, config = _routing_inputs(ROUTER_CONFIGS["deepseek_v3"], 256, torch.float32)
    second = first.roll(1, dims=1).flip(0)
    rr = RouterReplay()
    rr.set_target_indices(first)
    rr.set_target_indices(second)
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    for target in (first, second):
        _, _, routing_map = _route(logits, config, True, rr)
        assert torch.equal(routing_map, torch.zeros_like(routing_map).scatter(1, target, True))
    assert not rr.replay_backward_list


@requires_fused_topk_indices
@pytest.mark.internal
@pytest.mark.parametrize("config_name", sorted(ROUTER_CONFIGS))
def test_fused_record_matches_unfused(config_name):
    logits, _, config = _routing_inputs(ROUTER_CONFIGS[config_name], 4096, torch.float32)
    recorded = []
    for fused in (False, True):
        rr = RouterReplay()
        rr.set_router_replay_action(RouterReplayAction.RECORD)
        _, _, routing_map = _route(logits, config, fused, rr)
        indices = rr.get_recorded_indices()
        assert torch.equal(routing_map, torch.zeros_like(routing_map).scatter(1, indices, True))
        recorded.append(indices.sort(dim=1).values)
    assert torch.equal(recorded[0], recorded[1])


@requires_fused_topk_indices
@pytest.mark.internal
@pytest.mark.parametrize("config_name", ["deepseek_v3", "qwen3", "qwen2_moe"])
def test_fused_dense_output_matches_unfused(config_name):
    logits, _, config = _routing_inputs(ROUTER_CONFIGS[config_name], 1024, torch.float32)
    _, probs_ref, indices_ref = _route(logits, config, False, dense_output=True)
    _, probs_fused, indices_fused = _route(logits, config, True, dense_output=True)
    assert probs_fused.shape == indices_fused.shape == (1024, config["topk"])
    order_ref, order_fused = indices_ref.argsort(dim=1), indices_fused.argsort(dim=1)
    assert torch.equal(indices_ref.gather(1, order_ref), indices_fused.gather(1, order_fused))
    torch.testing.assert_close(probs_fused.gather(1, order_fused), probs_ref.gather(1, order_ref))


@pytest.mark.skipif(
    not torch.cuda.is_available() or moe_utils.fused_topk_with_score_function is None,
    reason="requires the TE fused router",
)
@pytest.mark.internal
def test_fused_replay_falls_back_without_te_support(monkeypatch):
    monkeypatch.setattr(moe_utils, "_fused_router_supports", lambda arg_name: False)
    logits, target, config = _routing_inputs(ROUTER_CONFIGS["glm5"], 512, torch.float32)
    rr = _replaying(RouterReplayAction.REPLAY_FORWARD, target)
    _, probs, routing_map = _route(logits, config, True, rr)
    _, probs_ref, _ = _route(logits, config, False, _replaying(rr.router_replay_action, target))
    assert torch.equal(routing_map, torch.zeros_like(routing_map).scatter(1, target, True))
    torch.testing.assert_close(probs, probs_ref)
