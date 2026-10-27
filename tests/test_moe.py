"""MoE gates, expert load and balance loss (CLAUDE.md §8)."""
from __future__ import annotations

import torch

from src.layers.moe import MoE
from src.losses import gate_stats, moe_balance_loss


def test_soft_gates_sum_to_one():
    torch.manual_seed(0)
    moe = MoE(8, 6, n_experts=4, d_hidden=12)
    x = torch.randn(10, 8)
    y, probs, gates = moe(x)
    assert y.shape == (10, 6) and probs.shape == (10, 4)
    torch.testing.assert_close(gates.sum(-1), torch.ones(10))
    torch.testing.assert_close(gates, probs)


def test_soft_output_is_gate_weighted_sum():
    torch.manual_seed(0)
    moe = MoE(8, 6, n_experts=3, d_hidden=12).eval()
    x = torch.randn(5, 8)
    y, _, g = moe(x)
    manual = torch.zeros(5, 6)
    for k in range(3):
        h = torch.nn.functional.gelu(x @ moe.w1[k] + moe.b1[k])
        manual += g[:, k:k + 1] * (h @ moe.w2[k] + moe.b2[k])
    torch.testing.assert_close(y, manual)


def test_topk_gates():
    torch.manual_seed(0)
    moe = MoE(8, 6, n_experts=4, d_hidden=12, top_k=2)
    _, probs, gates = moe(torch.randn(10, 8))
    assert ((gates > 0).sum(-1) == 2).all()
    torch.testing.assert_close(gates.sum(-1), torch.ones(10))
    torch.testing.assert_close(probs.sum(-1), torch.ones(10))


def test_load_and_balance_loss():
    mask = torch.tensor([True, True, True, False])
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    P, ent = gate_stats(probs, mask)
    torch.testing.assert_close(P, torch.tensor([2 / 3, 1 / 3]))  # padding row ignored
    assert abs(float(ent)) < 1e-6
    # soft: n * sum P^2, minimum 1 at uniform
    uni = torch.full((4, 2), 0.5)
    torch.testing.assert_close(moe_balance_loss(uni, uni, mask, None), torch.tensor(1.0))
    lb = moe_balance_loss(probs, probs, mask, None)
    torch.testing.assert_close(lb, torch.tensor(2 * (4 / 9 + 1 / 9)))
    assert lb > 1.0
    # random gates never go below the uniform minimum
    for _ in range(20):
        p = torch.softmax(torch.randn(16, 4), -1)
        assert moe_balance_loss(p, p, torch.ones(16, dtype=torch.bool), None) >= 1.0 - 1e-6
    # top-k: n * sum f_k P_k, uniform = 1
    all4 = torch.ones(4, dtype=torch.bool)
    top1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])  # balanced assignment
    torch.testing.assert_close(moe_balance_loss(uni, top1, all4, 1), torch.tensor(1.0))
    skew = torch.tensor([[1.0, 0.0]] * 4)
    torch.testing.assert_close(moe_balance_loss(skew, skew, all4, 1), torch.tensor(2.0))


def test_plain_mlp():
    moe = MoE(8, 6, n_experts=4, d_hidden=12, plain_mlp=True)
    y, p, g = moe(torch.randn(3, 8))
    assert y.shape == (3, 6) and p.shape == (3, 1) and (g == 1).all()
