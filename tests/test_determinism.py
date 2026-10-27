"""Same seed -> same losses and parameters after a few optimisation steps (CLAUDE.md §8)."""
from __future__ import annotations

import torch
from conftest import L, N_ITEMS

from src.build import build_loss
from src.models.duokd import DuoKD
from src.utils import set_seed


def _run(cfg: dict, feats, seed: int, steps: int = 3) -> tuple[list[float], dict[str, torch.Tensor]]:
    set_seed(seed)
    model = DuoKD(cfg, *feats).train()      # train mode: dropout + modality dropout are random
    loss_fn = build_loss(cfg)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-2)
    losses = []
    for _ in range(steps):
        seq = torch.randint(0, N_ITEMS + 1, (6, L))
        batch = {"seq": seq, "target": torch.randint(1, N_ITEMS + 1, (6,))}
        loss, _ = loss_fn(model(seq), batch, epoch=100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_determinism(tiny_cfg, feats):
    tiny_cfg["model"]["heads"] = ["t1", "t2", "gru"]
    tiny_cfg["model"]["dropout"] = 0.2
    tiny_cfg["item_encoder"]["modality_dropout"] = 0.3
    l1, s1 = _run(tiny_cfg, feats, seed=7)
    l2, s2 = _run(tiny_cfg, feats, seed=7)
    assert l1 == l2
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    l3, _ = _run(tiny_cfg, feats, seed=8)
    assert l3 != l1, "different seeds should differ"


def test_train_loader_order_deterministic():
    from src.trainer import make_train_loader

    data = {"train_seqs": [list(range(1, 12)), list(range(3, 9)), list(range(2, 20))]}
    cfg = {"data": {"max_len": 5}, "train": {"batch_size": 4, "num_workers": 0}}
    a = [b["target"].tolist() for b in make_train_loader(data, cfg, seed=3)]
    b = [b["target"].tolist() for b in make_train_loader(data, cfg, seed=3)]
    assert a == b
