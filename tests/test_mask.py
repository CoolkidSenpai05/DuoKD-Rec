"""Changing values at padding positions must not change T1 / T2 / GRU outputs (CLAUDE.md §8)."""
from __future__ import annotations

import pytest
import torch
from conftest import L, N_ITEMS

from src.models.duokd import DuoKD


def _seq():
    s = torch.randint(1, N_ITEMS + 1, (4, L))
    s[0, :5] = 0
    s[1, :2] = 0
    s[2, : L - 1] = 0  # a single real item
    return s


def _heads():
    return ["t1", "t2", "gru"] if "gru" in DuoKD._INPUT and _has_gru() else ["t1", "t2"]


def _has_gru() -> bool:
    try:
        from src.models import gru_rec  # noqa: F401
        return True
    except ImportError:
        return False


def test_padding_rows_do_not_matter(tiny_cfg, feats):
    """Perturb the padding row of every item table (what padding positions read)."""
    tiny_cfg["model"]["heads"] = _heads()
    torch.manual_seed(0)
    model = DuoKD(tiny_cfg, *feats).eval()
    seq = _seq()
    ref = model(seq)
    with torch.no_grad():
        model.item_encoder.emb_id.weight[0].normal_(0, 10.0)  # padding_idx row
    orig = model.item_encoder.forward

    def noisy(*a, **k):
        t = orig(*a, **k)
        for key in ("E_t", "E_v", "E_f"):
            t[key] = t[key].clone()
            t[key][0] = 10 * torch.randn_like(t[key][0])
        return t

    model.item_encoder.forward = noisy
    new = model(seq)
    for h in tiny_cfg["model"]["heads"]:
        torch.testing.assert_close(ref[f"z_{h}"], new[f"z_{h}"], msg=f"{h} depends on padding")


@pytest.mark.parametrize("head", ["t1", "t2"])
def test_tower_padding_inputs(tiny_cfg, feats, head):
    """Perturb the tower input vectors at padding positions directly."""
    torch.manual_seed(0)
    model = DuoKD(tiny_cfg, *feats).eval()
    seq = _seq()
    mask = seq != 0
    x = torch.randn(4, L, tiny_cfg["model"]["hidden"])
    x2 = x.clone()
    x2[~mask] = 100 * torch.randn(int((~mask).sum()), x.shape[-1])
    tower = model.towers[head]
    torch.testing.assert_close(tower(x, mask)[:, -1], tower(x2, mask)[:, -1])
