"""Baseline interface tests (M12, PROMPT §2.5), parametrised over every registered baseline.

Tiny synthetic catalog (N = 30, L = 8, D = 16) with random frozen text / image / category features.
"""
from __future__ import annotations

import copy
import io
import os
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.baselines.base import BaseSeqRec
from src.baselines.data import TimedTrainDataset
from src.baselines.registry import REGISTRY, baseline_class
from src.metrics import mask_history
from src.utils import load_config, set_seed

N, L, D, D_T, D_V, N_CAT, B = 30, 8, 16, 24, 20, 7, 16
MODELS = list(REGISTRY)
FEATURE_BUFFERS = ("plm_embedding", "pvm_embedding", "txt_embedding", "img_embedding", "cat_embedding")


def _data() -> dict:
    rng = random.Random(0)
    seqs, times = [], []
    for _ in range(24):
        n = rng.randint(5, 14)
        seqs.append(rng.sample(range(1, N + 1), n))
        t, ts = 1_300_000_000, []
        for _ in range(n):
            t += rng.choice([0, 0, 86400, 3 * 86400, 40 * 86400])  # ties (gap 0) like Amazon day data
            ts.append(t)
        times.append(ts)
    return {"n_users": len(seqs), "n_items": N,
            "train_seqs": [s[:-2] for s in seqs], "valid_items": [s[-2] for s in seqs],
            "test_items": [s[-1] for s in seqs], "train_ts": [t[:-2] for t in times],
            "valid_ts": [t[-2] for t in times], "test_ts": [t[-1] for t in times]}


def _feats() -> dict:
    g = torch.Generator().manual_seed(0)
    t, v = torch.randn(N + 1, D_T, generator=g), torch.randn(N + 1, D_V, generator=g)
    t[0] = v[0] = 0
    cat = torch.rand(N + 1, N_CAT, generator=g) < 0.3
    cat[0] = False
    return {"text": t, "image": v, "category": cat}


def _cfg(name: str, no_dropout: bool = False) -> dict:
    cfg = load_config(f"configs/baselines/{name}.yaml", [f"data.max_len={L}", f"model.hidden={D}"])
    if no_dropout:
        cfg["model"]["dropout"] = 0.0
        for k, v in cfg["model"].get(name, {}).items():
            if "dropout" in k:
                cfg["model"][name][k] = 0.0
    return cfg


def _model(name: str, seed: int = 0, no_dropout: bool = False) -> BaseSeqRec:
    set_seed(seed)
    return baseline_class(name)(_cfg(name, no_dropout), _data(), _feats())


def _batch(n: int = B) -> dict:
    data = _data()
    ds = TimedTrainDataset(data["train_seqs"], data["train_ts"], L)
    idx = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    return next(iter(DataLoader(torch.utils.data.Subset(ds, idx), batch_size=n)))


# 1. shapes, -inf padding column, finite loss ------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_shapes_and_finite(name):
    m = _model(name)
    batch = _batch()
    m.train()
    out = m.compute_loss(batch)
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    m.eval()
    with torch.no_grad():
        s = m.score_all(batch)
    assert s.shape == (B, N + 1)
    assert torch.isneginf(s[:, 0]).all()
    assert torch.isfinite(s[:, 1:]).all()


# 2. causality ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_causal(name):
    cls = baseline_class(name)
    if not cls.causal:
        pytest.skip(f"{name} is not causal by design (FFT over the window / bidirectional attention)")
    m = _model(name).eval()
    g = torch.Generator().manual_seed(1)
    seq = torch.randint(1, N + 1, (4, L), generator=g)               # full windows: no padding
    ts = torch.cumsum(torch.randint(0, 5, (4, L), generator=g) * 86400.0, 1).double() + 1.3e9
    b1 = {"seq": seq, "ts": ts, "target": seq[:, -1]}
    seq2 = seq.clone()
    seq2[:, 5:] = torch.randint(1, N + 1, (4, L - 5), generator=g)
    b2 = {"seq": seq2, "ts": ts, "target": seq[:, -1]}
    with torch.no_grad():
        o1, o2 = m.position_outputs(b1), m.position_outputs(b2)
    torch.testing.assert_close(o1[:, :5], o2[:, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(o1[:, 5:], o2[:, 5:])


# 3. BERT4Rec masking -------------------------------------------------------------------------------
def test_bert4rec_masking():
    m = _model("bert4rec").train()
    batch = _batch()
    torch.manual_seed(0)
    for _ in range(20):
        inp, labels = m.make_train_input(batch)
        sel = labels > 0
        assert sel.any(1).sum() > 0
        assert (inp[sel] == m.mask_token).all(), "a predicted (masked) item is visible in the input"
        assert (labels[sel] <= N).all() and (inp[~sel] != m.mask_token).all()
    # eval: the history + [MASK]; the target is never in the input; [MASK] is never ranked
    data = _data()
    seq = torch.tensor([[0] * (L - len(s[-(L - 1):])) + s[-(L - 1):] for s in data["train_seqs"][:4]])
    tgt = torch.tensor(data["valid_items"][:4])
    inp, pos = m.eval_input(seq)
    assert (inp[torch.arange(4), pos - 1] == m.mask_token).all()
    assert not (inp == tgt.unsqueeze(1)).any()
    with torch.no_grad():
        assert m.eval().score_all({"seq": seq}).shape[1] == N + 1  # ids 0..N only: mask id N+1 absent


# 4. padding invariance ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_padding_invariance(name):
    cls = baseline_class(name)
    if not cls.padding_invariant:
        pytest.skip(f"{name}: padding positions enter the FFT of the official model (not padding invariant)")
    m = _model(name).eval()
    batch = _batch()
    assert (batch["seq"] == 0).any(), "test batch must contain left padding"
    with torch.no_grad():
        ref = m.score_all(batch)
        for mod in m.modules():                                      # what padding positions read
            if isinstance(mod, torch.nn.Embedding) and mod.padding_idx == 0:
                mod.weight[0].normal_(0, 10.0)
        for nm in FEATURE_BUFFERS:
            if hasattr(m, nm):
                getattr(m, nm)[0] = 10 * torch.randn_like(getattr(m, nm)[0].float()).to(getattr(m, nm).dtype)
        b2 = dict(batch)
        b2["ts"] = torch.where(batch["seq"] == 0, torch.rand_like(batch["ts"]) * 1e9, batch["ts"])
        new = m.score_all(b2)
    torch.testing.assert_close(ref[:, 1:], new[:, 1:], atol=1e-5, rtol=1e-5)


# 5. overfit one batch ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_overfit_batch(name):
    m = _model(name, no_dropout=True)
    batch = _batch()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    first = None
    for step in range(300):
        if hasattr(m, "on_epoch_start") and step % 50 == 0:
            m.on_epoch_start(step)
        loss = m.compute_loss(batch)["loss"]
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        ce = m.compute_loss(batch)["loss"].item() if name != "bert4rec" else loss.item()
        r1 = (m.score_all(batch).argmax(1) == batch["target"]).float().mean().item()
    assert ce < 0.5 * first, f"loss {first:.3f} -> {ce:.3f}"
    assert r1 >= 0.9, f"Recall@1 on the training batch = {r1:.2f}"


# 6. determinism --------------------------------------------------------------------------------
def _three_losses(name: str) -> list[float]:
    m = _model(name, seed=7)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    batch = _batch()
    torch.manual_seed(11)
    np.random.seed(11)
    out = []
    for _ in range(3):
        if hasattr(m, "on_epoch_start"):
            m.on_epoch_start(0)
        m.train()
        loss = m.compute_loss(batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        out.append(loss.item())
    return out


@pytest.mark.parametrize("name", MODELS)
def test_determinism(name):
    assert _three_losses(name) == _three_losses(name)


# 7. frozen features ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_frozen_features(name):
    m = _model(name)
    keys = set(m.state_dict())
    for nm in FEATURE_BUFFERS:
        assert nm not in keys
        if hasattr(m, nm):
            assert not getattr(m, nm).requires_grad
    feats = {nm: getattr(m, nm).clone() for nm in FEATURE_BUFFERS if hasattr(m, nm)}
    sizes = {D_T, D_V, N_CAT}
    for k, v in m.state_dict().items():
        assert not (v.dim() == 2 and v.shape[0] == N + 1 and v.shape[1] in sizes and v.dtype != torch.long), \
            f"{k} looks like a frozen feature table inside the state_dict"
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    if hasattr(m, "on_epoch_start"):
        m.on_epoch_start(0)
    loss = m.compute_loss(_batch())["loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    for nm, v in feats.items():
        assert torch.equal(getattr(m, nm), v), f"{nm} changed after an optimiser step"


# 8. exclude_history --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_exclude_history(name):
    m = _model(name).eval()
    batch = _batch()
    with torch.no_grad():
        s = m.score_all(batch)[:, 1:]
    hist = torch.zeros(B, N, dtype=torch.bool)
    for r, row in enumerate(batch["seq"].tolist()):
        for i in row:
            if i:
                hist[r, i - 1] = True
    out = mask_history(s, hist)
    assert torch.isneginf(out[hist]).all() and torch.isfinite(out[~hist]).all()


# 10. checkpoint round trip -----------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_checkpoint_roundtrip(name):
    m = _model(name, seed=1)
    if hasattr(m, "on_epoch_start"):
        m.on_epoch_start(0)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = _model(name, seed=2)
    m2.load_state_dict(torch.load(buf))
    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(m.eval().score_all(batch), m2.eval().score_all(batch))


def test_registry_complete():
    assert set(MODELS) == {"sasrec", "gru4rec", "bert4rec", "bsarec", "unisrec", "missrec", "tedrec", "hm4sr", "ddmsr"}
    for name in MODELS:
        assert issubclass(baseline_class(name), BaseSeqRec)
