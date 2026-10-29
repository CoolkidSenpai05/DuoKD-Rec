"""Faithfulness of the re-implemented core modules vs the official code in third_party/ (PROMPT §2.5.9).

These tests IMPORT AND EXECUTE third-party code, so they are skipped unless DUOKD_FAITHFULNESS=1.
Review the cloned repos first (docs/baselines.md lists the commits), then:
    DUOKD_FAITHFULNESS=1 pytest tests/test_faithfulness.py -q
RecBole-based repos need `third_party/RecBole` (or the repo's bundled recbole) and RecBole's deps importable.
Each test copies the official module's weights into ours and compares outputs on random input (atol 1e-5).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("DUOKD_FAITHFULNESS") != "1",
                                reason="executes third_party code; set DUOKD_FAITHFULNESS=1 after reviewing it")

TP = Path(__file__).resolve().parents[1] / "third_party"
ATOL = 1e-5


def _import(module: str, *paths: str):
    """Import `module` with only the given third_party dirs prepended to sys.path (fresh recbole/model)."""
    for name in list(sys.modules):
        if name.split(".")[0] in {"recbole", "model", "unisrec", "tedrec", "ddmsr", "missrec", "model_utils", "HM4SR"}:
            del sys.modules[name]
    old = list(sys.path)
    sys.path[:0] = [str(TP / p) for p in paths]
    try:
        return importlib.import_module(module)
    except ImportError as e:
        pytest.skip(f"cannot import {module} from {paths}: {e}")
    finally:
        sys.path[:] = old


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, atol=ATOL, rtol=1e-5)


def test_bsarec_block():
    off = _import("model.bsarec", "BSARec/src")
    from src.baselines.bsarec import BSARecBlock
    from src.baselines.layers import extended_attention_mask

    torch.manual_seed(0)
    args = SimpleNamespace(hidden_size=16, num_attention_heads=2, attention_probs_dropout_prob=0.0,
                           hidden_dropout_prob=0.0, hidden_act="gelu", c=5, alpha=0.7)
    ob = off.BSARecBlock(args).eval()
    mine = BSARecBlock(16, 2, 5, 0.7, 0.0, 0.0, "gelu").eval()
    sd = {(k[len("layer."):] if k.startswith("layer.") else k): v for k, v in ob.state_dict().items()}
    mine.load_state_dict(sd)
    seq = torch.tensor([[0, 0, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
    x = torch.randn(2, 8, 16)
    mask = extended_attention_mask(seq)
    _close(mine(x, mask), ob(x, mask))


def test_unisrec_moe():
    off = _import("unisrec", "RecBole", "UniSRec")
    from src.baselines.layers import MoEAdaptor, PWLayer

    torch.manual_seed(0)
    o = off.MoEAdaptorLayer(4, [24, 16], 0.0).eval()
    with torch.no_grad():
        o.w_gate.normal_()
        for e in o.experts:
            e.bias.normal_()
    mine = MoEAdaptor([PWLayer(24, 16, 0.0) for _ in range(4)], 24).eval()
    mine.load_state_dict(o.state_dict())
    x = torch.randn(5, 7, 24)
    _close(mine(x), o(x))


def test_tedrec_fusion():
    off = _import("tedrec", "RecBole", "TedRec")
    from src.baselines.layers import MoEAdaptor, PWLayer
    from test_baselines import _model

    torch.manual_seed(0)
    mine = _model("tedrec").eval()
    x, y = torch.randn(3, 8, 16), torch.randn(3, 8, 16)
    _close(mine.contextual_convolution(x, y), off.TedRec.contextual_convolution(mine, x, y))
    o = off.MoEAdaptorLayer(4, [24, 16], 0.0, 8).eval()
    with torch.no_grad():
        o.w_gate.normal_()
        for e in o.experts:
            e.bias.normal_()
    m2 = MoEAdaptor([PWLayer(24, 16, 0.0, bias_shape=(1, 8, 24)) for _ in range(4)], 24).eval()
    m2.load_state_dict(o.state_dict())
    z = torch.randn(3, 8, 24)
    _close(m2(z), o(z))


def test_ddmsr_gfd_fsd():
    off = _import("ddmsr", "RecBole", "DDMSR")
    from src.baselines.ddmsr import FrequencyFilter, GraphFeatureDenoiser, MLPExpert, build_knn_graph
    from src.baselines.layers import MoEAdaptor

    torch.manual_seed(0)
    emb = torch.randn(40, 24)
    emb[0] = 0
    og = off.GraphFeatureDenoiser(emb, 24, k=5, n_layers=2)
    with torch.no_grad():
        for w in og.self_weight:
            w.normal_()
        og.norm.weight.normal_()
    mg = GraphFeatureDenoiser(build_knn_graph(emb, 5, device="cpu"), 24, 2)
    mg.load_state_dict(og.state_dict(), strict=False)  # official A is a plain attribute
    _close(mg.A.to_dense(), og.A)
    _close(mg(emb), og(emb))
    of = off.FrequencyFilter(16, 8, 0.0).eval()
    mf = FrequencyFilter(16, 8, 0.0).eval()
    mf.load_state_dict(of.state_dict())
    x = torch.randn(3, 8, 16)
    _close(mf(x), of(x))
    om = off.MoEAdaptorLayer(4, [24, 16, 16], 0.0).eval()
    with torch.no_grad():
        om.w_gate.normal_()
    mm = MoEAdaptor([MLPExpert(24, 16, 0.0) for _ in range(4)], 24).eval()
    mm.load_state_dict(om.state_dict())
    _close(mm(emb), om(emb))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="official HM4SR modules hard-code .to('cuda')")
def test_hm4sr_moe(monkeypatch):
    off = _import("recbole_model.HM4SR", "HM4SR")
    from src.baselines.hm4sr import AlignMoE, TemporalMoE

    torch.manual_seed(0)
    d = 16
    cfg = {"start_expert_num": 4, "hidden_size": d, "start_gate_selection": "softmax",
           "initializer_weight": [0.0, 1.0, 1.0], "dataset": "x", "interval_scale": 100,
           "temporal_expert_num": 4, "temporal_gate_selection": "softmax"}
    oa = off.Align_MoE(cfg).cuda().eval()
    ma = AlignMoE(d, 4, [0.0, 1.0, 1.0]).cuda().eval()
    with torch.no_grad():
        oa.weight.normal_()
    ma.load_state_dict(oa.state_dict())
    v = torch.randn(3, 8, 3 * d, device="cuda")
    for a, b in zip(ma(v), oa(v)):
        _close(a, b)
    monkeypatch.setattr(off.Temporal_MoE_C, "get_interval_num", lambda self: 28.0)
    ot = off.Temporal_MoE_C(cfg).cuda()
    mt = TemporalMoE(d, 4, 100, 28.0).cuda()
    sd = {k: v for k, v in ot.state_dict().items()}
    sd["expert"] = torch.stack([e.detach() for e in ot.expert])
    sd["inv_freq"] = mt.inv_freq
    mt.load_state_dict(sd)
    ts = (torch.cumsum(torch.randint(0, 5, (3, 8)) * 86400, 1) + 1_300_000_000).float().cuda()  # no padding
    lengths = torch.full((3,), 8, device="cuda")
    for a, b in zip(mt(v, mt.time_embedding_of(ts, lengths)), ot(v, ts)):
        _close(a, b)


def test_missrec_fusion():
    off = _import("missrec", "MM23-MISSRec")
    import torch.nn.functional as F

    from test_baselines import _model

    torch.manual_seed(0)
    mine = _model("missrec").eval()
    with torch.no_grad():
        mine.fusion_factor.fill_(0.7)
    fake = SimpleNamespace(item_mm_fusion="dynamic_shared", train_stage="transductive_ft", id_type="id",
                           fusion_factor=mine.fusion_factor, item_embedding=mine.item_embedding)
    s = F.normalize(torch.randn(4, 16), dim=1)
    with torch.no_grad():
        t, v = mine.text_adaptor(mine.plm_embedding), mine.img_adaptor(mine.img_embedding)
        ref = off.MISSRec._compute_dynamic_fused_logits(fake, s, t, v) / mine.temperature
        _close(mine.logits(s), ref)
