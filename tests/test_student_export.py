"""Exported student == training GRU head in eval; step-wise infer == full sequence (CLAUDE.md §3.6, §8)."""
from __future__ import annotations

import torch
from conftest import L, N_ITEMS

from src.models.duokd import DuoKD
from src.models.student import StudentRecommender, export_student


def _trained_model(tiny_cfg, feats):
    tiny_cfg["model"]["heads"] = ["t1", "t2", "gru"]
    tiny_cfg["model"]["dropout"] = 0.1
    tiny_cfg["item_encoder"]["modality_dropout"] = 0.3  # must be inactive after export
    torch.manual_seed(0)
    model = DuoKD(tiny_cfg, *feats)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    seq = torch.randint(1, N_ITEMS + 1, (8, L))
    model.train()
    for _ in range(3):  # move weights away from init
        out = model(seq)
        loss = sum(out[f"z_{h}"].logsumexp(-1).mean() for h in ("t1", "t2", "gru"))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _seqs():
    torch.manual_seed(1)
    s = torch.randint(1, N_ITEMS + 1, (5, L))
    s[0, :6] = 0
    s[1, :3] = 0
    s[2, : L - 1] = 0
    return s


def test_export_matches_train_path(tiny_cfg, feats, tmp_path):
    model = _trained_model(tiny_cfg, feats)
    path = tmp_path / "student.pt"
    export_student(model, path, tiny_cfg)
    assert model.training, "export must restore the training flag"
    st = torch.load(path, weights_only=False)
    keys = list(st["gru"]) + list(st["item_encoder"])
    assert not any(k.startswith(("towers.t1", "towers.t2", "t1", "t2", "ens_logits")) for k in keys)
    assert set(st) >= {"E_f", "gru", "item_encoder", "config"}

    student = StudentRecommender.from_checkpoint(path)
    model.eval()
    seq = _seqs()
    ref = model.predict(seq)["gru"]
    torch.testing.assert_close(student.score(seq), ref, rtol=0, atol=1e-6)
    torch.testing.assert_close(student.predict(seq)["student"], ref, rtol=0, atol=1e-6)


def test_stepwise_matches_full_sequence(tiny_cfg, feats, tmp_path):
    model = _trained_model(tiny_cfg, feats)
    path = tmp_path / "student.pt"
    export_student(model, path, tiny_cfg)
    student = StudentRecommender.from_checkpoint(path)
    seq = _seqs()
    full = student.score(seq)
    for b in range(seq.shape[0]):
        items = seq[b][seq[b] != 0]
        state = student.init_state(1)
        for t, i in enumerate(items.tolist()):
            scores, state = student.step(torch.tensor([i]), state)
            prefix = torch.zeros(1, L, dtype=torch.long)
            prefix[0, L - t - 1:] = items[: t + 1]
            torch.testing.assert_close(scores[0], student.score(prefix)[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(scores[0], full[b], rtol=1e-5, atol=1e-5)


def test_topk_excludes_history():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    excl = torch.tensor([[False, True, False, False]])
    _, ids = StudentRecommender.topk(scores, 2, excl)
    assert ids.tolist() == [[4, 3]]
