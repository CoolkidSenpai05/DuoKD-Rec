"""Deployable GRU student (CLAUDE.md §3.6): cached E_f + GRU only, no T1/T2."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.gru_rec import GRUEncoder

TEACHER_PREFIXES = ("towers.t1.", "towers.t2.", "ens_logits")


@torch.no_grad()
def export_student(model: nn.Module, path: str | Path, cfg: dict) -> dict:
    """Save a student-only checkpoint from a trained DuoKD model.

    Contents: cached item table E_f [N+1, D] (computed once in eval mode), GRU weights, and the
    item-encoder weights it was computed from (Emb_ID, Proj, MoE_t, MoE_v, Fusion) so E_f can be
    rebuilt from features. T1/T2 weights are never written.
    """
    was_training = model.training
    model.eval()
    tables = model.item_encoder(detach_shared=True)
    E_f = tables["E_f"].detach().cpu().clone()
    model.train(was_training)
    state = {
        "E_f": E_f,
        "gru": {k: v.cpu() for k, v in model.towers["gru"].state_dict().items()},
        "item_encoder": {k: v.cpu() for k, v in model.item_encoder.state_dict().items()},
        "config": {"hidden": cfg["model"]["hidden"], "gru_layers": cfg["model"]["gru"]["n_layers"],
                   "dropout": cfg["model"]["dropout"], "max_len": cfg["data"]["max_len"],
                   "score_against": cfg["model"].get("score_against", "own")},
        "n_items": E_f.shape[0] - 1,
    }
    if state["config"]["score_against"] == "id":
        state["E_score"] = tables["E_id"].detach().cpu().clone()
    assert not any(k.startswith(TEACHER_PREFIXES) for k in state["gru"])
    torch.save(state, path)
    return state


class StudentRecommender(nn.Module):
    """Inference-only student.

    score(seq [B, L]) -> [B, N]     full-sequence path (same as the training GRU head in eval)
    init_state(B) -> None           empty history
    step(item [B], state) -> (scores [B, N], state [n_layers, B, D])   O(1) per new interaction
    topk(scores, k, exclude) -> (values, item ids 1..N)
    """

    def __init__(self, E_f: torch.Tensor, hidden: int, gru_layers: int,
                 E_score: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("E_f", E_f)
        self.register_buffer("E_score", E_f if E_score is None else E_score)
        self.gru = GRUEncoder(hidden, gru_layers, dropout=0.0)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "StudentRecommender":
        st = torch.load(path, map_location=map_location, weights_only=False)
        c = st["config"]
        m = cls(st["E_f"], c["hidden"], c["gru_layers"], st.get("E_score"))
        m.gru.load_state_dict(st["gru"])
        return m.eval()

    @property
    def n_items(self) -> int:
        return self.E_f.shape[0] - 1

    def _scores(self, u: torch.Tensor) -> torch.Tensor:
        return u @ self.E_score[1:].T

    @torch.no_grad()
    def score(self, seq: torch.Tensor) -> torch.Tensor:
        return self._scores(self.gru(self.E_f[seq], seq != 0))

    @torch.no_grad()
    def predict(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluator interface."""
        return {"student": self.score(seq)}

    def init_state(self, batch_size: int) -> None:
        return None

    @torch.no_grad()
    def step(self, item: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """s_t = GRU(E_f[i_new], s_{t-1}); scores = s_t @ E_f[1:].T."""
        u, state = self.gru.step(self.E_f[item], state)
        return self._scores(u), state

    @staticmethod
    def topk(scores: torch.Tensor, k: int, exclude: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """exclude: bool [B, N] items to drop (e.g. history). Returns item ids in 1..N."""
        if exclude is not None:
            scores = scores.masked_fill(exclude, float("-inf"))
        v, i = scores.topk(k, dim=-1)
        return v, i + 1
