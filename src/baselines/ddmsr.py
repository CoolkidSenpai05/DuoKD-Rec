"""DDMSR (Luo et al., ACM MM 2026, arXiv 2607.18786), re-implemented from github.com/jluo00/DDMSR `ddmsr.py`.

(a) GFD: per-modality item kNN graph on the frozen raw features (cosine, top-k, softmax weights on the
    k neighbours, "symmetric" normalisation with the row degree as in the official code), L_g = 2 layers
    h <- sigmoid(w_l) h + (1 - sigmoid(w_l)) A h, output LN(h + x)  (residual + LayerNorm).
(b) MoE adaptor with noisy gating (UniSRec style), experts = RecBole MLP [d_in, D, D] (ReLU).
(c) FSD: f_layer x [rFFT -> learnable complex filter -> irFFT, position-wise gate, LayerNorm] before the
    SASRec encoder (RecBole post-LN, causal).
(d) item = e_t + e_v (fusion_type sum), no ID embedding.
(e) loss = CE(cos / temperature) + cl_weight * InfoNCE(image, text) over the unique target items of the batch.
The kNN graph is built once by blocks, stored sparse and cached in <feat_dir>/cache/ (never dense N x N).
Right padding as in RecBole; FFT over the whole window -> neither causal nor padding invariant.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseSeqRec, ce_real, gather_last, left_to_right, mask_pad_column
from src.baselines.layers import MoEAdaptor, TransformerEncoder, extended_attention_mask, recbole_init


def build_knn_graph(emb: torch.Tensor, k: int, block: int = 2048, device: str | None = None) -> torch.Tensor:
    """Sparse version of the official GraphFeatureDenoiser.build_knn_graph. emb [M, d] -> sparse [M, M].

    Row i keeps topk(k+1) cosine neighbours minus the first one (assumed to be i itself), weighted by a
    softmax over their similarities; then A <- deg^-1/2[:, None] * A * deg^-1/2[None, :] with deg = row sums.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    en = F.normalize(emb.to(dev), dim=-1)
    rows, cols, vals = [], [], []
    for s in range(0, en.shape[0], block):
        sim = en[s:s + block] @ en.T
        tv, ti = sim.topk(k + 1, dim=-1)
        tv, ti = tv[:, 1:], ti[:, 1:]
        rows.append(torch.arange(s, s + sim.shape[0], device=dev).repeat_interleave(k))
        cols.append(ti.reshape(-1))
        vals.append(F.softmax(tv, dim=-1).reshape(-1))
    r, c, v = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    M = en.shape[0]
    deg = torch.zeros(M, device=dev).index_add_(0, r, v).clamp(min=1e-6)
    dis = deg.pow(-0.5)
    v = dis[r] * v * dis[c]
    return torch.sparse_coo_tensor(torch.stack([r, c]), v, (M, M)).coalesce().cpu()


def cached_knn_graph(emb: torch.Tensor, k: int, cache_dir: Path | None, tag: str) -> torch.Tensor:
    """build_knn_graph with an on-disk cache keyed by (tag, k, sha1 of the features)."""
    if cache_dir is None:
        return build_knn_graph(emb, k)
    h = hashlib.sha1(emb.numpy().tobytes()).hexdigest()[:16]
    f = Path(cache_dir) / f"ddmsr_knn_{tag}_k{k}_{h}.pt"
    if f.exists():
        s = torch.load(f)
        return torch.sparse_coo_tensor(s["indices"], s["values"], tuple(s["shape"])).coalesce()
    A = build_knn_graph(emb, k)
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"indices": A.indices(), "values": A.values(), "shape": list(A.shape)}, f)
    return A


class GraphFeatureDenoiser(nn.Module):
    def __init__(self, A: torch.Tensor, feat_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        self.self_weight = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(feat_dim)
        self.register_buffer("A", A, persistent=False)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        h = emb
        for w in self.self_weight:
            a = torch.sigmoid(w)
            h = a * h + (1 - a) * torch.sparse.mm(self.A, h)
        return self.norm(h + emb)


class FrequencyFilter(nn.Module):
    def __init__(self, hidden: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, max_len // 2 + 1, hidden, 2) * 0.02)
        self.noise_gate = nn.Sequential(nn.Linear(hidden * 2, 1), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        f = torch.fft.rfft(x, dim=1, norm="ortho")
        filt = self.dropout(torch.fft.irfft(f * torch.view_as_complex(self.complex_weight), n=L, dim=1, norm="ortho"))
        alpha = self.noise_gate(torch.cat([x, filt], dim=-1))
        return self.layer_norm(alpha * filt + (1 - alpha) * x)


class MLPExpert(nn.Module):
    """RecBole MLPLayers([d_in, D, D], dropout, 'relu', init_method='norm'): (Dropout, Linear, ReLU) x 2."""

    def __init__(self, d_in: int, d: int, dropout: float) -> None:
        super().__init__()
        self.mlp_layers = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(d, d), nn.ReLU())
        for mod in self.mlp_layers:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight.data, 0, 0.01)
                if mod.bias is not None:
                    mod.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x)


def info_nce(z_i: torch.Tensor, z_j: torch.Tensor, temp: float) -> torch.Tensor:
    """Official DDMSR.info_nce with sim='cos' (NT-Xent over 2B views, both directions)."""
    b = z_i.size(0)
    z = F.normalize(torch.cat((z_i, z_j), dim=0), dim=1)
    sim = z @ z.T / temp
    pos = torch.cat((torch.diag(sim, b), torch.diag(sim, -b)), dim=0).reshape(2 * b, 1)
    mask = torch.ones((2 * b, 2 * b), dtype=torch.bool, device=z.device).fill_diagonal_(False)
    ar = torch.arange(b, device=z.device)
    mask[ar, ar + b] = False
    mask[ar + b, ar] = False
    neg = sim[mask].reshape(2 * b, -1)
    logits = torch.cat((pos, neg), dim=1)
    return F.cross_entropy(logits, torch.zeros(2 * b, dtype=torch.long, device=z.device))


class DDMSR(BaseSeqRec):
    name = "ddmsr"
    modalities = ("text", "image")
    causal = False
    padding_invariant = False

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["ddmsr"]
        d = cfg["model"]["hidden"]
        L = cfg["data"]["max_len"]
        self.n_items = data["n_items"]
        self.temperature, self.cl_weight, self.cl_temp = m["temperature"], m["cl_weight"], m["cl_temperature"]
        self.graph_denoise = m["graph_denoise"]
        assert m["fusion_type"] == "sum", "only fusion_type=sum (paper default) is implemented"
        self.trm_encoder = TransformerEncoder(m["n_layers"], m["n_heads"], d, m["inner_size"],
                                              m["hidden_dropout_prob"], m["attn_dropout_prob"],
                                              m["hidden_act"], m["layer_norm_eps"])
        self.position_embedding = nn.Embedding(L, d)
        self.LayerNorm = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        self.apply(lambda mod: recbole_init(mod, m["initializer_range"]))
        self.register_buffer("plm_embedding", feats["text"], persistent=False)
        self.register_buffer("pvm_embedding", feats["image"], persistent=False)
        d_t, d_v = feats["text"].shape[1], feats["image"].shape[1]
        cache = Path(feats["feat_dir"]) / "cache" if "feat_dir" in feats else None
        if self.graph_denoise:
            self.text_graph_denoiser = GraphFeatureDenoiser(
                cached_knn_graph(feats["text"], m["knn"], cache, "text"), d_t, m["g_layer"])
            self.image_graph_denoiser = GraphFeatureDenoiser(
                cached_knn_graph(feats["image"], m["knn"], cache, "image"), d_v, m["g_layer"])
        filt = FrequencyFilter(d, L, m["filter_dropout"])
        self.filter_layers = nn.ModuleList([copy.deepcopy(filt) for _ in range(m["f_layer"])])
        self.text_trs = MoEAdaptor([MLPExpert(d_t, d, m["adaptor_dropout_prob"]) for _ in range(m["n_exps"])], d_t)
        self.image_trs = MoEAdaptor([MLPExpert(d_v, d, m["adaptor_dropout_prob"]) for _ in range(m["n_exps"])], d_v)

    def item_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(text_emb, image_emb) [N+1, D] for the whole catalog (GFD -> MoE adaptor)."""
        t, v = self.plm_embedding, self.pvm_embedding
        if self.graph_denoise:
            t, v = self.text_graph_denoiser(t), self.image_graph_denoiser(v)
        return self.text_trs(t), self.image_trs(v)

    def user_repr(self, seq: torch.Tensor, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        seq_r, lengths, _ = left_to_right(seq)
        pos = self.position_embedding(torch.arange(seq_r.size(1), device=seq.device)).unsqueeze(0)
        x = self.dropout(self.LayerNorm(text_emb[seq_r] + image_emb[seq_r] + pos))
        for f in self.filter_layers:
            x = f(x)
        h = self.trm_encoder(x, extended_attention_mask(seq_r))
        return gather_last(h, lengths)

    def _cos(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t, v = self.item_tables()
        s = F.normalize(self.user_repr(seq, t, v), dim=1)
        return s @ F.normalize(t + v, dim=1).T, t, v

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cos, t, v = self._cos(batch["seq"])
        ce = ce_real(cos / self.temperature, batch["target"])
        u = torch.unique(batch["target"])
        cl = info_nce(v[u], t[u], self.cl_temp)
        return {"loss": ce + self.cl_weight * cl, "L_ce": ce, "L_cl": cl}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return mask_pad_column(self._cos(batch["seq"])[0])
