"""Offline frozen BERT / ViT feature extraction (CLAUDE.md §5).

Outputs in `features.feat_dir` (row 0 = padding, all zeros):
  text_feat.npy [N+1, d_bert] float32, img_feat.npy [N+1, d_vit] float32,
  has_text.npy / has_img.npy [N+1] bool, feat_meta.json.
Re-runnable: images are cached on disk, models run in eval mode (deterministic).
`--limit K` processes only items 1..K and writes to `<feat_dir>_limit<K>`.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_processed  # noqa: E402
from src.utils import get_logger, load_config, resolve, set_seed  # noqa: E402

log = get_logger("extract_feats")


def load_meta(meta_path: Path, item2id: dict[str, int]) -> dict[int, dict]:
    """item id -> raw metadata dict (only for kept items). Meta lines are python literals."""
    out: dict[int, dict] = {}
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            r = ast.literal_eval(line)
            if r["asin"] in item2id:
                out[item2id[r["asin"]]] = r
    return out


def build_text(r: dict, desc_max_chars: int) -> tuple[str, bool]:
    """title + brand + categories + truncated description. has_text = title or description present."""
    title = (r.get("title") or "").strip()
    brand = (r.get("brand") or "").strip()
    cats: list[str] = []
    for path in r.get("categories") or []:
        for c in path:
            if c not in cats:
                cats.append(c)
    desc = (r.get("description") or "").strip()[:desc_max_chars]
    parts = [p for p in (title, brand, ", ".join(cats), desc) if p]
    return ". ".join(parts), bool(title or desc)


def download_image(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        path.write_bytes(data)
        return True
    except Exception:  # noqa: BLE001 - any failure = missing image
        return False


def fill_missing(feat: np.ndarray, has: np.ndarray, strategy: str) -> None:
    """In-place: rows of real items (1..N) without the modality get zeros or the mean of present items."""
    miss = ~has
    miss[0] = False
    if strategy == "zeros":
        feat[miss] = 0.0
    elif strategy == "mean":
        present = has.copy()
        present[0] = False
        feat[miss] = feat[present].mean(0) if present.any() else 0.0
    else:
        raise ValueError(f"unknown missing strategy {strategy}")
    feat[0] = 0.0


@torch.no_grad()
def text_features(texts: list[str], cfg: dict, device: str) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["text_model"])
    model = AutoModel.from_pretrained(cfg["text_model"]).to(device).eval()
    outs = []
    bs = cfg["batch_size"]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], padding=True, truncation=True,
                  max_length=cfg["text_max_tokens"], return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state  # [b, T, d]
        if cfg["text_pooling"] == "cls":
            v = h[:, 0]
        elif cfg["text_pooling"] == "mean":
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            raise ValueError(cfg["text_pooling"])
        outs.append(v.float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, model.config.hidden_size), np.float32)


@torch.no_grad()
def image_features(paths: list[Path | None], cfg: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [n, d_vit], ok [n] bool). Unreadable images are marked missing."""
    from PIL import Image
    from transformers import AutoImageProcessor, ViTModel

    proc = AutoImageProcessor.from_pretrained(cfg["image_model"])
    model = ViTModel.from_pretrained(cfg["image_model"], add_pooling_layer=False).to(device).eval()
    d = model.config.hidden_size
    feats = np.zeros((len(paths), d), np.float32)
    ok = np.zeros(len(paths), bool)
    bs = cfg["batch_size"]
    for i in range(0, len(paths), bs):
        imgs, idx = [], []
        for j, p in enumerate(paths[i : i + bs]):
            if p is None:
                continue
            try:
                imgs.append(Image.open(io.BytesIO(p.read_bytes())).convert("RGB"))
                idx.append(i + j)
            except Exception:  # noqa: BLE001
                continue
        if not imgs:
            continue
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        feats[idx] = model(pixel_values=px).last_hidden_state[:, 0].float().cpu().numpy()
        ok[idx] = True
    return feats, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only process items 1..limit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    full = load_config(args.config)
    set_seed(full["seed"])
    dcfg, fcfg = full["data"], full["features"]

    data = load_processed(dcfg["processed_dir"])
    n = data["n_items"] if args.limit is None else min(args.limit, data["n_items"])
    out_dir = resolve(fcfg["feat_dir"] + (f"_limit{n}" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = resolve(fcfg["image_dir"])
    img_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(resolve(dcfg["raw_dir"]) / dcfg["meta_file"], data["item2id"])
    id2asin = {v: k for k, v in data["item2id"].items()}
    ids = list(range(1, n + 1))

    # ---- text
    texts, has_text = [], np.zeros(n + 1, bool)
    for i in ids:
        t, h = build_text(meta.get(i, {}), fcfg["desc_max_chars"])
        texts.append(t)
        has_text[i] = h
    log.info("text: %d/%d items have title/description", has_text.sum(), n)
    tf = text_features(texts, fcfg, args.device)
    text_feat = np.zeros((n + 1, tf.shape[1]), np.float32)
    text_feat[1:] = tf

    # ---- images (download with cache)
    def job(i: int) -> Path | None:
        url = meta.get(i, {}).get("imUrl")
        if not url:
            return None
        p = img_dir / f"{id2asin[i]}.jpg"
        return p if download_image(url, p) else None

    with ThreadPoolExecutor(fcfg["download_workers"]) as ex:
        paths = list(ex.map(job, ids))
    log.info("images: %d/%d downloaded", sum(p is not None for p in paths), n)
    vf, ok = image_features(paths, fcfg, args.device)
    img_feat = np.zeros((n + 1, vf.shape[1]), np.float32)
    img_feat[1:] = vf
    has_img = np.zeros(n + 1, bool)
    has_img[1:] = ok
    log.info("images: %d/%d decoded", has_img.sum(), n)

    fill_missing(text_feat, has_text, fcfg["missing"])
    fill_missing(img_feat, has_img, fcfg["missing"])

    np.save(out_dir / "text_feat.npy", text_feat)
    np.save(out_dir / "img_feat.npy", img_feat)
    np.save(out_dir / "has_text.npy", has_text)
    np.save(out_dir / "has_img.npy", has_img)
    info = {
        "n_items": n, "text_model": fcfg["text_model"], "text_pooling": fcfg["text_pooling"],
        "image_model": fcfg["image_model"], "missing": fcfg["missing"],
        "d_text": int(text_feat.shape[1]), "d_img": int(img_feat.shape[1]),
        "n_missing_text": int(n - has_text[1:].sum()), "n_missing_img": int(n - has_img[1:].sum()),
    }
    with open(out_dir / "feat_meta.json", "w") as f:
        json.dump(info, f, indent=2)
    log.info("missing-modality strategy=%s ; %s", fcfg["missing"], json.dumps(info))


if __name__ == "__main__":
    main()
