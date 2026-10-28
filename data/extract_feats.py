"""Offline frozen BERT / ViT feature extraction (CLAUDE.md §5).

Outputs in `features.feat_dir` (row 0 = padding, all zeros):
  text_feat.npy [N+1, d_bert] float32, img_feat.npy [N+1, d_vit] float32,
  has_text.npy / has_img.npy [N+1] bool, feat_meta.json.
Re-runnable: images are cached on disk, models run in eval mode (deterministic).
`--limit K` processes only items 1..K and writes to `<feat_dir>_limit<K>`.
Failed image URLs (asin, url, reason) -> `<image_dir>/failed.txt` (`failed_limit<K>.txt` with --limit).
"""
from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_processed  # noqa: E402
from src.utils import get_logger, load_config, resolve, set_seed  # noqa: E402

log = get_logger("extract_feats")


def load_meta(meta_path: Path, item2id: dict[str, int]) -> dict[int, dict]:
    """item id -> raw metadata dict (only for kept items). Meta lines are python literals.

    The asin is read with a regex first so that only catalog items go through literal_eval
    (lines without a matching asin pattern are still fully parsed, so the result is unchanged).
    """
    import re

    asin_re = re.compile(r"""['"]asin['"]: ['"]([^'"]+)['"]""")
    out: dict[int, dict] = {}
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            m = asin_re.search(line)
            if m is not None and m.group(1) not in item2id:
                continue
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


class RateLimiter:
    """Thread-safe limiter: at most `rate` calls per second overall (rate <= 0: unlimited)."""

    def __init__(self, rate: float) -> None:
        import threading

        self.interval = 1.0 / rate if rate and rate > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        import time

        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)


def image_ok(path: Path) -> bool:
    """True if the file decodes: PIL verify() (structure) then a full re-open + load (pixels)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:  # noqa: BLE001 - any decode failure = corrupt
        return False


def download_image(url: str, path: Path, timeout: float = 30.0, retries: int = 3, backoff: float = 1.0,
                   limiter: RateLimiter | None = None) -> tuple[bool, str]:
    """Resumable download with retry + exponential backoff. Returns (ok, reason).

    A cached file is reused only if it decodes; a corrupt cached / downloaded file is deleted
    (the item is then treated as missing the image).
    """
    import time

    if path.exists() and path.stat().st_size > 0:
        if image_ok(path):
            return True, "cached"
        path.unlink()
    reason = ""
    for attempt in range(retries):
        if limiter is not None:
            limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            if image_ok(tmp):
                tmp.replace(path)
                return True, "downloaded"
            tmp.unlink()
            return False, "corrupt"  # server answered with a non-image / broken file: no retry
        except urllib.error.HTTPError as e:  # noqa: PERF203
            reason = f"http {e.code}"
            if e.code in (403, 404, 410):
                break  # permanent
        except Exception as e:  # noqa: BLE001
            reason = type(e).__name__
        time.sleep(backoff * 2**attempt)
    return False, reason or "failed"


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
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.dataset=toys")
    args = ap.parse_args()
    full = load_config(args.config, args.set)
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
    limiter = RateLimiter(fcfg.get("download_rate", 0))

    def job(i: int) -> tuple[Path | None, str, str]:
        url = meta.get(i, {}).get("imUrl")
        if not url:
            return None, "", "no url"
        p = img_dir / f"{id2asin[i]}.jpg"
        ok, reason = download_image(url, p, fcfg.get("download_timeout", 30), fcfg.get("download_retries", 3),
                                    fcfg.get("download_backoff", 1.0), limiter)
        return (p if ok else None), url, reason

    with ThreadPoolExecutor(fcfg["download_workers"]) as ex:
        res = list(ex.map(job, ids))
    paths = [r[0] for r in res]
    failed = [(id2asin[i], url, reason) for i, (p, url, reason) in zip(ids, res) if p is None]
    failed_log = resolve(fcfg.get("failed_log") or img_dir / (f"failed_limit{n}.txt" if args.limit else "failed.txt"))
    failed_log.write_text("".join(f"{a}\t{u}\t{r}\n" for a, u, r in failed))
    reasons = Counter(r for _, _, r in failed)
    log.info("images: %d/%d available; failures by reason %s (-> %s)",
             sum(p is not None for p in paths), n, dict(reasons), failed_log)
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
