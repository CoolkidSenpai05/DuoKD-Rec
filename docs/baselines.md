# Baselines (M12)

All baselines implement `src/baselines/base.py::BaseSeqRec` (`compute_loss(batch)`, `score_all(batch) -> [B, N+1]`,
column 0 = -inf) and are trained / evaluated by the project trainer (`scripts/train.py --config configs/baselines/<model>.yaml`).
Registry: `src/baselines/registry.py`. Tests: `tests/test_baselines.py`.

## Common protocol (identical for every model, including DuoKD-Rec)

| | |
|---|---|
| Data | same processed 5-core data, leave-one-out, `L = 50`, left-padded windows from the project dataloader (models that pad on the right in their official code convert internally) |
| Training samples | sliding-window prefixes of the train sequence (= RecBole data augmentation) |
| Evaluation | project `Evaluator`: full ranking over the catalog, `exclude_history=true`, R@10/20, N@10/20, MRR, cold-user (history ≤ 5) / cold-item (≤ 10 train interactions) groups |
| Model selection | valid NDCG@20 of the model itself, early stopping patience 10 (as M3–M10) |
| Main loss | CE full softmax over the N real items. **Deviation**: RecBole also keeps the padding item 0 as a class; we drop that column (same as SASRec M3). BERT4Rec: CE on the masked positions |
| Frozen features | the M2 features for every multimodal model: `bert-base-uncased` [CLS] (768) + `google/vit-base-patch16-224-in21k` [CLS] (768). **Deviation**: MISSRec / DDMSR use CLIP (512) in the paper, DDMSR RoBERTa text, UniSRec/TedRec BERT [CLS] (same family), HM4SR its own extractors |
| Hidden size | D = 64 for every model. **Deviation**: UniSRec / MISSRec / TedRec / DDMSR repos use 300 |
| Pre-training | none: UniSRec and MISSRec are trained from scratch on the target dataset (`pretrain: false`), transductive variant (with item IDs) |
| Optimiser | AdamW with the repo's lr / weight decay (weight decay 0 = Adam), the repo's batch size, no gradient clipping (RecBole default) |
| Seeds | 42, 43, 44 (smoke: 42) |

Third-party code is cloned read-only into `third_party/` (git-ignored). It was **read** to re-implement each model; it is
never imported by the training code.

## Models

### GRU4Rec
- Source: Hidasi et al. 2016; reference implementation RecBole 1.2 `recbole/model/sequential_recommender/gru4rec.py`,
  repo `RUCAIBox/RecBole` @ `7b02be5ec80a88310f2d04a27a82adfcbb5dc211`.
- Modality: ID. Defaults (RecBole `GRU4Rec.yaml`): embedding 64, GRU hidden 128, 1 layer, no bias, dropout 0.3; lr 1e-3, batch 2048.
- Padding: RecBole right padding (state read at len−1), equivalent to the packing used by the DuoKD student.
- Tune: lr {1e-3, 5e-4}, hidden {64, 128}, layers {1, 2}, dropout {0.1, 0.3, 0.5}.
- Faithfulness test: not applicable beyond structure (RecBole module = nn.GRU + Linear; covered by shape/causal/padding tests).

### SASRec
- The M3 model (`src/models/transformer_rec.py`) wrapped as `src/baselines/sasrec.py::SASRecRec` (subclass: same
  parameters, state_dict keys, operations). Smoke run through the new interface is **bit-identical** to the M3 path
  (`docs/regression/log.md`). Pre-LN, CE without the padding column, as in M3.

### BERT4Rec
- Source: Sun et al. 2019; reference RecBole 1.2 `bert4rec.py` + `data/transform.py::MaskItemSequence` (same commit as above).
- Modality: ID. Defaults (RecBole `BERT4Rec.yaml`): 2 layers, 2 heads, inner 256, dropout 0.2, mask_ratio 0.2, ft_ratio 0.5.
- `[MASK]` = id N+1 (own embedding row; never ranked). Eval: history + `[MASK]`, cut to L.
- **Deviation**: the cloze sequence of a training sample is prefix + target (RecBole masks the prefix only, so the last
  train item of each user is never a label there). The target is a train interaction, never valid/test.
  RecBole truncates masked positions to `int(mask_ratio * L)` per sequence; we use all masked positions (rarely > 10 with these lengths).
- Tune: mask_ratio {0.2, 0.4, 0.6} (paper: 0.6 Beauty), ft_ratio {0, 0.5}, lr.

### BSARec
- Source: Shin et al., AAAI 2024; repo `yehjin-shin/BSARec` @ `c80bdc0f1a889b78fe9a27e2058389cc84caea84`,
  `src/model/bsarec.py`, `_modules.py`, `_abstract_model.py`.
- Modality: ID. Defaults: README Beauty command (lr 5e-4, alpha 0.7, c 5, 1 head), `utils.py` (2 layers, dropout 0.5, batch 256, wd 0).
  The repo publishes only Beauty / LastFM settings: the Beauty setting is used for every dataset (→ QUESTIONS).
- Left padding as the official code. The frequency layer filters the whole window, padding included → not causal, not padding invariant (by design).
- Tune: alpha {0.1..0.9}, c {1,3,5,7,9}, heads {1,2,4}, lr.

### UniSRec
- Source: Hou et al., KDD 2022; repo `RUCAIBox/UniSRec` @ `05aa5cba2809112c32808f70d16abc61c05c6538`, `unisrec.py`, `props/`.
- Modality: text + ID (transductive). Parametric whitening experts + noisy-gated MoE adaptor (8 experts), cosine / temperature 0.07.
- **Deviations**: no pre-training; D = 64 (repo 300); BERT [CLS] of our text template (repo: BERT [CLS] of title+categories+brand).
- Tune: lr, dropout {0.2, 0.5}, temperature {0.05, 0.07, 0.1}.

### MISSRec
- Source: Wang et al., ACM MM 2023; repo `gimpong/MM23-MISSRec` @ `c3167c5d054c17c85be88d11a8c6e29d2a7bb037`,
  `missrec.py`, `model_utils.py`, `trainer.py`, `cluster_utils.py`, `torchpq/clustering/MinibatchKMeans.py`.
- Modality: text + image + ID (transductive_ft, img_text, seq_mm_fusion add, item_mm_fusion dynamic_shared).
- Interest discovery (k-means, K = ceil(0.5 · 2N)) before every training epoch, interest-token encoder, decoder over items.
- Faithful quirks kept: decoder self-attention is **non-causal** (`is_casual=False` in the official code); the k-means stops
  after one pass because the official tolerance check compares the updated centroids with themselves (error = 0);
  empty clusters get a zero centroid (torchpq NaN → 0).
- **Deviations**: no pre-training; D = 64 (repo 300); ViT/BERT 768-d features instead of CLIP 512-d.
- Tune: lr, interest_ratio {0.1, 0.5}, gamma {0, 1e-4, 1e-3}.

### TedRec
- Source: Xu et al., CIKM 2024; repo `RUCAIBox/TedRec` @ `e6b3cb9e93b7cd70f96e0a6e772b1104bf01deb7`, `tedrec.py`, `props/`.
- Modality: text + ID. DTR experts (position-wise whitening bias), MoE adaptor, contextual convolution in the frequency domain;
  items scored with the ID table only (as the official code).
- FFT over the whole (right-padded) window → not causal / padding invariant (by design).
- **Deviation**: D = 64 (repo 300).
- Tune: lr, dropout {0.2, 0.5}, n_exps {4, 8}.

### HM4SR
- Source: Zhang et al., WWW 2025; repo `SStarCCat/HM4SR` @ `f1c0a15f0193846e6af71c2a02ae324fd647eeaf`,
  `recbole_model/HM4SR.py`, `config/`, `dataprocess/data_process.py`.
- Modality: text + image + ID + timestamps (+ categories for the CP loss: `data/extract_categories.py`, first category path).
- Loss = CE + IDCL + CP + PCL, as in the official `calculate_loss`.
- **Leakage check**: the model reads `timestamp_list` = the times of the INPUT items only (RecBole augmentation);
  the target time is never used. `use_target_time` exists and must stay `false` (raising otherwise).
- Faithful quirks kept: the temporal experts are a plain Python list in the official code, hence never trained (kept as a
  fixed buffer, saved in the checkpoint); timestamps used as float32 seconds (RecBole FLOAT); contrastive "same target"
  masking sets logits to 0 (exp(0) = 1 terms remain), as in the official code.
- **Deviations**: right-padding boundary gives log2(negative) = NaN in the official code → intervals clamped at 0 there
  (masked positions only); interval table size from train/valid inputs (official: whole file incl. test); intervals above
  the table are clamped.
- Tune: lr, beta {0.1, 0.3, 0.5}, phcl_weight {0.1, 1}, temperature {0.1, 0.2, 0.5}.

### DDMSR
- Source: Luo et al., ACM MM 2026 (arXiv 2607.18786); repo `jluo00/DDMSR` @ `79549b0ff479ea0c95ada46dfe2375ad2e46b5e0`, `ddmsr.py`, `props/`.
- Modality: text + image (no ID). GFD (kNN graph k = 5, 2 layers), MoE adaptor (MLP experts, noisy gating), FSD (2 filter layers),
  item = e_t + e_v, loss = CE(cos / 0.07) + 0.1 · InfoNCE(image, text; τ = 0.2) over the unique targets of the batch.
- Code vs paper: the official "symmetric" normalisation uses the row degree on both sides and rows already sum to 1
  (softmax), so A is the row-softmax kNN matrix; we follow the code.
- kNN graph built once per dataset by blocks (never dense N×N), stored sparse, cached in `<feat_dir>/cache/`.
- **Deviations**: D = 64 (repo 300); BERT/ViT 768-d features (repo: RoBERTa-style text 768, CLIP image 512).
- Tune: knn {5, 10, 20}, f_layer {1, 2, 3}, cl_weight {0.01, 0.1, 0.5}, lr.

## Faithfulness tests (§2.5.9)

Executing code from `third_party/` was **not permitted in this session** (sandbox policy on external code), so the
faithfulness comparison against the official modules could not be run here. `tests/test_faithfulness.py` contains the
comparisons and is skipped unless `DUOKD_FAITHFULNESS=1` is set; run it yourself after reviewing the cloned repos
(it needs `third_party/RecBole` importable for the RecBole-based repos). Status per module:

| Module | Test | Status |
|---|---|---|
| BSARec block (frequency layer + attention mix + FFN) | `test_bsarec_block` | written, not executed |
| UniSRec PW + MoE adaptor | `test_unisrec_moe` | written, not executed |
| TedRec contextual convolution + DTR MoE | `test_tedrec_fusion` | written, not executed |
| DDMSR GFD + FSD | `test_ddmsr_gfd_fsd` | written, not executed |
| HM4SR Align MoE + Temporal MoE | `test_hm4sr_moe` | written, not executed (official code hard-codes `.to('cuda')` → needs a GPU) |
| MISSRec dynamic fused logits | `test_missrec_fusion` | written, not executed |
