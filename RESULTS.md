# RESULTS — DuoKD-Rec

## Milestone 1 — Dữ liệu (Amazon Beauty 2014, 5-core)

Nguồn: `reviews_Beauty_5.json.gz` (SNAP, McAuley 2014). Lọc 5-core lặp lại (file gốc đã là 5-core nên không đổi).

| Thống kê | Giá trị |
|---|---|
| Users | 22,363 |
| Items | 12,101 |
| Interactions | 198,502 |
| Density | 0.073% |
| Độ dài chuỗi TB / trung vị | 8.88 / 6 |
| Phân vị 90 / 95 / max | 14 / 21 / 204 |
| Mẫu train (cửa sổ trượt) | 131,413 |
| Target valid/test đã có trong lịch sử | 0 / 0 (không có mua lặp → `exclude_history` không loại nhầm target) |

Khớp với số liệu chuẩn của Beauty trong SASRec/BERT4Rec (22,363 / 12,101 / 198,502).

Test: `pytest tests/ -x -q` → `3 passed`.

Giả định: thứ tự các tương tác cùng timestamp giữ theo thứ tự trong file (stable sort); item ID gán theo thứ tự xuất hiện; không lọc theo modality ở bước này (xử lý thiếu modality ở M2).

## Milestone 2 — Đặc trưng đóng băng

| | Text | Ảnh |
|---|---|---|
| Mô hình | `bert-base-uncased`, `[CLS]`, ≤128 token | `google/vit-base-patch16-224-in21k`, `[CLS]` |
| Đầu vào | title + brand + categories + description (≤300 ký tự) | `imUrl` (cache tại `data/raw/images/Beauty`) |
| Shape | `[12102, 768]` | `[12102, 768]` |
| Item thiếu | 7 / 12,101 | 7 / 12,101 (không có URL) |

Item thiếu modality: `missing: zeros` (vector 0 + cờ `has_text.npy` / `has_img.npy`); có tùy chọn `mean`. Hàng 0 = zeros. `--limit 300` chạy thử OK (ghi vào `feats_limit300/`).
Modality dropout (§5) sẽ cài ở item encoder (M4) vì là bước train.

Test: `pytest tests/test_data_split.py tests/test_features.py` → `7 passed`.

## Milestone 3 — Baseline SASRec (chỉ ID)

Cấu hình `configs/sasrec.yaml`: D=64, 2 lớp, 2 head, dropout 0.2, pre-LN, L=50, CE full-softmax trên vị trí cuối (mẫu cửa sổ trượt), AdamW lr 1e-3 wd 1e-2, batch 256, seed 42.
Chọn checkpoint theo valid NDCG@20 (best epoch 35, early stop epoch 45, ~13 phút trên RTX 3090). Full ranking, `exclude_history=true`.

| Split | R@10 | N@10 | R@20 | N@20 | MRR |
|---|---|---|---|---|---|
| Valid | 0.0968 | 0.0595 | 0.1304 | 0.0680 | 0.0540 |
| **Test** | **0.0759** | **0.0464** | **0.1047** | **0.0537** | 0.0426 |

Nhóm cold-start (test): user có lịch sử ≤5 item (11,383 user): R@20 0.0887 / N@20 0.0452; item target có ≤10 tương tác train (10,458 user): R@20 0.0426 / N@20 0.0246.

So với số liệu công bố (Beauty, full ranking): SASRec loss BCE + 1 negative thường báo R@10 ≈ 0.06, N@10 ≈ 0.03 (các bảng CL4SRec/DuoRec, trích từ trí nhớ, chưa đối chiếu lại). Bản ở đây cao hơn, phù hợp với kết quả đã biết là CE full-softmax tốt hơn BCE đáng kể. Pipeline hợp lý: test < valid (không có dấu hiệu rò rỉ), cold-item thấp hơn hẳn.

Test: `pytest tests/ -x -q` → `13 passed` (data_split, features, metrics, causal).

## Milestone 4 — Item encoder (MoE, h_t, h_v, h_f)

- `src/layers/moe.py`: MoE với expert MLP 2 lớp (GELU), router Linear+softmax; soft (mặc định), top-k (`top_k`), hoặc MLP thường (`plain_mlp`, cho ablation). Trả về `probs` (cho `L_bal`) và `gates` (trọng số dùng thật).
- `src/layers/fusion.py`: `g = sigmoid(W[h_t;h_v])`, `h_f = Proj_f(g·h_t + (1-g)·h_v)`.
- `src/models/item_encoder.py`: đặc trưng BERT/ViT là buffer đóng băng (không vào `state_dict`, không có grad); `Proj = LayerNorm + Linear`; tính bảng `E_id, E_t, E_v, E_f` `[N+1, D]` cho toàn catalog một lần mỗi bước (hàng 0 = 0), `h_* = E_*[seq]`. Cờ `detach_shared` detach `h_t, h_v` trước Fusion. Modality dropout p=0.1 theo item mỗi bước (chỉ khi train), thay bằng vector "thiếu" cùng quy ước `features.missing`.
- `src/losses.py`: `moe_balance_loss` (soft: `n·ΣP²`; top-k: `n·Σf·P`), `gate_stats` (tải + entropy, có mask).

Sanity trên đặc trưng thật: không NaN; gate khởi tạo gần đều (tải ≈ [0.21, 0.33, 0.23, 0.22], entropy 1.37 / ln4 = 1.39); 1.09M tham số; ~50 ms fwd+bwd toàn catalog trên 3090.

Test: `pytest tests/ -x -q` → `21 passed` (thêm `test_moe`, `test_shapes`, `test_frozen_encoders`). Phần `test_mask` / `test_shapes` cho T1, T2, GRU sẽ bổ sung ở M5/M6 khi các mạng này tồn tại.
