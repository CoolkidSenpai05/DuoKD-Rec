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

## Milestone 5 — T1 và T2 train riêng (chưa distill)

Loss: `L_sup + 0.01·L_bal`. T1: Transformer nhân quả trên `h_t`, chấm với `E_t`; T2: trên `h_v`, chấm với `E_v`. Seed 42, chọn checkpoint theo valid NDCG@20 (head ghi trong cột "chọn theo"). Chạy trong tmux, `exclude_history=true`.

| Run | Head | Chọn theo | Best ep | Valid N@20 | Test R@10 | Test N@10 | Test R@20 | Test N@20 | cold-user N@20 | cold-item N@20 |
|---|---|---|---|---|---|---|---|---|---|---|
| SASRec (M3) | ID | sasrec | 35 | 0.0680 | 0.0759 | 0.0464 | 0.1047 | 0.0537 | 0.0452 | 0.0246 |
| `m5_t1` | T1 | t1 | 29 | 0.0794 | 0.0893 | 0.0527 | 0.1251 | 0.0617 | 0.0539 | 0.0247 |
| `m5_t2` | T2 | t2 | 36 | 0.0782 | 0.0897 | 0.0522 | 0.1244 | 0.0609 | 0.0546 | 0.0244 |
| `m5_t1t2` | T1 | ens | 35 | 0.0785 | 0.0899 | 0.0531 | 0.1260 | 0.0622 | 0.0549 | 0.0272 |
| `m5_t1t2` | T2 | ens | 35 | 0.0780 | 0.0896 | 0.0527 | 0.1255 | 0.0618 | 0.0549 | 0.0265 |
| `m5_t1t2` | ens (0.5/0.5) | ens | 35 | 0.0791 | 0.0904 | 0.0532 | 0.1262 | 0.0623 | 0.0552 | 0.0272 |

Chẩn đoán (`m5_t1t2`, epoch 0 → 35 → 45):
- Bất đồng top-1 T1–T2: 0.86 → 0.65 → 0.64.
- Entropy gate (max ln4 = 1.386): MoE_t 1.35 → 1.30 → 1.33; MoE_v 1.28 → 1.28 → 1.32. Tải expert ở epoch 35: MoE_t [0.21, 0.22, 0.25, 0.32], MoE_v [0.23, 0.24, 0.24, 0.28] → không sụp expert. `L_bal` ≈ 2.03 (min = 2.0 cho hai MoE).

Nhận xét:
- Thêm đặc trưng đa phương thức giúp rõ: test N@20 +15% so với SASRec (0.0537 → 0.0617).
- **Nhánh ảnh không yếu**: T2 ≈ T1. Có thể vì cả hai đều chứa `e_id` trong đầu vào MoE nên phần ID chi phối. → Câu hỏi §13 "T2 nhận h_f nếu nhánh ảnh yếu" có lẽ không cần.
- Ensemble chỉ hơn teacher tốt nhất một chút (+0.0001 test N@20), dù hai mạng bất đồng top-1 ở 64% mẫu. Hai mạng sai khác nhau nhưng chưa bổ trợ nhiều cho nhau.
- Train chung (dùng chung `Emb_ID`) ngang hoặc hơn nhẹ train riêng, cold-item tốt hơn (0.0247 → 0.0272).
- Chỉ 1 seed: chênh lệch ±0.001 nằm trong nhiễu.

Test: `pytest tests/ -x -q` → `25 passed` (thêm `test_mask` cho T1/T2, shape của T1/T2).

## Milestone 6 — GRU student, chỉ nhãn thật

`src/models/gru_rec.py`: GRU 2 lớp, hidden D, dropout đầu vào. Padding trái được dồn về cuối rồi `pack_padded_sequence` → trạng thái ẩn bắt đầu từ 0 ở item thật đầu tiên, padding không chạm vào trạng thái (test). `u_g` = trạng thái lớp cuối, chấm với `E_f`. Có `step()` cho infer O(1) (M9).

Hai mốc (seed 42, chọn theo valid NDCG@20 của GRU):
- `m6_gru`: chỉ GRU, `gru_detach_shared=false` (bắt buộc: không có teacher thì detach sẽ để Emb_ID/MoE không được train).
- `m6_all_sup`: T1 + T2 + GRU, chỉ CE (λ_mut = λ_kd = 0), `gru_detach_shared=true`. Mốc đối chứng trực tiếp cho M8.

| Run | Head | Best ep | Valid N@20 | Test R@10 | Test N@10 | Test R@20 | Test N@20 | cold-user N@20 | cold-item N@20 |
|---|---|---|---|---|---|---|---|---|---|
| `m6_gru` | GRU | 21 | 0.0724 | 0.0840 | 0.0464 | 0.1204 | 0.0556 | 0.0484 | 0.0186 |
| `m6_all_sup` | T1 | 35 | 0.0776 | 0.0893 | 0.0520 | 0.1264 | 0.0613 | 0.0545 | 0.0271 |
| `m6_all_sup` | T2 | 35 | 0.0781 | 0.0903 | 0.0527 | 0.1251 | 0.0615 | 0.0542 | 0.0269 |
| `m6_all_sup` | GRU | 35 | 0.0756 | 0.0863 | 0.0500 | 0.1217 | 0.0589 | 0.0520 | 0.0230 |
| `m6_all_sup` | ens | 35 | 0.0784 | 0.0907 | 0.0530 | 0.1270 | 0.0621 | 0.0550 | 0.0272 |

Chẩn đoán `m6_all_sup` (epoch 35): bất đồng T1–T2 0.65; entropy gate 1.24 / 1.24; tải MoE_t [0.23, 0.23, 0.29, 0.25], MoE_v [0.27, 0.25, 0.24, 0.23].

Nhận xét:
- GRU student (test N@20 0.0589) > SASRec (0.0537) nhưng < teacher (~0.061) và ensemble (0.0621): khoảng cách ≈ 0.003 là phần KD (M8) cần lấp.
- GRU trên encoder do teacher huấn luyện (detach) **tốt hơn** GRU tự train encoder (0.0589 vs 0.0556, cold-item 0.0230 vs 0.0186): biểu diễn item từ T1/T2 có ích cho student ngay cả khi chưa KD.
- Teacher trong `m6_all_sup` ≈ M5 (detach đúng như thiết kế: GRU không ảnh hưởng teacher).
- Chạy lúc máy tải nặng (I/O đĩa chung), ~65–85 s/epoch; không ảnh hưởng kết quả.

Test: `pytest tests/ -x -q` → `27 passed` (thêm GRU vào `test_mask`, `test_shapes`).
