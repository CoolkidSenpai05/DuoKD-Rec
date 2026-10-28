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

## Milestone 7 — Mutual distillation T1 ⇄ T2

`L_mut = τ²[KL(p2.detach‖p1) + KL(p1.detach‖p2)]`, τ = 2. `λ_mut` warm-up tuyến tính (`src/schedules.py`): 0 tới epoch 5, tăng đều, đạt 0.3 ở epoch 10. `m7_mut` giống hệt `m6_all_sup`, chỉ khác `λ_mut` (epoch 0 trùng khớp đến từng chữ số → deterministic). Seed 42, chọn theo valid N@20 của GRU.

| Run | Head | Best ep | Valid N@20 | Test R@10 | Test N@10 | Test R@20 | Test N@20 | cold-user N@20 | cold-item N@20 |
|---|---|---|---|---|---|---|---|---|---|
| `m6_all_sup` (λ_mut=0) | T1 | 35 | 0.0776 | 0.0893 | 0.0520 | 0.1264 | 0.0613 | 0.0545 | 0.0271 |
| | T2 | 35 | 0.0781 | 0.0903 | 0.0527 | 0.1251 | 0.0615 | 0.0542 | 0.0269 |
| | GRU | 35 | 0.0756 | 0.0863 | 0.0500 | 0.1217 | 0.0589 | 0.0520 | 0.0230 |
| | ens | 35 | 0.0784 | 0.0907 | 0.0530 | 0.1270 | 0.0621 | 0.0550 | 0.0272 |
| `m7_mut` (λ_mut=0.3) | T1 | 35 | 0.0786 | 0.0897 | 0.0520 | 0.1287 | 0.0619 | 0.0551 | 0.0249 |
| | T2 | 35 | 0.0790 | 0.0901 | 0.0528 | 0.1274 | 0.0622 | 0.0551 | 0.0259 |
| | GRU | 35 | 0.0769 | 0.0876 | 0.0505 | 0.1243 | 0.0597 | 0.0527 | 0.0212 |
| | ens | 35 | 0.0793 | 0.0903 | 0.0527 | 0.1286 | 0.0623 | 0.0555 | 0.0259 |

Độ bất đồng top-1 T1–T2 theo epoch (0, 5, 10, 20, 30, 35, 45):
- `m6_all_sup`: 0.857, 0.741, 0.716, 0.695, 0.662, 0.649, 0.637
- `m7_mut`:     0.857, 0.741, 0.684, 0.665, 0.640, 0.622, 0.605

`L_mut` (chưa nhân λ): 0.84 (ep 0) → 0.54 (ep 10) → 0.77 (ep 35); tăng lại khi phân phối hai mạng sắc hơn. Entropy gate ≈ 1.23 cho cả hai MoE (không sụp).

Nhận xét (1 seed, chênh lệch ~0.001 chưa đủ kết luận):
- Mutual giúp từng teacher nhẹ: valid N@20 +0.0010 (T1), +0.0009 (T2); test N@20 +0.0006 / +0.0007.
- Bất đồng giảm đều ~0.03 → hai teacher giống nhau hơn; ensemble hầu như không hưởng lợi (test N@20 0.0621 → 0.0623). Lợi ích ensemble so với teacher tốt nhất vẫn rất nhỏ.
- GRU tăng nhẹ (test N@20 0.0589 → 0.0597) dù chưa có KD — chỉ nhờ encoder dùng chung tốt hơn.
- Cold-item giảm (ens 0.0272 → 0.0259): mutual có thể đẩy teacher về item phổ biến. Cần kiểm tra lại với nhiều seed ở M10.

Test: `pytest tests/ -x -q` → `31 passed` (thêm `test_kd_detach` phần mutual + `test_schedules`).
Config M5/M6 được bổ sung `lambda_mut: 0` để giữ đúng thiết lập cũ khi base bật mutual.

## Milestone 8 — Ensemble KD → GRU

`z_ens = w1·z1.detach() + w2·z2.detach()` (w = 0.5/0.5), `L_kd = τ²·KL(softmax(z_ens/τ) ‖ softmax(z_g/τ))`, τ = 2, `λ_kd` warm-up 0 → 1.0 (epoch 5 → 10). `gru_detach_shared=true`. `m8_kd` = `m7_mut` + KD (cùng seed 42, chọn theo valid N@20 của GRU). Đã cài thêm `kd_teacher: ens|t1|t2` và `learned_ens_weights` (w = softmax(a), a chỉ học qua CE(z_ens)) cho ablation M10.

| Run | Head | Best ep | Valid N@20 | Test R@10 | Test N@10 | Test R@20 | Test N@20 | cold-user N@20 | cold-item N@20 |
|---|---|---|---|---|---|---|---|---|---|
| `m6_all_sup` (CE) | GRU | 35 | 0.0756 | 0.0863 | 0.0500 | 0.1217 | 0.0589 | 0.0520 | 0.0230 |
| `m7_mut` (+mutual) | GRU | 35 | 0.0769 | 0.0876 | 0.0505 | 0.1243 | 0.0597 | 0.0527 | 0.0212 |
| **`m8_kd` (+mutual +KD)** | **GRU** | 35 | **0.0785** | **0.0903** | **0.0523** | **0.1278** | **0.0618** | 0.0543 | 0.0226 |
| `m8_kd` | T1 | 35 | 0.0786 | 0.0897 | 0.0520 | 0.1287 | 0.0619 | 0.0551 | 0.0249 |
| `m8_kd` | T2 | 35 | 0.0790 | 0.0901 | 0.0528 | 0.1274 | 0.0622 | 0.0551 | 0.0259 |
| `m8_kd` | ens | 35 | 0.0793 | 0.0903 | 0.0527 | 0.1286 | 0.0623 | 0.0555 | 0.0259 |

Nhận xét:
- **KD có hiệu quả**: GRU test N@20 0.0597 → 0.0618 (+3.5%), R@20 0.1243 → 0.1278. Khoảng cách GRU–ensemble thu hẹp từ 0.0026 xuống 0.0005; GRU ngang T1 và chỉ kém ensemble rất ít.
- So với các mốc: SASRec 0.0537 → GRU-student 0.0618 (+15%); GRU một mình (`m6_gru`) 0.0556 → 0.0618 (+11%).
- **Teacher không bị ảnh hưởng bởi KD**: số liệu T1/T2/ens của `m8_kd` trùng khớp tuyệt đối với `m7_mut` (cùng seed) → xác nhận trên run thật là `L_kd` không đưa gradient về T1/T2/encoder dùng chung.
- `L_kd` (chưa nhân λ): 0.135 (ep 0) → 0.127 (ep 10) → 0.252 (ep 35) — tăng khi teacher sắc hơn, student theo sau.
- Cold-item của GRU (0.0226) vẫn thấp hơn teacher (~0.025): student học được phần lớn nhưng không hết về item hiếm.
- 1 seed; cần ≥ 3 seed ở M10 để xác nhận.

Test: `pytest tests/ -x -q` → `37 passed` (thêm `test_kd_detach` đầy đủ: L_kd, detach=false, trọng số ensemble học được; `test_gru_detach_shared`). Bộ test mất 15 phút do ổ đĩa chung quá tải I/O lúc chạy (bình thường ~10–30 s).
Config M5–M7 được bổ sung `lambda_kd: 0` để giữ thiết lập cũ.

## Milestone 9 — Export student + infer từng bước

- `src/models/student.py`:
  - `export_student(model, path, cfg)`: tính `E_f` toàn catalog một lần ở chế độ eval (không modality dropout, không dropout), lưu `E_f` cache + trọng số GRU + trọng số item encoder dùng để dựng lại `E_f` (Emb_ID, Proj, MoE_t, MoE_v, Fusion). **Không có T1/T2 / trọng số ensemble** (có assert + test).
  - `StudentRecommender`: `score(seq)` (đường cả chuỗi), `step(item, state)` (`s_t = GRU(E_f[i_new], s_{t-1})`, O(1) mỗi tương tác, `z = s_t @ E_f[1:].T`), `topk(scores, k, exclude)`.
- `scripts/export_student.py --ckpt <path> --out <path> [--verify]`.

Kiểm tra trên checkpoint thật `m8_kd` (`checkpoints/m8_kd_student.pt`, 7.3 MB):

| Split | GRU head (mô hình đầy đủ) R@20 / N@20 | Student export R@20 / N@20 | max \|diff\| mọi metric |
|---|---|---|---|
| Valid | 0.1587 / 0.0785 | 0.1587 / 0.0785 | 3.7e-6 |
| Test | 0.1278 / 0.0618 | 0.1278 / 0.0618 | 1.9e-9 |

Chênh lệch ~1e-6 trên valid là do sai số số thực GPU làm đổi thứ hạng ở vài điểm hòa (≈ 0.1 user / 22k), không phải sai logic; test CPU khớp tới 1e-6.

Test: `test_student_export` (3 test): đầu ra student = GRU head ở eval (atol 1e-6, kể cả khi model có dropout/modality dropout lúc train); `step()` từng bước = chạy cả chuỗi ở mọi tiền tố; checkpoint không chứa khóa T1/T2; `topk` loại lịch sử. Toàn bộ: xem dòng cuối mục này.

Lưu ý: infer từng bước giữ trạng thái trên **toàn bộ** lịch sử, còn đường train cắt `L = 50` item gần nhất → hai đường chỉ trùng nhau khi lịch sử ≤ 50 (95% user có ≤ 21 tương tác). Với lịch sử dài hơn, step-wise không cắt.

Toàn bộ `pytest tests/ -x -q` → `40 passed in 47.25s`.

## Milestone 10 — Ablation (đang chạy dở)

Hạ tầng: `configs/ablation_*.yaml` (19 file, mỗi file đổi đúng một yếu tố so với mô hình đầy đủ `m8_kd`), `scripts/run_ablation.py` (nhiều seed, chạy song song, bỏ qua run đã xong, xuất `runs/ablation/summary.{md,csv}`), `scripts/benchmark.py` (latency/bộ nhớ). Cờ mới (mặc định giữ hành vi cũ): `model.t2_input: E_v|E_f`, `item_encoder.use_text` / `use_image`. Test mới: `test_overfit_batch`, `test_determinism`, `test_ablation_flags` → toàn bộ `pytest tests/ -q` → `46 passed in 29.86s`.

### Đã chạy xong (seed 42, 43, 44)

Test, mean ± std (ddof=1); checkpoint chọn theo valid NDCG@20 của GRU (SASRec: của chính nó); full ranking, exclude_history.

| config | head | best ep. | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR | cold_user N@20 | cold_item N@20 |
|---|---|---|---|---|---|---|---|---|---|
| sasrec | sasrec | 33 | 0.0768 ± 0.0022 | 0.1050 ± 0.0006 | 0.0468 ± 0.0010 | 0.0539 ± 0.0007 | 0.0429 ± 0.0006 | 0.0458 ± 0.0011 | 0.0245 ± 0.0012 |
| gru_sup | gru | 20 | 0.0848 ± 0.0010 | 0.1227 ± 0.0027 | 0.0471 ± 0.0009 | 0.0567 ± 0.0013 | 0.0423 ± 0.0011 | 0.0502 ± 0.0015 | 0.0203 ± 0.0027 |
| full | t1 | 39 | 0.0900 ± 0.0004 | 0.1270 ± 0.0015 | 0.0528 ± 0.0007 | 0.0621 ± 0.0003 | 0.0478 ± 0.0007 | 0.0549 ± 0.0003 | 0.0264 ± 0.0015 |
| full | t2 | 39 | 0.0903 ± 0.0002 | 0.1266 ± 0.0008 | 0.0530 ± 0.0003 | 0.0622 ± 0.0002 | 0.0480 ± 0.0003 | 0.0550 ± 0.0003 | 0.0265 ± 0.0006 |
| full | ens | 39 | 0.0906 ± 0.0002 | 0.1274 ± 0.0010 | 0.0532 ± 0.0004 | 0.0624 ± 0.0001 | 0.0481 ± 0.0004 | 0.0553 ± 0.0003 | 0.0266 ± 0.0008 |
| full | **gru** | 39 | 0.0903 ± 0.0003 | 0.1268 ± 0.0011 | 0.0523 ± 0.0004 | **0.0615 ± 0.0003** | 0.0472 ± 0.0003 | 0.0545 ± 0.0001 | 0.0231 ± 0.0006 |
| kd_t1 | t1 | 42 | 0.0909 ± 0.0015 | 0.1271 ± 0.0015 | 0.0536 ± 0.0009 | 0.0628 ± 0.0009 | 0.0486 ± 0.0007 | 0.0552 ± 0.0007 | 0.0275 ± 0.0008 |
| kd_t1 | t2 | 42 | 0.0910 ± 0.0010 | 0.1267 ± 0.0009 | 0.0535 ± 0.0006 | 0.0625 ± 0.0006 | 0.0484 ± 0.0005 | 0.0550 ± 0.0004 | 0.0274 ± 0.0009 |
| kd_t1 | ens | 42 | 0.0916 ± 0.0014 | 0.1274 ± 0.0010 | 0.0539 ± 0.0008 | 0.0629 ± 0.0007 | 0.0487 ± 0.0005 | 0.0553 ± 0.0004 | 0.0275 ± 0.0009 |
| kd_t1 | **gru** | 42 | 0.0898 ± 0.0005 | 0.1269 ± 0.0011 | 0.0521 ± 0.0002 | **0.0614 ± 0.0001** | 0.0470 ± 0.0003 | 0.0546 ± 0.0002 | 0.0241 ± 0.0004 |

Nhận xét:
- **KD có hiệu quả**: GRU 0.0567 (chỉ nhãn thật) → 0.0615 (+8.5%, ≫ std); student chỉ kém ensemble ~1%. So với SASRec: +14%.
- **Ensemble chưa hơn teacher đơn**: GRU ← T1 đơn = 0.0614 ≈ GRU ← ensemble 0.0615 (trong khoảng std). Chờ `kd_t2`.
- Teacher ở `kd_t1` khác `full` một chút (dù có detach) vì checkpoint được chọn theo GRU → epoch dừng khác (42 vs 39), không phải rò gradient.
- **Tái lập**: `ablation_full_s42` trùng từng bit với `m8_kd_s42` (epoch 35, mọi metric).
- Cold-item: GRU (0.023–0.024) vẫn thấp hơn teacher (~0.027).

### Latency / bộ nhớ (chạy thử, checkpoint `m8_kd_s42`)

Bảng item đã cache; đo phần mạng tuần tự + chấm điểm toàn catalog, median ms. GPU lúc đo **đang bị tiến trình khác dùng chung**, warm-up 10 vòng → số liệu sơ bộ, sẽ đo lại (warm-up 100, 200 vòng).

| Đường | tham số | GPU B=1 | GPU B=256 | CPU B=1 | CPU B=256 |
|---|---|---|---|---|---|
| T1 (cả chuỗi L=50) | 103,296 | 4.86* | 2.98 | 2.55 | 33.6 |
| T2 (cả chuỗi L=50) | 103,296 | 4.85* | 2.96 | 2.84 | 33.3 |
| GRU cả chuỗi | 49,920 | 1.12 | 2.76 | 1.98 | 16.0 |
| GRU một bước (O(1)) | 49,920 | 0.39 | 0.51 | 0.59 | 3.4 |

\* đo đầu tiên, GPU chưa tăng xung — không đáng tin. Trạng thái mỗi user: GRU 512 byte (2×64 float) vs Transformer cần giữ 50 item id. Bộ nhớ đỉnh GPU (B=256): T1 40.5 MiB, GRU 11.9 MiB. Item encoder dùng chung 1.09M tham số (chủ yếu Emb_ID và Proj từ 768-d); khi infer chỉ cần bảng `E_f` (3.1 MB).

### Chưa chạy
- Nhóm 1 còn lại: `kd_t2`, `mut0`, `detach_false` (runner bị dừng bằng Ctrl+C lúc 07:05 khi đang chạy `kd_t2`; các run dở dang không có `result.json`, chạy lại sẽ train lại từ đầu).
- Nhóm 2: `mut0.1`, `mut1.0`, `t2_hf`, `moe_topk`, `moe_mlp`, `learned_ens`, `len20/100/200`, `text_only`, `image_only`, `id_only`.
- Đo latency chính thức khi GPU rảnh.
