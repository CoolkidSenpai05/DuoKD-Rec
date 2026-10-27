# CLAUDE.md — DuoKD-Rec

> Đọc toàn bộ file này trước khi làm bất kỳ việc gì. Đây là đặc tả duy nhất, ưu tiên cao hơn mọi giả định của bạn.
> Nếu đặc tả thiếu hoặc mâu thuẫn: **hỏi lại hoặc ghi rõ giả định**, không tự đoán âm thầm.

---

## 1. Tổng quan

**DuoKD-Rec** là hệ thống gợi ý tuần tự đa phương thức (ID + text + ảnh).

- **Đầu vào:** chuỗi lịch sử `S = (i_1, ..., i_L)` của một người dùng (cắt lấy L item gần nhất, padding bên trái).
- **Đầu ra:** phân phối trên toàn bộ catalog cho item tiếp theo `i_{L+1}`.
- **Ý tưởng:** hai Transformer *teacher* mỗi bên nhìn một modality (T1: text, T2: ảnh) học lẫn nhau bằng mutual distillation; ensemble của chúng được chưng cất sang một *student* GRU nhẹ. **Chỉ GRU được dùng khi infer.** Cả ba mạng đều học thêm từ nhãn thật.

```
                     ┌───────────── e_txt = Proj_t(BERT(text))  [frozen, precomputed]
item i ──► e_id ─────┤
                     └───────────── e_img = Proj_v(ViT(image))  [frozen, precomputed]

h_t = MoE_t([e_id ; e_txt])        h_v = MoE_v([e_id ; e_img])
h_f = Fusion(h_t, h_v)

 seq(h_t) ──► T1 ──► z1 ─┐   ⇄ mutual KD ⇄   ┌─ z2 ◄── T2 ◄── seq(h_v)
                         └──► z_ens = w1·z1 + w2·z2 ──(KD, detach)──► z_g ◄── GRU ◄── seq(h_f)

Train: T1, T2, GRU (cả 3 có CE với nhãn thật)
Infer: chỉ GRU (+ item tables đã cache)
```

---

## 2. Các quyết định mặc định (có thể đổi qua config)

| Mục | Mặc định |
|---|---|
| Dataset | Amazon Beauty (có thể chuyển Sports/Toys), chỉ giữ item có cả text và ảnh hoặc xử lý thiếu modality (xem §5) |
| Max length `L` | 50 |
| Lọc dữ liệu | 5-core (user và item đều ≥ 5 tương tác) |
| Split | Leave-one-out: item cuối = test, kế cuối = valid, còn lại = train |
| BERT / ViT | **Đóng băng**, trích đặc trưng offline ra `.npy`, không bao giờ chạy online khi train/infer |
| Hidden size `D` | 64 |
| MoE | Soft MoE (dùng mọi expert theo trọng số gate), `n_experts=4`, mỗi expert là MLP 2 lớp |
| Fusion | Gate: `g = sigmoid(W[h_t ; h_v])`, `h_f = g*h_t + (1-g)*h_v` |
| T1 | Transformer encoder nhân quả (causal), 2 lớp, 2 head, đầu vào `h_t` |
| T2 | Giống T1 về cấu hình, đầu vào `h_v` |
| GRU | 2 lớp, hidden `D`, đầu vào `h_f` |
| Chấm điểm | Mỗi mạng chấm điểm với bảng item tương ứng: T1 với `E_t`, T2 với `E_v`, GRU với `E_f` (xem §3.3) |
| Loss CE | Full softmax trên toàn catalog |
| Ensemble | Trung bình logits (`w1=w2=0.5`), tùy chọn trọng số học được |
| Nhiệt độ KD `τ` | 2.0 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-2, batch 256 |
| Seed | 42 (mọi lần chạy phải đặt seed và ghi vào log) |

---

## 3. Đặc tả chi tiết

### 3.1 Ký hiệu và shape

| Tensor | Shape | Ghi chú |
|---|---|---|
| `seq` | `[B, L]` | ID item, `0` = padding (padding bên trái) |
| `mask` | `[B, L]` | `True` ở vị trí thật, `False` ở padding |
| `target` | `[B]` | item kế tiếp cần dự đoán |
| `e_id` | `[B, L, D]` | ID embedding học được, `padding_idx=0` |
| `e_txt`, `e_img` | `[B, L, D]` | sau `Proj` từ đặc trưng đóng băng |
| `h_t`, `h_v`, `h_f` | `[B, L, D]` | đầu ra item encoder |
| `u_1, u_2, u_g` | `[B, D]` | biểu diễn người dùng (trạng thái vị trí cuối) |
| `E_t, E_v, E_f` | `[N+1, D]` | bảng biểu diễn toàn bộ item (index 0 = padding, bị loại khỏi logits) |
| `z_1, z_2, z_g` | `[B, N]` | logits trên N item thật |

### 3.2 Item encoder (`src/models/item_encoder.py`)

```
e_id  = Emb_ID(i)
e_txt = Proj_t(text_feat[i])      # text_feat: buffer đóng băng, [N+1, d_bert]
e_img = Proj_v(img_feat[i])       # img_feat: buffer đóng băng, [N+1, d_vit]
h_t   = MoE_t(concat[e_id, e_txt])
h_v   = MoE_v(concat[e_id, e_img])
h_f   = Fusion(h_t, h_v)
```

- Item encoder hoạt động **ở mức item**, không phụ thuộc ngữ cảnh chuỗi. Nhờ vậy có thể tính `E_t, E_v, E_f` cho toàn catalog trong một lần forward mỗi bước train và cache khi infer.
- Router của MoE: tuyến tính + softmax trên đầu vào của MoE. Có tùy chọn `top_k` (mặc định `null` = soft MoE).

### 3.3 Ba mạng dự đoán

- **T1:** `u_1 = Transformer_1(h_t, mask)[:, -1]`, `z_1 = u_1 @ E_t[1:].T`
- **T2:** `u_2 = Transformer_2(h_v, mask)[:, -1]`, `z_2 = u_2 @ E_v[1:].T`
- **GRU:** `u_g = GRU(h_f, mask)[:, -1]`, `z_g = u_g @ E_f[1:].T`
- Transformer dùng attention nhân quả + key padding mask + positional embedding học được.
- Vì padding bên trái, vị trí `-1` luôn là item thật cuối cùng. Với GRU, dùng `pack_padded_sequence` hoặc đảm bảo padding không làm nhiễu trạng thái ẩn (có test, xem §8).
- Tùy chọn `score_against: own | id`: `id` nghĩa là cả ba mạng cùng chấm với `Emb_ID`, dùng để ablation.

### 3.4 Hàm mất mát (`src/losses.py`)

**a) Nhãn thật (cả ba mạng):**
```
L_sup = CE(z_1, y) + CE(z_2, y) + CE(z_g, y)
```

**b) Mutual distillation T1 ⇄ T2** (mỗi chiều detach phía "giáo viên"):
```
L_mut = τ² · [ KL( softmax(z_2.detach()/τ) ‖ softmax(z_1/τ) )
             + KL( softmax(z_1.detach()/τ) ‖ softmax(z_2/τ) ) ]
```

**c) Ensemble → GRU** (một chiều, **không** có gradient từ GRU về T1/T2):
```
z_ens = w_1 · z_1.detach() + w_2 · z_2.detach()
L_kd  = τ² · KL( softmax(z_ens/τ) ‖ softmax(z_g/τ) )
```
- Nếu `learned_ens_weights=true`: `w = softmax(a)` với `a` là tham số học được, cập nhật bằng CE của `z_ens` (không dùng KL).

**d) Cân bằng tải MoE** (cho mỗi MoE):
- Soft MoE: `L_bal = n_experts · Σ_k (P_k)²`, với `P_k` là xác suất gate trung bình theo batch (có mask). Đạt cực tiểu khi đều.
- Top-k MoE: `L_bal = n_experts · Σ_k f_k · P_k` (f_k: tỉ lệ mẫu được gán).
- Tùy chọn router z-loss, mặc định tắt.

**Tổng:**
```
L = L_sup + λ_mut(t)·L_mut + λ_kd(t)·L_kd + λ_bal·(L_bal_t + L_bal_v)
```
- `λ_mut` tối đa 0.3, `λ_kd` tối đa 1.0, `λ_bal = 0.01`.
- Warm-up tuyến tính từ 0: `λ_mut` bắt đầu sau `mut_start_epoch` (mặc định 5) và tăng trong 5 epoch; `λ_kd` tương tự với `kd_start_epoch` (mặc định 5).

### 3.5 Luồng gradient (QUAN TRỌNG)

| Cờ config | Mặc định | Ý nghĩa |
|---|---|---|
| `gru_detach_shared` | `true` | `h_t`, `h_v` được `.detach()` **trước khi** vào `Fusion` cho nhánh GRU. Khi đó MoE và ID embedding chỉ được huấn luyện bởi T1/T2; `Fusion` và `Proj` riêng của GRU chỉ học từ GRU. Khi `false`, gradient của GRU chảy vào tầng dùng chung |
| `detach_teacher_in_kd` | `true` (bắt buộc) | Mọi logits đóng vai trò giáo viên trong KL đều `.detach()` |
| `freeze_encoders` | `true` | BERT/ViT không có gradient |

> Khi `gru_detach_shared=true`, `E_f` vẫn phải là hàm khả vi của tham số riêng GRU (Fusion). Không được để nhánh GRU hoàn toàn không có tham số học.

### 3.6 Suy luận (infer)

- Chỉ dùng: `Emb_ID`, `Proj`, `MoE_t`, `MoE_v`, `Fusion`, `GRU`. **Bỏ** T1, T2.
- Tính trước `E_f` cho toàn catalog một lần, cache.
- Với mỗi người dùng lưu trạng thái ẩn GRU; khi có tương tác mới: `s_t = GRU(E_f[i_new], s_{t-1})`, độ phức tạp O(1) mỗi bước.
- Điểm: `z = s_t @ E_f[1:].T`, lấy top-K.
- Phải có hàm `export_student()` lưu riêng checkpoint student (không chứa T1/T2) và test đảm bảo đầu ra giống hệt khi chạy qua đường train ở chế độ eval.

---

## 4. Dữ liệu (`data/prepare.py`)

1. Đọc tương tác (user, item, timestamp), sắp xếp theo thời gian, **lọc 5-core**.
2. Ánh xạ ID: item từ `1..N` (0 = padding), user từ `0..U-1`.
3. Split leave-one-out theo từng user.
4. Mẫu huấn luyện: với mỗi user, sinh nhiều mẫu `(prefix → item kế tiếp)` bằng cửa sổ trượt trên chuỗi train (không bao gồm item valid/test).
5. Đầu vào valid: chuỗi train → dự đoán item valid. Đầu vào test: chuỗi train + item valid → dự đoán item test.
6. Lưu ra `data/processed/` kèm file thống kê (số user, item, độ dài trung bình, phân vị 90/95).

**Rò rỉ dữ liệu là lỗi nghiêm trọng nhất.** Item valid/test không được xuất hiện ở bất kỳ đâu trong đầu vào hoặc nhãn huấn luyện. Không dùng thống kê từ valid/test để chuẩn hóa.

---

## 5. Trích đặc trưng (`data/extract_feats.py`)

- Text: ghép `title + brand + categories (+ description cắt ngắn)`, đưa qua BERT, lấy `[CLS]` hoặc mean-pool (cấu hình được), lưu `text_feat.npy` shape `[N+1, d_bert]`, hàng 0 là zeros.
- Ảnh: ViT, lấy `[CLS]`, lưu `img_feat.npy` shape `[N+1, d_vit]`, hàng 0 là zeros.
- **Item thiếu modality:** dùng vector zeros kèm cờ `has_text`/`has_img` (lưu thành mảng bool), hoặc vector trung bình toàn bộ item. Chọn qua config, ghi rõ trong log.
- Tùy chọn **modality dropout** khi train (xác suất mặc định 0.1): thay `e_txt` hoặc `e_img` của một item bằng vector "thiếu".
- Script phải có thể chạy lại, deterministic, và có `--limit` để chạy thử trên vài trăm item.

---

## 6. Đánh giá (`src/metrics.py`, `scripts/eval.py`)

- **Full ranking** trên toàn catalog (không dùng negative sampling để báo cáo).
- Mặc định `exclude_history=true`: đặt điểm của item đã có trong lịch sử = `-inf` (ghi rõ trong báo cáo vì ảnh hưởng kết quả).
- Metric: `Recall@{10,20}`, `NDCG@{10,20}`; thêm `MRR` tùy chọn.
- Chọn checkpoint theo `NDCG@20` trên **valid**, chỉ báo cáo test một lần cho checkpoint đó.
- **Luôn báo cáo đồng thời:** T1, T2, ensemble, GRU. Đây là cách duy nhất để biết distillation có hiệu quả.
- Đánh giá thêm theo nhóm: người dùng/item ít tương tác (cold-start), tính riêng.
- Báo cáo trung bình và độ lệch chuẩn qua ≥ 3 seed khi chạy ablation chính thức.

### Log bắt buộc mỗi epoch
- Loss từng thành phần: `L_sup` (tách 3 mạng), `L_mut`, `L_kd`, `L_bal`.
- Giá trị `λ_mut`, `λ_kd` hiện tại.
- Tải mỗi expert (trung bình gate) và entropy gate, cho cả hai MoE.
- **Độ bất đồng T1–T2:** tỉ lệ mẫu mà top-1 khác nhau.
- Metric valid của T1, T2, ensemble, GRU.

---

## 7. Cấu trúc repo

```
duokd-rec/
├── CLAUDE.md
├── RESULTS.md                 # ghi kết quả từng milestone
├── configs/
│   ├── base.yaml
│   └── ablation_*.yaml
├── data/
│   ├── prepare.py
│   ├── extract_feats.py
│   └── processed/             # không commit
├── src/
│   ├── dataset.py
│   ├── layers/
│   │   ├── moe.py
│   │   └── fusion.py
│   ├── models/
│   │   ├── item_encoder.py
│   │   ├── transformer_rec.py
│   │   ├── gru_rec.py
│   │   └── duokd.py           # ghép T1, T2, GRU, item encoder
│   ├── losses.py
│   ├── metrics.py
│   ├── schedules.py           # warm-up λ
│   └── trainer.py
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── run_ablation.py
│   └── export_student.py
└── tests/
```

---

## 8. Kiểm thử (bắt buộc, trong `tests/`)

| Test | Kiểm tra gì |
|---|---|
| `test_data_split` | Item valid/test không xuất hiện trong đầu vào hoặc nhãn train; chuỗi test chứa item valid |
| `test_shapes` | Shape của mọi module đúng theo §3.1 |
| `test_mask` | Đổi giá trị tại vị trí padding thì output của T1, T2, GRU không đổi |
| `test_causal` | Đổi item ở vị trí sau không làm đổi output ở vị trí trước (Transformer) |
| `test_frozen_encoders` | Đặc trưng BERT/ViT không có `requires_grad` và không đổi sau một bước tối ưu |
| `test_kd_detach` | `L_kd` không tạo gradient lên tham số của T1, T2 (kể cả `L_mut` không tạo gradient lên phía giáo viên của từng chiều) |
| `test_gru_detach_shared` | Khi `gru_detach_shared=true`: gradient của `CE(z_g)` và `L_kd` bằng 0 tại `Emb_ID`, `MoE_t`, `MoE_v`; khác 0 tại `Fusion` và GRU |
| `test_moe` | Tổng gate = 1, tải expert tính đúng, `L_bal` đạt cực tiểu khi gate đều |
| `test_metrics` | Tính tay Recall/NDCG trên ví dụ nhỏ, so với `metrics.py` |
| `test_overfit_batch` | Overfit một batch nhỏ: loss của cả ba mạng xuống gần 0 |
| `test_student_export` | Student export cho đầu ra giống hệt đường train ở chế độ eval; infer từng bước với trạng thái ẩn khớp với chạy cả chuỗi |
| `test_determinism` | Cùng seed cho cùng kết quả sau vài bước |

---

## 9. Lộ trình milestone

Mỗi milestone: viết code + test, **chạy test cho pass**, ghi kết quả vào `RESULTS.md`, commit git, rồi mới sang bước tiếp. Không làm trước milestone sau.

1. **Dữ liệu:** `prepare.py`, `dataset.py`, `test_data_split`.
2. **Đặc trưng:** `extract_feats.py`, xử lý item thiếu modality.
3. **Baseline SASRec (chỉ ID)** + `metrics.py` + `test_metrics`. Mốc kiểm tra pipeline (so với số liệu đã công bố trên cùng dataset).
4. **Item encoder** (MoE, `h_t`, `h_v`, `h_f`) + test shape, mask, MoE.
5. **T1 và T2 train riêng**, chưa có distill. Ghi metric từng mạng.
6. **GRU train riêng** (chỉ nhãn thật, đầu vào `h_f`).
7. **Mutual distillation** T1 ⇄ T2 với warm-up. Theo dõi độ bất đồng.
8. **Ensemble KD → GRU** với các cờ detach. Test gradient.
9. **Export student + infer từng bước** + test.
10. **Ablation** (xem §10) và bảng kết quả.

---

## 10. Ablation cần có (`configs/ablation_*.yaml`)

| Thí nghiệm | Mục đích |
|---|---|
| SASRec (ID) | Mốc đối chứng |
| GRU chỉ nhãn thật (đầu vào `h_f`) | Mốc student |
| GRU ← T1 đơn / GRU ← T2 đơn / GRU ← ensemble | Ensemble có hơn teacher đơn |
| T1, T2 có/không mutual | Đóng góp của mutual |
| `λ_mut` ∈ {0, 0.1, 0.3, 1.0} | Độ nhạy, theo dõi mất đa dạng |
| T2 nhận `h_v` vs T2 nhận `h_f` | Phương án khi nhánh ảnh yếu |
| Soft MoE vs top-k vs MLP thường | MoE có thật sự cần |
| `gru_detach_shared` true/false | Ảnh hưởng gradient từ student |
| Ensemble trung bình vs trọng số học được | Cách gộp |
| `max_len` ∈ {20, 50, 100, 200} | Độ nhạy theo độ dài chuỗi |
| Bỏ modality (text-only / image-only / ID-only) | Đóng góp từng modality |
| Latency/bộ nhớ: GRU vs T1/T2 | Xác nhận lợi ích infer |

---

## 11. Quy ước code

- Python 3.10+, PyTorch, type hints đầy đủ, docstring ngắn nêu rõ shape tensor vào/ra.
- Mọi siêu tham số nằm trong YAML, không hard-code trong code.
- Đặt seed cho `random`, `numpy`, `torch` (CPU/GPU) ở một hàm duy nhất `set_seed`.
- Không dùng `print` cho log; dùng `logging` hoặc ghi CSV/TensorBoard.
- Lưu checkpoint tốt nhất theo valid, kèm config và seed.
- Giữ mỗi file tập trung một trách nhiệm; không gộp loss vào model.

### Lệnh
```bash
pytest tests/ -x -q
python data/prepare.py --config configs/base.yaml
python data/extract_feats.py --config configs/base.yaml
python scripts/train.py --config configs/base.yaml
python scripts/eval.py --config configs/base.yaml --ckpt <path>
python scripts/run_ablation.py --configs configs/ablation_*.yaml
python scripts/export_student.py --ckpt <path> --out <path>
```

---

## 12. Quy tắc làm việc với Claude

1. **Chỉ làm trong phạm vi milestone được giao.** Không sửa file ngoài phạm vi nếu chưa nói rõ lý do.
2. **Không được sửa test cho pass.** Chỉ được sửa code. Nếu nghi ngờ test sai, giải thích và đề xuất, chờ xác nhận.
3. **Không nới lỏng ràng buộc cứng** (detach, frozen, leakage, mask) để chạy cho nhanh.
4. **Chạy test thật** và dán kết quả; không tuyên bố "đã pass" khi chưa chạy.
5. Khi gặp bug khó: in shape tensor, giá trị min/max/NaN, và kiểm tra bằng overfit một batch trước khi đoán nguyên nhân.
6. **Cuối mỗi lượt**, tóm tắt: (a) đã làm gì, (b) những giả định tự đưa ra, (c) điểm chưa chắc chắn, (d) bước tiếp theo đề xuất.
7. Không thêm thư viện lớn hoặc tính năng ngoài đặc tả (ví dụ fine-tune BERT/ViT, thêm kiến trúc mới) khi chưa được yêu cầu.
8. Nếu một quyết định thiết kế chưa được nêu ở đây, dùng giá trị mặc định ở §2 và ghi vào danh sách giả định.

---

## 13. Câu hỏi còn mở (cập nhật khi chốt)

- [ ] Dataset chính thức (Beauty, Sports, hay bộ khác).
- [ ] Cách gộp ensemble cuối cùng (trung bình hay học được).
- [ ] `gru_detach_shared` mặc định cuối cùng sau ablation.
- [ ] Có cần phương án T2 nhận `h_f` nếu nhánh ảnh quá yếu.
- [ ] Có mở rộng sang fine-tune BERT/ViT hay giữ đóng băng.