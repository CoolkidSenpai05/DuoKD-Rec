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
