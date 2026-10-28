# Open questions / choices made (M11–M12)

Mỗi mục: vấn đề → phương án đã chọn → lý do. Mục đánh dấu **[cần quyết định]** chờ bạn chốt.

## M11

### Q1. Tham số hóa test cũ theo dataset
- Vấn đề: prompt yêu cầu tham số hóa `test_data_split`, `test_features`; CLAUDE.md §12 cấm sửa test.
- Chọn: giữ nguyên hai file cũ (46 test cũ không đổi), thêm `tests/test_datasets.py` tham số hóa
  theo `synthetic + beauty/sports/toys/clothing`, gọi lại `_check` của `test_data_split` cho mọi dataset.
- Lý do: đáp ứng yêu cầu mà không đụng test cũ.

### Q2. Timestamp trùng nhau (Amazon 2014 có độ phân giải ngày)
- Thực tế: 42–48% cặp tương tác liên tiếp của cùng user có cùng timestamp (xem `docs/dataset_stats.md`).
- Chọn: thứ tự giữ như M1 (stable sort theo thứ tự trong file), không đổi. Khoảng thời gian = 0 là
  hợp lệ và được xử lý riêng ở HM4SR (M12).
- Hệ quả: thứ tự trong cùng ngày (và việc item nào là valid/test khi trùng ngày) phụ thuộc thứ tự file;
  giống cách các bài SR khác làm với cùng dữ liệu.

### Q3. Đường dẫn Beauty
- Chọn (đã được duyệt): Beauty giữ `data/processed/Beauty`, `data/processed/Beauty/feats`, `data/raw/*.json.gz`,
  `runs/` qua alias `configs/datasets/beauty.yaml`; dataset mới dùng `data/{raw,processed,feats}/<ds>/`, `runs/<ds>/`.
