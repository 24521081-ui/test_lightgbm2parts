# Biên bản kiểm chứng bàn giao

> Đây là biên bản tại lần bàn giao ban đầu. Sau đó đã kiểm toán kết quả Full CA_1 do bạn chạy, sửa kiểm tra đầu vào và bổ sung thành 43 unit test. Kết luận và bằng chứng hiện hành nằm ở [CODE_AUDIT_REPORT.md](../CODE_AUDIT_REPORT.md); số test và giới hạn Full dưới đây mô tả thời điểm cũ.

Ngày kiểm chứng: **06/11/2026**, múi giờ người dùng Asia/Saigon.

## Môi trường thực tế

- Windows, CPU; không dùng GPU hoặc CUDA.
- Python **3.14.8**, môi trường ảo `C:\Users\Admin\Desktop\PTDLKD\.venv`.
- NumPy 2.5.4, pandas 2.3.3, LightGBM 4.7.0, scikit-learn 1.9.1.
- Matplotlib 3.11.2, PyYAML 6.0.3, pytest 9.2.0.
- `pip check`: không có xung đột thư viện.
- Danh sách phiên bản đầy đủ: [environment-tested.txt](environment-tested.txt).

Máy hiện có Python 3.14, nên lần bàn giao này **chưa chạy trực tiếp bằng Python 3.11**. Mã dùng cú pháp tương thích 3.11; requirements dùng khoảng phiên bản để pip chọn thư viện phù hợp phiên bản Python. Danh sách môi trường thực tế không phải tệp khóa dành cho Python 3.11.

## Các kiểm tra đã chạy

```powershell
cd C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie
..\.venv\Scripts\python.exe -m pytest -q
```

Kết quả: **23 passed in 6.25s** sau khi cập nhật mã cuối.

Bao phủ:

- Lag 7, Rolling chỉ dùng quá khứ, không trộn sản phẩm/cửa hàng.
- Đặc trưng Train và dự báo cùng công thức.
- 28 bước đệ quy, thứ tự mã và cửa hàng, không sửa lịch sử gốc.
- Sửa lượng bán tương lai không đổi kết quả; không gọi fit trong vòng dự báo.
- Chặn NaN/vô hạn, cắt dự báo âm về 0, kiểm tra CSV đủ horizon.
- MAE/RMSE tính tay; mẫu số RMSSE, chuỗi hằng/toàn 0, căn chỉnh theo mã.
- Chọn mẫu Train không đổi khi sửa holdout; giá tương lai không đổi Train.
- Giá thiếu đầu chuỗi giữ NaN, không gán 0 hoặc điền ngược.
- Test cuối thiếu cờ xác nhận bị chặn; file dữ liệu thiếu báo rõ.
- Tích hợp LightGBM thật trên cả Validation không giá và Test có giá: lưu/nạp mô hình, hai CSV, chỉ số và bốn hình.

Các lệnh CLI đã chạy thành công trên **dữ liệu giả lập**:

```powershell
..\.venv\Scripts\python.exe run.py --mode smoke --store CA_1 --demo
..\.venv\Scripts\python.exe run.py --mode validation --all-stores --scope full --demo
..\.venv\Scripts\python.exe run.py --mode test --store CA_1 --scope full --confirm-final-test --demo
```

Smoke: 30 sản phẩm, 55.710 hàng fit, 840 dòng dự báo/mô hình. Validation Full demo và Test Full demo: 36 sản phẩm, 1.008 dòng dự báo/mô hình. Demo chỉ có một cửa hàng, nên lệnh `--all-stores --demo` duyệt cửa hàng đó; chưa phải kiểm chứng hiệu năng trên 10 cửa hàng thật. Đã mở xem PNG để kiểm tra nhãn tiếng Việt và nhãn giả lập hiển thị đúng.

Lần Smoke hoàn tất cuối có tên `validation_smoke_CA_1_20261106T144029351057Z` trong `results/demo/`. Lần Validation Full demo: `validation_full_CA_1_20261106T144046857677Z`. Lần Test Full demo: `test_full_CA_1_20261106T144046846197Z`.

Đã chạy lệnh Smoke không `--demo`: báo thiếu các file M5 trong `data/raw/` và dừng đúng thiết kế. Đã chạy Test thiếu `--confirm-final-test`: CLI từ chối trước huấn luyện.

## Giới hạn bằng chứng

Trong lần rà soát cuối đã phát hiện bộ CSV M5 ở `dataset/`, sao chép ba tệp cần thiết vào `data/raw/`, giữ nguyên bản gốc và chạy thêm Smoke thật:

```powershell
..\.venv\Scripts\python.exe run.py --mode smoke --store CA_1
```

Kết quả thành công: CA_1, 30 sản phẩm, 55.710 hàng fit, mỗi mô hình có 840 dòng dự báo đủ 28 ngày Validation. Bảng thật ở `results/metrics/validation_smoke_CA_1_20261106T144152266621Z/summary.csv`.

| Mô hình | MAE | RMSE | RMSSE trung bình |
|---|---:|---:|---:|
| LightGBM–Tweedie | 1.273140 | 2.573507 | 0.757725 |
| Seasonal Naive | 1.463095 | 3.474910 | 0.976218 |

Thời gian fit LightGBM khoảng 0,304 giây, chưa gồm đọc CSV, xây đặc trưng và đánh giá. Cả 30 chuỗi có RMSSE xác định. Đây chỉ là kết quả **Smoke trên mẫu nhỏ**, không đại diện toàn bộ M5 và không phải WRMSSE chính thức.

**Chưa huấn luyện/đánh giá Full M5 thật**, chưa đo RAM đỉnh hoặc thời gian trên 3049 sản phẩm × 10 cửa hàng. Không chạy Test cuối trên dữ liệu thật. Không tính WRMSSE chính thức. Không dùng kết quả giả lập để kết luận LightGBM tốt hơn mô hình khác trên Walmart M5.
