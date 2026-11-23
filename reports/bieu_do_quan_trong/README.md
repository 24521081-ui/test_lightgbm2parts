# Biểu đồ quan trọng cho báo cáo M5

Thư mục này tập trung sáu hình PNG 250 DPI, chữ DejaVu Sans hỗ trợ tiếng Việt. Hình gốc trong `results/` được giữ nguyên. Chỉ dùng kết quả Full đã hoàn tất; không dùng Smoke để rút ra kết luận chính thức.

| Tệp | Nội dung và phạm vi | Vị trí phù hợp trong báo cáo |
|---|---|---|
| `02_RMSSE_theo_nhom_luong_ban.png` | C5/C6 theo ba nhóm gián đoạn đã xác định từ lịch sử; Validation, 30.490 chuỗi | Phân tích đặc điểm lượng bán |
| `03_RMSSE_theo_10_cua_hang.png` | RMSSE C5/C6/C7 tại 10 cửa hàng; Validation | Phân tích theo cửa hàng |
| `04_so_sanh_C5_C6_C7.png` | RMSSE, MAE, sai lệch tổng ở ba trục riêng; hàng trên Validation C5/C6/C7, hàng dưới Test C6/C7 | Đánh giá tác động hiệu chuẩn |
| `05_hieu_chuan_xac_suat.png` | Reliability Diagram của raw và Platt trên cùng đường dự báo C6 ngoài mẫu; Validation | Đánh giá xác suất |
| `06_thuc_te_va_du_bao_28_ngay.png` | Tổng lượng bán từng ngày toàn hệ thống, thực tế và C6/C7; Validation | Phân tích diễn biến 28 ngày |
| `07_sai_lech_tong_luong_ban.png` | Dự báo thừa (+), thiếu (−), tham chiếu 0%; Validation và Test tách riêng | Phân tích sai lệch tổng |

C5 dùng xác suất gốc, C6 dùng Platt, C7 dùng λ = 0,75 và một đường đệ quy riêng. Validation là d_1886–d_1913 (origin 1885), Test là d_1914–d_1941 (origin 1913). Mọi hình dùng 10 cửa hàng, 30.490 chuỗi; RMSSE là trung bình không trọng số trên chuỗi hợp lệ, không phải WRMSSE M5.

## Nguồn dữ liệu

- `results/analysis/stage_C7_validation_d1885_lambda075/`: manifest, `01_by_origin.csv`, `02_by_store.csv`, `folds/<store>/completed.json` và `forecast_C7.csv`.
- `results/analysis/final_test_C6_C7_frozen/comparison_C6_C7/`: manifest và bảng `01_by_origin.csv`.
- `results/analysis/validation_C6_platt_10_stores_20261121T145343079375Z/stage_A/`: `09_by_sales_pattern.csv` và `06_reliability_overall.csv`.
- `results/validation/validation_C6_platt_10_stores_20261121T145343079375Z_<store>/actual_vs_predictions.csv`: thực tế, dự báo C6, khóa và thành phần xác suất ngoài mẫu.
- Manifest C5/C6 trong `results/metrics/` dùng để kiểm tra cấu hình, split, nguồn và phạm vi đã hoàn tất.

Danh sách đường dẫn cụ thể đã đọc nằm trong [nguon_du_lieu.json](nguon_du_lieu.json). Các bảng tổng hợp được dùng nguyên giá trị; đường dự báo ngày được ghép theo origin, cửa hàng, sản phẩm, ngày đích và horizon, không ghép theo vị trí dòng.

## Hình còn thiếu và giới hạn

`01_so_sanh_RMSSE_6_mo_hinh.png` chưa tạo: repository chưa có kết quả hoàn tất ES_bu, Croston–SBA, Random Forest và LightGBM–Tweedie trên cùng tập đánh giá. Không thay bằng số giả hoặc lấy kết quả từ tập khác.

Hình 02 hiện chỉ có C5/C6 theo nhóm lịch sử đã lưu. Hình 05 không có C7 vì chưa lưu xác suất từng ngày trên đường C7; không lấy xác suất C6 rồi pha trộn để giả lập đường C7. Hình 06 hiện chỉ có C6/C7. Không suy luận chất lượng hiệu chuẩn chỉ từ RMSSE.

## Tạo lại

Chạy PowerShell tại thư mục repository:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts/build_important_figures.py
```

Script chỉ đọc CSV/JSON trong `results/`, kiểm tra manifest, nguồn, khóa, phạm vi và giá trị trước khi vẽ; chỉ ghi hình và danh sách nguồn trong thư mục này. Cache font nằm trong `.cache/matplotlib/`, được Git bỏ qua. Không huấn luyện model, không chạy Validation và không ghi đè output nghiên cứu.
