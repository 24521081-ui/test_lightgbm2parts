# Kiểm tra hồi quy sau refactor

Ngày kiểm tra: 23/11/2026. Không commit, push, thay thuật toán hoặc huấn luyện lại model nghiên cứu.

## Kết quả và phạm vi

| Kiểm tra | Trạng thái | Bằng chứng và giới hạn |
|---|---|---|
| Toàn vẹn output/model/metadata có trước refactor | PASS | 1.729 file, khoảng 6,23 GB, SHA-256 và kích thước không đổi; schema CSV và metadata JSON khớp baseline. Đây là kiểm tra file, không phải chạy lại suy luận |
| C7 λ = 0,75, CA_1, Validation | PASS | Chạy độc lập hàm gốc và hàm mới trên cùng model C5 đã lưu, hệ số Platt C6, dữ liệu và giá lịch sử. 3.049 chuỗi × 28 ngày = 85.372 dự báo. Khớp tuyệt đối từng giá trị |
| Các thành phần C7 trong lượt đối chiếu CA_1 | PASS | `p_raw`, `p_platt`, `p_C7`, `mu`, `yhat_C7` khớp tuyệt đối; không dùng C6 forecast làm đầu ra C7 |
| Chỉ số CA_1 trước/sau | PASS | RMSSE từng chuỗi, scale, RMSSE trung bình, MAE, RMSE, tổng dự báo và bias khớp; sai khác lớn nhất 0 |
| Dự báo CA_1 chạy lại so với CSV đã lưu | PASS | Ghép khóa độc lập, parser `float_precision='round_trip'`; sai khác lớn nhất 0 |
| Hai lambda 0,50 và 0,75 trên dữ liệu kiểm thử | PASS | So hàm gốc/mới, 28 bước, cả forecast và component; xáo trộn dòng rồi đối chiếu theo khóa; sai khác lớn nhất 0 |
| Kiểm thử hành vi và resume | PASS | 21 kiểm thử; bao gồm đệ quy, khóa, cutoff/holdout, fallback, tái dùng regressor, lambda cố định, xác suất sai, cache hợp lệ và cache sai lambda |
| Cú pháp, import, YAML, dependency | PASS | Compile toàn bộ Python; import 30 module/entrypoint; kiểm tra mốc 1885/1913 và horizon 28; `pip check` không phát hiện dependency hỏng |
| Git whitespace và file lớn đã tracked | PASS | `git diff --check` không báo lỗi; không có file tracked lớn hơn 5 MB. Git có thông báo chuyển LF/CRLF theo thiết lập hiện tại |
| Chạy lại 9 cửa hàng còn lại và toàn bộ Test | NOT VERIFIED | Chưa chạy độc lập sau refactor; chỉ xác minh hash của các kết quả đã có |
| Huấn luyện lại, so toàn bộ C5/C6 mới | NOT VERIFIED | Không chạy lại. Các module huấn luyện/đặc trưng/chấm điểm và code C6/Stage B được xác minh giữ nguyên byte; C5 chỉ đổi mức log |
| Xác suất C7 từng ngày trong output lịch sử | NOT VERIFIED | Chưa có dữ liệu thành phần C7 lưu riêng. Lượt đối chiếu CA_1 có kiểm tra component trong bộ nhớ, không giả tạo output lịch sử |
| Báo cáo tuần mới cho toàn bộ cửa hàng | NOT VERIFIED | Các CSV tuần hiện có giữ nguyên hash; không tuyên bố đã tính lại toàn bộ bảng tuần |

## Giá trị đối chiếu CA_1

| Chỉ số | Trước và sau |
|---|---:|
| RMSSE trung bình | 0.7155342115621935 |
| MAE | 1.1246409033174403 |
| RMSE | 2.195155614627311 |
| Tổng thực tế | 127955.0 |
| Tổng dự báo | 131138.66528764222 |
| Sai lệch tổng (%) | 2.4881132332790656 |
| Số chuỗi RMSSE hợp lệ | 3049 |
| Sai khác tuyệt đối lớn nhất trước/sau | 0 |

Khóa forecast gồm `origin_day`, `store_id`, `item_id`, `target_day`; giữ và kiểm tra các cột schema khác. Component được ghép bằng `item_id`, `target_day` trong cùng lượt cửa hàng/origin. Bảng chỉ số từng chuỗi được ghép bằng `item_id`. Không so CSV bằng vị trí dòng. Hai hàm dự báo gốc/mới thực sự tính độc lập; CSV gốc chỉ dùng làm mốc đối chiếu bổ sung.

## Môi trường kiểm thử

Trước refactor, `python -m pytest -q` cho 9 PASS, 4 FAIL vì LightGBM báo scikit-learn chưa được cài khi import LightGBM trước. scikit-learn 1.9.1 thực tế đã có; `pip check` PASS. Không sửa thư viện hoặc thêm workaround vào thuật toán nghiên cứu.

Khi nạp scikit-learn trước, bộ cũ có 13 PASS trước refactor; bộ mở rộng có 21 PASS sau refactor:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -c "import sklearn, pytest; raise SystemExit(pytest.main(['-q','--tb=short']))"
.\.venv\Scripts\python.exe -m compileall -q run.py src scripts tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/refactor_audit.py check
```

Lệnh pytest thông thường vẫn cần kiểm tra lại sau khi xử lý môi trường. Không xem 4 lỗi có trước là hồi quy do đợt sửa này, và không che chúng bằng skip/xfail.

## Bằng chứng và cách kiểm tra tiếp

- `baseline_before.json`: hash, kích thước, schema, metadata và các bảng tổng hợp đã chụp trước khi sửa code hiện có. Script từ chối ghi đè baseline.
- `artifact_integrity.json`: kiểm tra file trước/sau, không thay thế suy luận.
- `real_replay.json`, `synthetic_replay.json`: kết quả đối chiếu độc lập.
- `static_check.json`: các file giữ nguyên byte và kết quả import.
- `source_inventory.csv`, `source_changes.json`: danh mục và thay đổi của riêng đợt refactor, tách với thay đổi đã có trước đó.

Bản nguồn gốc để đối chiếu giữ cục bộ ở `.cache/refactor_before/`, được Git bỏ qua. Có thể chạy lại đối chiếu CA_1 tại máy hiện tại bằng `.\.venv\Scripts\python.exe .cache/replay_refactor.py`. Script này chỉ suy luận và chấm điểm trong bộ nhớ, ghi báo cáo audit; không ghi `results/` hoặc huấn luyện.

Để mở rộng xác minh, dùng cùng cách nạp model/calibrator, kiểm tra item order và origin, chạy độc lập hai phiên bản cho từng cửa hàng vào thư mục audit riêng; ghép theo khóa và so cả xác suất, lượng bán có điều kiện, forecast, RMSSE từng chuỗi và bảng tuần. Không dùng lệnh Train/Validation hàng loạt chỉ để kiểm tra hình thức code.
