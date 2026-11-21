# Báo cáo thực nghiệm LightGBM Two-Part trên Walmart M5

Ngày kiểm chứng: **19/11/2026**, múi giờ Asia/Saigon. Repo: `D:\Phân tích dữ liệu kinh doanh\Code\IS403.R11-M5`. Nhánh đang dùng: **lightgbm-two-part**, tạo từ `tweedie` tại `2f05d89`. Khi bắt đầu, Git sạch; không cần worktree, không thay đổi nhánh nền, không commit hoặc push.

## 1. Thiết kế mô hình

Mỗi cửa hàng huấn luyện một cặp mô hình chung cho các sản phẩm được chọn:

- LGBMClassifier, objective binary: nhãn 1 khi lượng bán > 0, nhãn 0 khi không bán. Học trên mọi dòng có đủ đặc trưng, kể cả lượng bán bằng 0.
- LGBMRegressor, objective regression: học lượng bán gốc trên các dòng bán dương. Không dùng log nhãn. Bình phương sai số hướng mô hình tới kỳ vọng lượng bán có điều kiện.

Dự báo tại một ngày là `P(y>0|X) × E(y|y>0,X)`. Xác suất nằm trong [0,1]; dự báo lượng bán có điều kiện được chặn về không âm; từ chối NaN/vô cực. Không đặt ngưỡng phân lớp 0,5 để quyết định có bán hay không, không làm tròn lượng bán. Nếu Train chỉ có một lớp thì xác suất cố định bằng lớp đó. Nếu không có dòng dương thì cả dự báo lượng bán có điều kiện và dự báo cuối bằng 0. Metadata ghi chính sách fallback.

Lag 1/7/14/28, rolling mean 7/28 sau shift 1, lịch, SNAP và category sản phẩm giữ công thức từ Tweedie. Tạo đặc trưng trên đầy đủ lịch sử trước khi lọc dòng dương, tránh làm lag trở thành độ trễ giữa các lần bán. 28 ngày đầu chưa đủ cửa sổ được bỏ khỏi fit, vẫn giữ trong lịch sử. Toàn bộ ngày 0 còn lại được giữ cho phân loại.

Dự báo đệ quy từng ngày: tạo đặc trưng từ lịch sử thật đến origin và các dự báo trước đó; đưa tích hai thành phần vào lịch sử tạm. Hai mô hình Two-Part và Seasonal Naïve có lịch sử tạm riêng. Seasonal Naïve lặp chu kỳ 7 ngày, từ tuần thứ hai dùng dự báo của chính nó.

## 2. Chia tập và phạm vi đã chạy

| Giai đoạn | Lịch sử | Dự báo | Đã thực hiện trong lượt này |
|---|---|---|---|
| Smoke Validation CA_1 | d_1–d_1885 | d_1886–d_1913 | Có, 30 sản phẩm |
| Validation Full | d_1–d_1885 | d_1886–d_1913 | Chưa |
| Test cuối, học lại | d_1–d_1913 | d_1914–d_1941 | Chưa; chờ khóa cấu hình |

Nhãn holdout chỉ đi vào bước chấm điểm sau khi cả hai mô hình đã tạo đủ 28 bước. Lựa chọn hình minh họa được khóa từ Train trước khi đọc holdout. Test CLI vẫn yêu cầu `--confirm-final-test`.

## 3. Kiểm tra dữ liệu thật và môi trường

Nguồn: `D:\Phân tích dữ liệu kinh doanh\Dataset_mau\m5-forecasting-accuracy`.

- `sales_train_validation.csv`: 1947 cột, gồm sáu cột định danh và d_1–d_1941; ID hàng đầu là `HOBBIES_1_001_CA_1_evaluation`. Chép vào `data/raw/sales_train_evaluation.csv`.
- `sales_train.csv`: 1919 cột, chỉ d_1–d_1913; ID hàng đầu kết thúc `_validation`. Không dùng tệp này.
- `calendar.csv`: có đủ d, date, wm_yr_wk, event_name_1/2, snap_CA/TX/WI; loader kiểm tra ngày liên tục, khóa không trùng và SNAP 0/1.
- Đích ban đầu chỉ có `.gitkeep`, không ghi đè CSV cũ. Nguồn giữ nguyên; hai bản sao được Git bỏ qua. ID của các dòng CA_1 được loader kiểm tra hậu tố evaluation; tập cửa hàng đọc được giữ đủ 3049 mã trước khi lấy mẫu Smoke.

SHA256 nguồn và đích trùng nhau:

```text
sales:    4B4A47C44C38380D2A9168216FEA8C9FF2F31B1DDB772F8A0995952A038B8AA0
calendar: D12B5914EF03E66649ADF5DD9E996E6602251C22B7A6AF8F1F7E3AA12F8860F5
```

Môi trường thực tế: Python 3.14.7, NumPy 2.5.5, pandas 2.3.3, LightGBM 4.7.0, scikit-learn 1.9.1, Matplotlib 3.11.2, PyYAML 6.0.3, joblib 1.6.0, pytest 9.2.0. Cài tại `.venv/`. Chưa kiểm chứng riêng trên Python 3.11. Lần cài pip trong sandbox bị chặn socket; đã cài thành công sau khi được cấp quyền mạng.

## 4. Cấu hình thực tế và lần chạy thành công

Run: **validation_smoke_CA_1_20261119T060917732180Z** (bắt đầu khoảng 13:09 ngày 19/11/2026 giờ Việt Nam; tên run dùng UTC).

Cấu hình đầy đủ: [metadata.json](../results/models/validation_smoke_CA_1_20261119T060917732180Z/metadata.json), gồm cả YAML đã đọc và `get_params()` hiệu lực của hai thành phần.

| Tham số | Giá trị thực tế |
|---|---|
| classifier / regressor objective | binary / regression |
| n_estimators, learning_rate, num_leaves | 200, 0.05, 31 cho cả hai |
| random_state, n_jobs | 42, 6 cho cả hai |
| CPU, force_col_wise, deterministic | cpu, true, true |
| importance_type, verbosity | gain, -1 |
| smoke_items, seed, chunk_size | 30, 42, 500 |
| use_prices, price_policy | false, last_known |
| horizon, origin Validation / Test | 28, 1885 / 1913 |
| Dòng phân loại / dòng hồi quy | 55.710 / 17.418 |
| Dòng nhãn 0 phân loại / warmup bỏ khỏi fit | 38.292 / 840 |
| Khoảng ngày fit | d_29–d_1885 |
| Thời gian fit cả cặp | 0,518279 giây; không gồm đọc CSV, feature và vẽ hình |

Không cân bằng lớp tự động; chưa hiệu chỉnh xác suất hoặc tối ưu siêu tham số. Giá đang tắt. Chính sách giá từ Tweedie được giữ: Train chỉ ffill theo chuỗi, giá chưa biết vẫn NaN; dự báo giữ giá cuối biết tại origin. Chưa chạy nhánh bật giá trên M5 thật trong lượt này.

## 5. Điểm số đã đối chiếu độc lập

| Mô hình | MAE | RMSE | RMSSE_mean_unweighted | Chuỗi hợp lệ / không xác định |
|---|---:|---:|---:|---:|
| LightGBM Two-Part | 1,278218 | 2,548138 | 0,768830 | 30 / 0 |
| Seasonal Naïve | 1,463095 | 3,474910 | 0,976218 | 30 / 0 |

Mỗi mô hình có 30 × 28 = **840 dự báo**. MAE/RMSE lấy trên cùng 840 cặp sản phẩm–ngày. RMSSE mỗi chuỗi dùng scale `mean(diff(Train sau lần bán dương đầu tiên)^2)`. Chuỗi scale 0/không đủ điểm để trống RMSSE, không thay epsilon; tính trung bình không trọng số trên chuỗi xác định và báo cả hai số lượng. Đây không phải WRMSSE chính thức M5.

Tổng thực tế toàn mẫu trong kỳ: **1263 đơn vị**. Two-Part dự báo **1313,621307**, dư **50,621307**; Seasonal Naïve dự báo **1044**, thiếu **219**. Sai số tổng lượng bán và sai số từng ngày là hai góc nhìn khác nhau; dự báo tổng gần hơn vẫn có thể lệch thời điểm bán.

Two-Part có RMSSE thấp hơn baseline ở 25/30 mã: nhóm gián đoạn 14/16 (87,5%), nhóm rất gián đoạn 11/14 (78,57%). Mẫu này không có mã thuộc nhóm ít gián đoạn. Kết quả chỉ phản ánh 30 mã đã chạy, không đủ để kết luận cho toàn CA_1 hoặc toàn M5, cũng chưa so sánh trực tiếp với kết quả Tweedie trên cùng mẫu.

## 6. Năm biểu đồ và cách đọc

Mọi hình đều ghi **SMOKE**, CA_1, 30 sản phẩm và d_1886–d_1913. Tỷ lệ ngày bán 0 tính trên toàn Train d_1–d_1885: <25% là ít gián đoạn, [25%,75%) là gián đoạn, >=75% là rất gián đoạn. Đây là quy tắc mô tả của thí nghiệm, không phải phân loại ADI/CV² chuẩn và không loại tiền tố 0 của sản phẩm mới.

1. **01_actual_vs_forecast.png**: lấy mã đầu theo từ điển ở mỗi nhóm hiện có: `FOODS_1_182`, `FOODS_2_129`. Cột xám là thực tế; đường xanh có điểm là Two-Part; cam nét đứt là Naïve. Giữ trục đủ 28 ngày và ngày 0. Dùng để xem sai thời điểm bán và bỏ lỡ đột biến; ví dụ hình có đỉnh thực tế 8 đơn vị trong khi dự báo Two-Part thấp hơn nhiều. Đã mở ảnh để kiểm tra trực quan: chữ tiếng Việt đọc được, nhãn mã rõ, ba chuỗi phân biệt được, không cắt chú thích.
2. **02_cumulative_units.png**: dùng toàn bộ 30 mã, cộng lượng bán theo ngày rồi tích lũy. Ba đường cho biết thừa/thiếu tổng đơn vị và ghi chênh lệch tại ngày 28. Không dùng riêng các mã minh họa của hình 01 để tính tổng.
3. **03_mae_by_horizon.png**: hai đường MAE theo bước 1–28 trên cùng toàn bộ 30 mã. So sánh độ khó của dự báo xa hơn; đường dao động không tự chứng minh quan hệ nhân quả giữa horizon và sai số.
4. **04_model_gain_by_sales_pattern.png**: phân phối `RMSSE Two-Part − RMSSE Naïve`. Giá trị âm có lợi cho Two-Part. Vạch 0, số chuỗi hợp lệ, tỷ lệ thắng cho từng nhóm; nhóm nhỏ vẫn hiển thị từng điểm. Nhóm trống ghi n=0 và tỷ lệ không xác định. Bảng nền: `sales_pattern_gain.csv` cạnh hình.
5. **05_two_part_components.png**: chọn `HOBBIES_1_062` vì có tỷ lệ ngày 0 Train cao nhất trong mẫu (95,6499%); hòa chọn mã đầu từ điển. Ô 1 là xác suất có bán và nhãn có bán thực tế; ô 2 so lượng bán có điều kiện với thực tế ở ngày bán dương; ô 3 so tích Two-Part với lượng bán thật (thêm Naïve để tham chiếu). Đã mở kiểm tra ảnh. Không chọn lại mã sau khi biết sai số.

## 7. Kiểm tra đã thực hiện

- **8 kiểm thử tự động đạt**: giữ ngày 0 và đồng nhất lag/rolling Train–Predict; nhãn hồi quy chỉ dương; fallback toàn 0/toàn dương; lưu/nạp model; lưu văn bản qua thư mục tiếng Việt; đệ quy đưa tích vào ngày tiếp theo; Naïve chu kỳ 7; khóa trùng; actual trùng; scale RMSSE; chọn mẫu từ Train; thành phần không hợp lệ; chặn Test thiếu cờ. Một kiểm thử có thể kiểm tra nhiều khẳng định.
- Kiểm thử chống leakage dùng dữ liệu giả lập: điều phối thật đọc Train đến 1885, hoàn thành hai dự báo rồi mới đọc 1886–1913. Đổi toàn bộ nhãn holdout thành 999999 giữ nguyên mọi dự báo. Không mở Test M5 trong kiểm thử này.
- **M5 thật**: schema đúng tám cột, mỗi mã đủ 28 bước, khóa duy nhất, dự báo hữu hạn/không âm, chi tiết ghép actual 1:1 đúng toàn bộ 840 dòng, xác suất [0,1], dự báo khớp tích thành phần với rtol/atol 1e-12.
- Script độc lập đọc lại CSV gốc và CSV kết quả, tự tính MAE/RMSE/RMSSE và đối chiếu summary; xác nhận Naïve bằng lặp 4 lần 7 ngày cuối Train; nạp bundle để phát lại đủ 28 ngày, khớp CSV. Bằng chứng: [verification.json](../results/metrics/validation_smoke_CA_1_20261119T060917732180Z/verification.json).
- `compileall` và `git diff --check` đạt. Chưa chạy Full, 10 cửa hàng, đo RAM đỉnh hoặc kiểm chứng nhánh giá trên M5 thật.

Smoke đầu tiên `validation_smoke_CA_1_20261119T060827009156Z` dừng ở bước xuất model TXT do native LightGBM không ghi được đường dẫn Windows có dấu. Đã sửa bằng `model_to_string()` và Python `write_text(encoding='utf-8')`, thêm kiểm thử thư mục tiếng Việt, chạy lại thành công với run nêu trên. Thư mục dở dang được giữ riêng, không có summary/metadata hoàn chỉnh; không dùng làm kết quả thực nghiệm.

## 8. Lệnh PowerShell đã chạy

```powershell
Set-Location -LiteralPath 'D:\Phân tích dữ liệu kinh doanh\Code\IS403.R11-M5'
git status --short
git branch -av
git switch -c lightgbm-two-part
# Đọc header và hàng đầu bằng System.IO.File.OpenText trên các đường dẫn LiteralPath.
Copy-Item -LiteralPath 'D:\Phân tích dữ liệu kinh doanh\Dataset_mau\m5-forecasting-accuracy\sales_train_validation.csv' -Destination 'data/raw/sales_train_evaluation.csv'
Copy-Item -LiteralPath 'D:\Phân tích dữ liệu kinh doanh\Dataset_mau\m5-forecasting-accuracy\calendar.csv' -Destination 'data/raw/calendar.csv'
git check-ignore data/raw/sales_train_evaluation.csv data/raw/calendar.csv
& 'C:\Users\LENOVO\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q run.py src tests scripts/verify_two_part_results.py
.\.venv\Scripts\python.exe run.py --mode smoke --store CA_1
.\.venv\Scripts\python.exe scripts/verify_two_part_results.py validation_smoke_CA_1_20261119T060917732180Z
git diff --check
```

Đã dùng thêm `Get-FileHash -Algorithm SHA256` cho hai tệp nguồn và hai tệp đích. `python`/`py` mặc định trỏ alias không chạy được trong phiên này, vì vậy đã dùng executable thật để tạo venv. `PYTHONUTF8=1` giúp log/bảng tiếng Việt hiển thị đúng qua PowerShell.

## 9. File thay đổi và đường dẫn bàn giao

Sửa `.gitignore` (giữ cấu trúc results qua .gitkeep), `README.md`, `configs/config.yaml`, `run.py`, `src/data_loader.py`, `src/feature_engineering.py`, `src/training.py`, `src/forecasting.py`, `src/reporting.py`. Giữ nguyên `src/preprocessing.py`. Thêm `tests/test_two_part.py`, `scripts/verify_two_part_results.py`, báo cáo này và `.gitkeep` ở năm thư mục kết quả. Các tài liệu/tiện ích Tweedie cũ giữ làm lịch sử; README phân biệt rõ.

Các đường dẫn dưới đây tính từ repo `D:\Phân tích dữ liệu kinh doanh\Code\IS403.R11-M5`:

- [Cấu hình](../configs/config.yaml)
- [Dự báo Two-Part](../results/validation/validation_smoke_CA_1_20261119T060917732180Z/lightgbm_two_part.csv)
- [Dự báo Seasonal Naïve](../results/validation/validation_smoke_CA_1_20261119T060917732180Z/seasonal_naive.csv)
- [Chi tiết thực tế và hai thành phần](../results/validation/validation_smoke_CA_1_20261119T060917732180Z/actual_vs_predictions.csv)
- [Bảng điểm tổng](../results/metrics/validation_smoke_CA_1_20261119T060917732180Z/summary.csv)
- [Sai số từng sản phẩm Two-Part](../results/metrics/validation_smoke_CA_1_20261119T060917732180Z/lightgbm_two_part_per_item.csv)
- [Sai số từng sản phẩm baseline](../results/metrics/validation_smoke_CA_1_20261119T060917732180Z/seasonal_naive_per_item.csv)
- [Thư mục năm hình](../results/figures/validation_smoke_CA_1_20261119T060917732180Z/)
- [Bundle mô hình](../results/models/validation_smoke_CA_1_20261119T060917732180Z/model.joblib), [classifier](../results/models/validation_smoke_CA_1_20261119T060917732180Z/classifier.joblib), [regressor](../results/models/validation_smoke_CA_1_20261119T060917732180Z/regressor.joblib)

Các kết quả và CSV dữ liệu là hiện vật cục bộ được Git bỏ qua. Mã nguồn, hướng dẫn, kiểm thử và báo cáo nằm trong các thay đổi chưa commit trên nhánh Two-Part.
