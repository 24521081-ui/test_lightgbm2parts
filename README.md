# LightGBM Two-Part dự báo lượng bán Walmart M5

Nhánh `lightgbm-two-part` phát triển từ `tweedie` (commit `2f05d89`). Giữ `run.py`, sáu module nghiệp vụ và đặc trưng của Tweedie để so sánh cùng đầu vào. Không push lên GitHub.

## Kết quả hiện tại và bàn giao

Manifest hiện có ghi nhận Validation C5/C6/C7 trên d_1886–d_1913 và Test đóng băng C6/C7 trên d_1914–d_1941 đã hoàn tất cho 10 cửa hàng. C7 chính dùng λ = 0,75; script λ = 0,50 và các thử nghiệm C1–C4 được giữ để tái lập lịch sử.

- [Biểu đồ quan trọng bằng tiếng Việt](reports/bieu_do_quan_trong/README.md): sáu hình PNG 250 DPI, có nguồn dữ liệu và giới hạn sử dụng.
- [Kiểm tra hồi quy sau refactor](reports/refactor_audit/regression_check.md): phạm vi PASS và NOT VERIFIED.
- [Kiểm kê, khác biệt cấu hình và lưu ý trước push](reports/refactor_audit/repository_review.md).

Tạo lại hình từ kết quả đã lưu, không Train hoặc chạy lại Validation:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts/build_important_figures.py
.\.venv\Scripts\python.exe scripts/refactor_audit.py check
```

`configs/config.yaml` vẫn là cấu hình C0 tắt giá. Các script C5/C6/C7 lấy cấu hình đóng băng và nguồn model theo manifest, dùng `previous_completed_week`; không đổi YAML sang C7 vì việc đó sẽ làm hỏng giả định tái lập C0–C4. `run.py --mode validation` có thể huấn luyện khi không chỉ định model đã lưu. Chỉ dùng lệnh đọc kết quả ở trên khi cần vẽ lại báo cáo.

Các script đang dùng:

| Mục đích | Script | Ghi chú |
|---|---|---|
| Pipeline C0, Train/Validation/Test | `run.py` | Cấu hình YAML; xem lệnh bên dưới |
| Validation C5 và C6 | `scripts/run_c5_price_features.py`, `scripts/run_c6_platt.py` | Có bước Train trong quy trình gốc |
| Kiểm tra nhiều gốc lịch sử | `scripts/run_stage_b_robustness.py` | Giữ kiểm tra nguồn model và cache |
| C7 lịch sử, λ = 0,75 | `scripts/run_c7_lambda075.py` | `--stores CA_1` hoặc bỏ tùy chọn để chạy 10 cửa hàng |
| C7 Validation chính | `scripts/run_c7_validation_1885.py` | Tái sử dụng C5 và hệ số Platt C6; cache từng cửa hàng |
| Test đóng băng | `scripts/run_c6_final_test.py`, `scripts/run_c7_final_test.py` | Chỉ thực hiện sau khi đã chốt cấu hình; C7 dùng lại model C6 |

Đợt refactor không huấn luyện lại model nghiên cứu hoặc chạy lại toàn bộ 10 cửa hàng. Đã chạy lại suy luận C7 CA_1 để đối chiếu trước/sau. Các kết quả gốc được giữ nguyên trên ổ đĩa; `results/`, dữ liệu lớn, model, môi trường ảo và cache được bỏ qua khi thêm file vào Git.

## Bốn chỗ cần dùng

| Việc cần làm | Vị trí |
|---|---|
| Chạy chương trình | Mở PowerShell tại thư mục chứa [run.py](run.py), dùng `.\.venv\Scripts\python.exe run.py --mode smoke --store CA_1` |
| Chỉnh cấu hình | [configs/config.yaml](configs/config.yaml) |
| Xem dự báo đã có | [Thư mục Smoke CA_1](results/validation/validation_smoke_CA_1_20261119T060917732180Z/): `lightgbm_two_part.csv`, `seasonal_naive.csv`, `actual_vs_predictions.csv` |
| Xem biểu đồ đã có | [Năm hình Smoke CA_1](results/figures/validation_smoke_CA_1_20261119T060917732180Z/), bắt đầu từ `01_actual_vs_forecast.png` |

Máy hiện có `.venv` và kết quả Smoke; có thể dùng ngay lệnh chạy trên. Phần cài đặt dưới đây dành cho khi cần thiết lập môi trường mới. Lệnh `run.py` sẽ huấn luyện và tạo kết quả mới; để đọc kết quả đã có, mở trực tiếp CSV hoặc PNG.

## Cài đặt trên PowerShell

Mở Terminal tại repo:

```powershell
Set-Location -LiteralPath 'D:\Phân tích dữ liệu kinh doanh\Code\IS403.R11-M5'
$env:PYTHONUTF8 = '1'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --mode smoke --store CA_1
```

Có thể dùng Python 3.11 trở lên, bản 64 bit. Nếu `python` trỏ tới Microsoft Store, chọn Python đã cài bằng đường dẫn đầy đủ. Môi trường thực tế đã kiểm tra được ghi trong [báo cáo](docs/bao_cao_thuc_nghiem.md). Không cần kích hoạt môi trường hoặc thay ExecutionPolicy khi dùng trực tiếp `.venv\Scripts\python.exe`.

## Hai CSV đầu vào

Đặt `sales_train_evaluation.csv` và `calendar.csv` trong `data/raw/`. Giữ `.gitkeep`; CSV được `.gitignore` bỏ qua.

Nguồn trên máy: `D:\Phân tích dữ liệu kinh doanh\Dataset_mau\m5-forecasting-accuracy`.

| Tệp nguồn thực tế | Nội dung đã kiểm tra | Tệp đích |
|---|---|---|
| `sales_train_validation.csv` | d_1–d_1941, ID `_evaluation` | `data/raw/sales_train_evaluation.csv` |
| `sales_train.csv` | d_1–d_1913, ID `_validation` | Không dùng cho pipeline này |
| `calendar.csv` | d, date, wm_yr_wk, sự kiện, SNAP | `data/raw/calendar.csv` |

Tên nguồn dễ gây nhầm: luôn kiểm tra header và ID trước khi chép, không dựa vào tên hiển thị trong Explorer. Không ghi đè file đích khác nội dung. Lần triển khai này đã chép hai bản sao, giữ nguyên nguồn. `sell_prices.csv` chỉ cần khi bật giá; `sample_submission.csv` là mẫu Kaggle, `paper_m5.pdf` là tài liệu.

## Mô hình và cấu hình

Chỉnh [configs/config.yaml](configs/config.yaml). `classifier` là LGBMClassifier binary học nhãn có bán trên mọi dòng đủ đặc trưng; `regressor` là LGBMRegressor regression (bình phương sai số) học lượng bán thật chỉ ở các dòng dương. Mỗi cửa hàng có một cặp mô hình chung cho các sản phẩm.

Dự báo = `probability_positive × conditional_units`. Chặn phần lượng bán có điều kiện về không âm; từ chối NaN/vô cực. Không làm tròn. Mỗi bước đưa chính dự báo tích vào lịch sử tạm, rồi tạo đặc trưng ngày tiếp theo. Seasonal Naïve dùng lag 7 và đệ quy bằng dự báo của chính nó.

Cả hai nhóm mặc định: 200 cây, learning_rate 0.05, num_leaves 31, random_state 42, CPU 6 luồng, deterministic. Đây là cấu hình khởi điểm, chưa tối ưu. Không dùng holdout làm eval_set hoặc early stopping; không cân bằng lớp tự động vì cần xác suất có bán. Train một lớp dùng xác suất hằng 0/1; không có dòng dương dùng lượng bán có điều kiện hằng 0 và lưu chính sách vào metadata.

`smoke_items: 30`, `seed: 42` chọn tối đa 30 mã theo mã và seed, không theo sai số. `use_prices: false` giữ cùng Tweedie. Khi bật giá, Train chỉ ffill trong chuỗi; dự báo giữ giá cuối biết tại origin, không lấy giá tương lai.

Đặc trưng giữ từ Tweedie: item dạng category, thứ (thứ Hai=0), tháng, cờ sự kiện, SNAP theo bang, lag 1/7/14/28 và rolling mean 7/28 với shift 1. Tạo lag/rolling trên toàn lịch sử kể cả số 0 trước khi lọc dòng dương cho hồi quy. Bỏ 28 ngày đầu khỏi fit do chưa đủ cửa sổ, vẫn giữ chúng trong lịch sử.

## Validation và Test

Đánh giá C0 trên **chính 10 cặp mô hình đã Train**, không fit lại:

```powershell
.\.venv\Scripts\python.exe run.py --mode validation --all-stores --scope full --config configs/config.yaml --from-training-summary results/models/train_C0_20261119_summary.csv
```

Kết quả gộp ở `results/validation/validation_C0_full_10_stores_<UTC>/`; workbook ở `results/metrics/validation_C0_full_10_stores_<UTC>/validation_summary.xlsx`. CSV chi tiết thêm `signed_error_two_part`, `absolute_error_two_part`, `signed_error_seasonal_naive`, `absolute_error_seasonal_naive`. Sai lệch có dấu = dự báo − thực tế: dương là dự báo thừa, âm là dự báo thiếu. Tính trên số chưa làm tròn; chỉ định dạng Excel/hình khi hiển thị. Workbook có tổng lượng bán 28 ngày toàn bộ/từng cửa hàng, sai lệch trung bình và MAE theo horizon toàn bộ/từng cửa hàng; sheet `Model_sources` dẫn tới CSV/hình và các mô hình gốc. Hình 01 và 05 thêm ô sai lệch theo ngày có vạch 0.

Chỉ huấn luyện Full C0 đến d_1885 và lưu mô hình cho 10 cửa hàng, **không dự báo/chấm điểm**:

```powershell
.\.venv\Scripts\python.exe run.py --mode train --all-stores --scope full --config configs/config.yaml --resume-training
```

Chế độ này yêu cầu C0 (200 cây, learning rate 0.05, 31 lá, tắt giá), chạy tuần tự và lưu tại `results/models/train_full_<store>_<UTC>/`. Mỗi bundle có hai mô hình, thứ tự đặc trưng, danh mục item_id và `origin_day=1885`; metadata ghi số dòng fit từng phần, số cây thực tế, thời gian và xác minh nạp lại. `--resume-training` chỉ bỏ qua cửa hàng đã hoàn tất, cùng cấu hình và nạp lại kiểm tra thành công. Nếu một cửa hàng lỗi, các cửa hàng khác vẫn được xử lý; chạy lại cùng lệnh để tiếp tục phần thiếu. Validation không kèm `--from-training-summary` vẫn huấn luyện lại.

| Chế độ | Học đến | Dự báo |
|---|---|---|
| smoke / validation | d_1885 | d_1886–d_1913 |
| test | d_1913, học lại | d_1914–d_1941 |

```powershell
.\.venv\Scripts\python.exe run.py --mode validation --store CA_1 --scope full
.\.venv\Scripts\python.exe run.py --mode validation --all-stores --scope full
.\.venv\Scripts\python.exe run.py --mode validation --store CA_1 --scope smoke --config configs/config.yaml
```

Một cửa hàng qua `--store`; nhiều cửa hàng dùng `--all-stores` chạy tuần tự 10 cửa hàng theo giao diện Tweedie. Full yêu cầu 3049 sản phẩm/cửa hàng và giữ toàn lịch sử. Lượng bán thật 28 ngày chỉ đọc sau khi hoàn thành cả hai dự báo, dùng riêng để chấm điểm.

Chỉ sau khi chốt cấu hình từ Validation mới chạy:

```powershell
.\.venv\Scripts\python.exe run.py --mode test --store CA_1 --scope full --confirm-final-test
```

Các lệnh ở phần này mô tả pipeline C0 ban đầu. Trạng thái C5/C6/C7 hiện tại nằm ở phần bàn giao đầu README và các manifest, không suy ra từ kết quả Smoke.

## Mở kết quả

Mỗi cửa hàng/lần chạy có tên riêng với thời điểm UTC có microsecond, tránh ghi đè:

```text
results/validation/<run>/lightgbm_two_part.csv
results/validation/<run>/seasonal_naive.csv
results/validation/<run>/actual_vs_predictions.csv
results/test/<run>/...                        # chỉ khi chạy Test
results/metrics/<run>/summary.csv
results/metrics/<run>/*_per_item.csv
results/models/<run>/model.joblib             # bundle cả cặp + đặc trưng/category
results/models/<run>/classifier.joblib
results/models/<run>/regressor.joblib
results/models/<run>/classifier.txt           # khi thành phần là LightGBM
results/models/<run>/regressor.txt
results/models/<run>/metadata.json
results/figures/<run>/01_actual_vs_forecast.png
results/figures/<run>/02_cumulative_units.png
results/figures/<run>/03_mae_by_horizon.png
results/figures/<run>/04_model_gain_by_sales_pattern.png
results/figures/<run>/05_two_part_components.png
```

Hai CSV dự báo giữ đúng tám cột để ghép trên main:

```text
model,split,origin_day,item_id,store_id,target_day,horizon,y_pred
```

`model` là `lightgbm_two_part` hoặc `seasonal_naive`. Bảng chi tiết thêm `y_actual`, `y_pred_two_part`, `y_pred_seasonal_naive`, `probability_positive`, `conditional_units`, tỷ lệ ngày 0 Train và nhóm gián đoạn. Metadata lưu cấu hình thực tế, tham số hiệu lực, môi trường, mã được chọn, số dòng từng phần và kiểm tra dữ liệu. Chỉ mở joblib từ nguồn tin cậy.

MAE/RMSE tính trên toàn bộ mẫu. RMSSE từng chuỗi dùng mẫu số trung bình bình phương sai phân Train sau lần bán dương đầu tiên. Chuỗi toàn 0/hằng/không đủ điểm có RMSSE không xác định, không thay mẫu số bằng epsilon. `summary.csv` báo `RMSSE_mean_unweighted`, số chuỗi hợp lệ và không xác định; không phải WRMSSE chính thức M5. Bảng từng sản phẩm có MAE, RMSE, RMSSE, scale.

Hình 01 lấy mã đầu từ điển ở mỗi nhóm tỷ lệ ngày 0 Train: <25%, [25%,75%), >=75%. Hình 05 lấy mã có tỷ lệ ngày 0 cao nhất, hòa thì mã đầu từ điển. Khóa lựa chọn trước khi đọc nhãn holdout. Hình 02–04 dùng toàn tập sản phẩm đã chạy; mọi hình có nhãn SMOKE/FULL, cửa hàng, số sản phẩm và ngày. Chi tiết cách đọc và kết quả xác minh: [báo cáo thực nghiệm](docs/bao_cao_thuc_nghiem.md).

## Mã nguồn và kiểm tra

`run.py` điều phối `data_loader.py`, `preprocessing.py`, `feature_engineering.py`, `training.py`, `forecasting.py`, `reporting.py`. Hai module đầu giữ sườn Tweedie; loader kiểm tra thêm đầu vào evaluation. Kiểm tra:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q run.py src tests
.\.venv\Scripts\python.exe scripts/verify_two_part_results.py validation_smoke_CA_1_20261119T060917732180Z
```

Các báo cáo kiểm toán và hướng dẫn demo cũ còn nhắc Tweedie là tài liệu lịch sử của nhánh nền; dùng báo cáo refactor và manifest để xác định kết quả hiện tại. Các entrypoint hiện hành gồm `run.py`, các script thực nghiệm trong bảng đầu README và bộ kiểm thử trong `tests/`.

Môi trường Windows hiện tại có lỗi thứ tự import LightGBM/scikit-learn: lệnh pytest thông thường có thể báo thiếu scikit-learn dù đã cài. Lỗi đã xuất hiện trước refactor; không thay thư viện hoặc thuật toán để che lỗi. Lệnh kiểm tra đã xác minh trong môi trường này:

```powershell
.\.venv\Scripts\python.exe -c "import sklearn, pytest; raise SystemExit(pytest.main(['-q']))"
```
