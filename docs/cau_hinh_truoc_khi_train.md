# Cấu hình cần duyệt trước khi huấn luyện Two-Part

Đối chiếu ngày 19/11/2026 trên nhánh `lightgbm-two-part`; chỉ đọc mã/cấu hình và soạn tài liệu, **chưa chạy Train, Validation hay Test trong lượt này**. Số dòng nguồn dưới đây tính theo file hiện tại trong repo.

## 1. Cấu hình mô hình hiện tại

| Thành phần | Học đáp án gì; giữ dòng nào; trả về gì | Tham số hiện tại | Phạm vi và điều kiện dừng | Nguồn |
|---|---|---|---|---|
| LightGBM phân loại (`LGBMClassifier`) | Nhãn `1` nếu `sales > 0`, ngược lại `0`; giữ **mọi dòng đủ đặc trưng**, kể cả ngày bán 0; lấy xác suất lớp 1 bằng `predict_proba()[:, 1]`. | `objective=binary`; tối đa **200 cây**; `learning_rate=0.05`; tối đa **31 lá/cây**; `random_state=42`. | Mỗi cửa hàng một bộ phân loại học chung các sản phẩm. Không có early stopping theo Validation, không truyền `eval_set`/callback; số cây tối đa do `n_estimators`. Train một lớp dùng xác suất hằng 0 hoặc 1, không fit LightGBM. | `configs/config.yaml:13–23`; `src/training.py:25–32,37–49`; `run.py:50–56,236–243`. |
| LightGBM hồi quy (`LGBMRegressor`) | Nhãn là **lượng bán gốc**, không lấy log; chỉ giữ `sales > 0` **sau khi đã tạo đặc trưng**; ước lượng lượng bán khi có bán. Đầu ra sử dụng được chặn về ≥0. | `objective=regression` (bình phương sai số); tối đa **200 cây**; `learning_rate=0.05`; tối đa **31 lá/cây**; `random_state=42`. | Mỗi cửa hàng một bộ hồi quy dùng cùng tập biến với phân loại. Không có early stopping theo Validation. Không có dòng dương thì lượng bán có điều kiện hằng 0, không fit LightGBM. | `configs/config.yaml:24–34`; `src/training.py:25–35,41–56`; `src/feature_engineering.py:92–103`. |

Cả hai dùng CPU, `n_jobs=6`, `force_col_wise=true`, `deterministic=true`, `verbosity=-1` (YAML). Dự báo cuối = xác suất có bán × lượng bán có điều kiện; không làm tròn. **200 là mức tối đa, chưa xác minh số cây thực tế của lần Full chưa chạy.**

**Giá trị không ghi trong YAML nhưng thực sự được áp dụng:**

- `run.py:127–128` thêm `importance_type="gain"`, thay mặc định thư viện `"split"`.
- Constructor LightGBM đang cài mặc định: `boosting_type="gbdt"`, `max_depth=-1` (không giới hạn độ sâu bằng tham số này), `min_child_samples=20`, `min_child_weight=0.001`, `min_split_gain=0`, `subsample=1`, `subsample_freq=0`, `colsample_bytree=1`, `reg_alpha=reg_lambda=0`, `class_weight=None`, `subsample_for_bin=200000`. Nguồn: `.venv/Lib/site-packages/lightgbm/sklearn.py:1454–1476` (hồi quy), `1569–1591` (phân loại); không tự cân bằng lớp. Các giá trị đã có trong YAML được ưu tiên, ví dụ 200 cây và learning rate 0.05 thay mặc định thư viện 100 và 0.1.
- CLI mặc định `--mode smoke`, `--scope full`, `--config configs/config.yaml`; không chọn cửa hàng thì `CA_1`. Riêng mode Smoke luôn đổi scope thành Smoke, lấy tối đa `smoke_items=30`, seed lấy mẫu `42` (`run.py:53–56,200–211,236`; `src/data_loader.py:98–101`). `load_store()` mặc định `start_day=1`; `run.py` truyền rõ start_day cho holdout (`src/data_loader.py:52–60`; `run.py:167–173`). Các khóa YAML được truy cập trực tiếp trong `run.py` không có cơ chế tự điền nếu bị xóa.

**Phân biệt với Smoke cũ:** `results/models/validation_smoke_CA_1_20261119T060917732180Z/metadata.json` có `config` **khớp hoàn toàn YAML hiện tại**. Nhưng đó là scope Smoke, 30 mã CA_1, 55.710 dòng phân loại và 17.418 dòng hồi quy, không phải quy mô Full (`metadata.json:3–8,77–114`). Không dùng số dòng hay điểm Smoke để thay cho kết quả Full.

## 2. Dữ liệu Train và Validation

Quy mô chính dưới đây ứng với lệnh **Validation Full CA_1** ở cuối trang. Số dòng là tính từ thiết kế code, chưa chạy Full để xác nhận thực tế.

| Nội dung | Cấu hình/cách lấy thực tế | Nguồn |
|---|---|---|
| File nguồn | `data/raw/sales_train_evaluation.csv`: cột d_1–d_1941, ID `_evaluation`; lịch từ `data/raw/calendar.csv`. Đường dẫn tính từ thư mục repo. Không đọc `sample_submission.csv`; không cần `sell_prices.csv` vì giá tắt. | YAML:2,7; `run.py:227–235`; `src/data_loader.py:13–39,105–117`. |
| Cửa hàng và chuỗi dự kiến | `--store CA_1 --scope full`: **3.049 cặp sản phẩm–cửa hàng**, một cặp mô hình. Thay bằng `--all-stores`: code yêu cầu 10 cửa hàng, dự kiến **30.490 cặp**, chạy tuần tự và fit riêng từng cửa hàng. Full không lấy mẫu 30 mã. | `run.py:53–56,86–89,236–243`. |
| Lịch sử Train ≠ dòng fit | Lịch sử **d_1–d_1885**, giữ cả ngày 0: CA_1 dự kiến `3049×1885 = 5.747.365` dòng. Fit chỉ **d_29–d_1885** do cần đủ lag/rolling: phân loại `3049×1857 = 5.661.993` dòng. Hồi quy là số dòng `sales>0` trong tập đó: **chưa xác minh/chưa đếm cho Full**. Với 10 cửa hàng: dự kiến 57.473.650 dòng lịch sử và 56.619.930 dòng phân loại; số dòng hồi quy chưa đếm. | `src/preprocessing.py:11–23`; `src/feature_engineering.py:92–103,118–120`; `src/training.py:41–52`. |
| Validation | **d_1886–d_1913**, 28 ngày; dự kiến `3049×28 = 85.372` dự báo/mô hình tại CA_1 (853.720 nếu đủ 10 cửa hàng). Cố định mô hình trong 28 bước, mỗi mô hình cập nhật lịch sử bằng dự báo của chính nó. Nhãn thật holdout chỉ được nạp cho chấm điểm/xuất so sánh **sau khi cả hai dự báo đã hoàn tất**. | YAML:10–11; `run.py:140–173`; `src/forecasting.py:57–102`. |
| Ví dụ **d_1886** | Tất cả lịch sử là **thật**: `lag_1=y1885`, `lag_7=y1879`, `lag_14=y1872`, `lag_28=y1858`; rolling 7 = mean(y1879…y1885), rolling 28 = mean(y1858…y1885). Tạo ŷ1886 rồi thêm vào lịch sử tạm. | `src/feature_engineering.py:63–74`; `src/forecasting.py:61–72,100–102`. |
| Ví dụ **d_1887** | `lag_1=ŷ1886` (**dự báo Two-Part**); `lag_7=y1880`, `lag_14=y1873`, `lag_28=y1859` vẫn là **thật**. Rolling 7 = (sum(y1880…y1885)+ŷ1886)/7; rolling 28 = (sum(y1859…y1885)+ŷ1886)/28. **Không dùng y thật d_1886.** Naïve có lịch sử riêng, dùng ŷ của chính baseline. | Cùng nguồn trên; `run.py:154–159`. |
| Test cuối | Học lại bằng d_1–d_1913, fit từ d_29; dự báo **d_1914–d_1941**. Cần `--mode test --confirm-final-test`. Các mốc Train/Validation/Test trong code **không lệch yêu cầu**; chưa chạy Test cuối trong lượt này. | YAML:10–12; `run.py:31–36,53–54,220–221`. |

Trong hai ví dụ, `y` là lượng bán thật, `ŷ` là dự báo. Các ngày xa hơn tiếp tục đưa dự báo vào lag/rolling khi cửa sổ đi qua origin; Seasonal Naïve lấy `lag_7`, từ tuần thứ hai dùng dự báo tuần trước của chính nó (`src/forecasting.py:109–114`).

## 3. Biến đầu vào thực sự vào mô hình

Hai phần dùng **cùng 11 biến hiện tại**, cùng thứ tự; hồi quy chỉ khác tập dòng/nhãn. Ký hiệu C/R = phân loại và hồi quy. Với ngày cần dự báo t, `z` là lịch sử thật đến origin rồi nối các dự báo của chính mô hình.

| Biến | File/cột nguồn và cách lấy | Phần dùng; vai trò |
|---|---|---|
| `item_id` | CSV doanh số, cột `item_id`; chuyển sang category, lưu danh mục, truyền `categorical_feature=['item_id']`. | C/R; phân biệt sản phẩm trong cửa hàng. |
| `day_of_week` | `calendar.csv:date` → thứ Hai=0…Chủ nhật=6, **không dùng trực tiếp cột wday**. | C/R; mùa vụ theo thứ. |
| `month` | `calendar.csv:date` → tháng 1…12. | C/R; mùa vụ theo tháng. |
| `event_flag` | 1 nếu `event_name_1` **hoặc** `event_name_2` không thiếu, ngược lại 0; không đưa tên sự kiện vào model. | C/R; dấu hiệu ngày có sự kiện. |
| `snap` | Chọn `snap_CA`, `snap_TX` hoặc `snap_WI` theo `state_id` của cửa hàng; CA_1 dùng `snap_CA`. | C/R; lịch hỗ trợ SNAP theo bang. |
| `lag_1` | Doanh số `d_*` → `sales`; `z[t−1]`. | C/R; lượng bán ngày gần nhất. |
| `lag_7` | Cùng nguồn; `z[t−7]`. | C/R; cùng thứ tuần trước; cũng là dự báo của Naïve. |
| `lag_14` | Cùng nguồn; `z[t−14]`. | C/R; cùng thứ hai tuần trước. |
| `lag_28` | Cùng nguồn; `z[t−28]`. | C/R; cùng thứ bốn tuần trước. |
| `rolling_mean_7` | Cùng nguồn; mean(z[t−7]…z[t−1]); Train dùng `shift(1).rolling(7, min_periods=7).mean()`. | C/R; mức bán gần đây trong 7 ngày. |
| `rolling_mean_28` | Cùng nguồn; mean(z[t−28]…z[t−1]); Train dùng `shift(1).rolling(28, min_periods=28).mean()`. | C/R; mức bán nền trong 28 ngày. |
| `sell_price` — **đang tắt** | `use_prices:false`, không nằm trong X. Nếu bật: `sell_prices.csv:sell_price`, nối store/item/`wm_yr_wk`; Train chỉ ffill trong chuỗi, dự báo giữ giá cuối biết tại origin (`last_known`). | Hiện không dùng; nếu bật sẽ vào C/R. Chưa xác minh thực nghiệm nhánh giá trong lượt này. |
| `store_id` — **không vào X** | CSV doanh số; lọc cửa hàng, nhóm chuỗi, khóa ghép/xuất kết quả; `state_id` chọn SNAP. `id`, `dept_id`, `cat_id`, `d/day`, doanh số hiện tại cũng không vào X. | Chia phạm vi huấn luyện: **mỗi cửa hàng một cặp mô hình**, không phải một đặc trưng dự báo. |

Nguồn bảng biến: `src/feature_engineering.py:6–17,40–48,63–74,92–103`; `src/data_loader.py:133–141`; `src/preprocessing.py:14–23,44–63`; `src/training.py:48–52`; `run.py:104–110`. Code **tạo lag/rolling trên toàn lịch sử gồm ngày 0 trước**, rồi mới lọc nhãn dương ở bước fit hồi quy; không tính lag giữa riêng các lần có bán.

## Lệnh chờ duyệt và tiêu chí so sánh

Lệnh Validation Full CA_1, đã đối chiếu định nghĩa CLI (`run.py:200–211`), **chưa thực thi**:

```powershell
Set-Location -LiteralPath 'D:\Phân tích dữ liệu kinh doanh\Code\IS403.R11-M5'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe run.py --mode validation --store CA_1 --scope full --config configs/config.yaml
```

Nếu duyệt chạy đủ 10 cửa hàng, thay `--store CA_1` bằng `--all-stores`; giữ `--mode validation --scope full`. Đầu ra theo lần chạy riêng dưới `results/validation`, `metrics`, `models`, `figures` (`run.py:57–66`).

**Tiêu chí đề xuất để duyệt cấu hình:** ưu tiên `RMSSE_mean_unweighted` thấp hơn trên cùng tập chuỗi/cửa sổ, kèm số chuỗi RMSSE hợp lệ/không xác định; xem thêm MAE, RMSE và so Seasonal Naïve. Scale RMSSE chỉ từ Train sau lần bán dương đầu tiên; scale không xác định không thay bằng epsilon. Đây **không phải WRMSSE chính thức M5** (`src/reporting.py:31–59,90–109`). Code xuất cả ba chỉ số, **không tự tìm/chọn cấu hình tốt nhất**.

**Cần bạn duyệt trước khi chạy:** (1) CA_1 Full hay đủ 10 cửa hàng; (2) giữ 200 cây/0.05/31 lá cho cả hai phần và CPU 6 luồng; (3) tiếp tục tắt giá; (4) dùng RMSSE trung bình làm tiêu chí chính, cùng quy tắc so sánh; (5) chỉ Validation lúc này, khóa cấu hình trước Test cuối. Số dòng hồi quy Full, thời gian và RAM Full **chưa xác minh**, không suy từ Smoke thành số đo Full.
