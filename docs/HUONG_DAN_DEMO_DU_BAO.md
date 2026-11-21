# Hướng dẫn demo dự báo LightGBM–Tweedie và Seasonal Naive

## 1. Demo này thực hiện việc gì?

Nạp mô hình LightGBM đã huấn luyện, dự báo 28 ngày cho một sản phẩm, chạy Seasonal Naive trên cùng lịch sử, rồi so sánh với lượng bán thật của đoạn Validation. Không huấn luyện lại và không chạy Test cuối. Đây là phát lại một giai đoạn lịch sử chưa được dùng để fit mô hình, không phải dự báo ngày hiện tại.

Script: `scripts/demo_saved_forecast.py`. Lệnh `run.py` hiện có luôn gọi huấn luyện trước dự báo, nên không dùng lệnh đó trong buổi demo này. Cờ `run.py --demo` mang nghĩa dữ liệu giả lập và vẫn huấn luyện; khác với demo bằng mô hình đã lưu ở đây.

## 2. Thực hiện lần lượt trong PowerShell

### Bước 1 — Mở đúng thư mục

Mở Terminal trong VS Code, chọn PowerShell, chạy:

```powershell
Set-Location C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie
```

### Bước 2 — Kiểm tra Python đang có

```powershell
& ..\.venv\Scripts\python.exe --version
```

Đây là môi trường đã có tại `PTDLKD\.venv`. Không cần kích hoạt môi trường và không cần tạo môi trường mới. Dấu `&` là cách PowerShell gọi chương trình.

### Bước 3 — Chạy dự báo bằng mô hình đã lưu

```powershell
& ..\.venv\Scripts\python.exe scripts/demo_saved_forecast.py
```

Mặc định dùng lần chạy `validation_full_CA_1_20261106T152044652294Z`, cùng lần CA_1 được chọn trong báo cáo tổng hợp mới. Sản phẩm mặc định là mã đầu tiên theo alphabet trong tập đã huấn luyện, không chọn theo kết quả dự báo đẹp nhất.

Chương trình sẽ:

1. Đọc metadata, xác nhận mô hình thuộc Validation với origin d_1885 và horizon 28.
2. Nạp `results/models/<tên_lần_chạy>/model.joblib`, gồm mô hình, đặc trưng và mã phân loại.
3. Đọc lịch sử thật d_1–d_1885 của một sản phẩm và đọc lịch ngày.
4. Dự báo LightGBM lần lượt ngày d_1886 đến d_1913; log hiện bước 1/28 đến 28/28.
5. Dự báo Seasonal Naive trên cùng lịch sử; log cũng hiện đủ 28 bước.
6. Đối chiếu dự báo vừa tạo với CSV của chính lần chạy nguồn; dừng nếu lệch ngoài ngưỡng cho phép.
7. Sau khi dự báo xong mới đọc lượng bán thật d_1886–d_1913 để tính sai số.
8. Lưu CSV, biểu đồ và thông tin nguồn vào thư mục mới `results/demo_predictions/<thời_điểm_UTC>/`.

Việc đọc lịch sử có thể mất thời gian vì CSV gốc chứa cả 10 cửa hàng. Thời gian demo không phải thời gian riêng của hàm dự báo.

Trong lần chạy kiểm chứng, bước nạp thư viện phụ thuộc của mô hình mất hơn một phút trước khi có log dự báo. Vì vậy không dùng toàn bộ thời gian khởi động để kết luận tốc độ suy luận của LightGBM. Script hiển thị thông báo nạp mô hình trước bước này.

### Bước 4 — Xác định thư mục vừa sinh

```powershell
$demoResult = Get-ChildItem .\results\demo_predictions -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1
$demoResult.FullName
```

Đối chiếu đường dẫn này với dòng `OUTPUT_DIR=...` cuối Terminal. Mỗi lần chạy tạo thư mục mới; kết quả cũ giữ nguyên.

### Bước 5 — Mở đồ thị trước

```powershell
Invoke-Item (Join-Path $demoResult.FullName 'prediction_demo.png')
```

Giải thích ba đường: Thực tế là nhãn chỉ dùng khi đánh giá; LightGBM–Tweedie là dự báo từ mô hình đã học; Seasonal Naive là dự báo lặp tuần. Trục ngang là bước dự báo 1–28, trục dọc là lượng bán.

Một sản phẩm có thể được LightGBM dự báo kém hơn baseline. Không dùng một biểu đồ đơn lẻ để thay kết luận từ toàn bộ 30.490 chuỗi.

### Bước 6 — Xem dự báo từng ngày và sai số

```powershell
Import-Csv (Join-Path $demoResult.FullName 'comparison.csv') | Format-Table -AutoSize
Import-Csv (Join-Path $demoResult.FullName 'summary.csv') | Format-Table -AutoSize
```

`comparison.csv` có mã sản phẩm, ngày M5, horizon, hai dự báo, lượng bán thật và ngày lịch. `summary.csv` là chỉ số của **một sản phẩm trong 28 ngày**, không phải chỉ số cả CA_1 hoặc cả 10 cửa hàng.

`lightgbm_tweedie.csv` và `seasonal_naive.csv` giữ định dạng tám cột chuẩn của project. `y_pred` là lượng bán kỳ vọng, có thể là số thập phân. Không hiểu 1.7 là đã bán một phần sản phẩm và không tự làm tròn trước khi đánh giá.

### Bước 7 — Chứng minh không huấn luyện lại

```powershell
Get-Content -Encoding utf8 (Join-Path $demoResult.FullName 'provenance.json')
```

Chỉ ra `source_run`, `retrained: false`, `final_test_run: false` và độ lệch với dự báo đã lưu. Script không gọi `fit()`.

### Bước 8 — Trình bày kết quả toàn bộ 10 cửa hàng

```powershell
$aggregate = '.\results\aggregated\validation_full_10_stores_20261111T151420617263Z'
Invoke-Item (Join-Path $aggregate 'rmsse_mean_overall.png')
Invoke-Item (Join-Path $aggregate 'rmsse_mean_by_store.png')
Import-Csv (Join-Path $aggregate 'metrics_overall.csv') | Format-Table -AutoSize
```

Đọc báo cáo `CHUONG_5_VALIDATION_REPORT.md` trong cùng thư mục để xem công thức, số liệu từng cửa hàng và truy vết. Đây là báo cáo mới; tab báo cáo có tên thư mục `20261106T161515114372Z` đang mở trong IDE là bản cũ được giữ nguyên.

## 3. Chọn sản phẩm hoặc lần chạy khác

Ví dụ chỉ định rõ sản phẩm:

```powershell
& ..\.venv\Scripts\python.exe scripts/demo_saved_forecast.py --item HOBBIES_1_001
```

Ví dụ dùng đúng lần CA_1 trong tab dự báo cũ đang mở ở IDE:

```powershell
& ..\.venv\Scripts\python.exe scripts/demo_saved_forecast.py --run validation_full_CA_1_20261106T145400435654Z --item HOBBIES_1_001
```

Tên sản phẩm phải nằm trong tập đã huấn luyện. Với cửa hàng khác, truyền tên thư mục Full Validation tương ứng trong `results/models/` qua `--run`; script lấy cửa hàng từ metadata. Phiên bản demo này hỗ trợ mô hình không dùng giá, phù hợp toàn bộ mô hình của báo cáo hiện tại.

## 4. Hai mô hình khác nhau như thế nào?

| Nội dung | LightGBM–Tweedie hiện tại | Seasonal Naive hiện tại |
| --- | --- | --- |
| Cơ chế | Học tập hợp cây quyết định bằng boosting, tối ưu loss Tweedie | Sao chép lượng bán cùng thứ tuần trước, `predict(X) = X['lag_7']` |
| Phạm vi học | Một mô hình chung cho các sản phẩm trong một cửa hàng | Không có bước fit, mỗi chuỗi lặp mẫu riêng |
| Thông tin dùng | Mã sản phẩm, thứ, tháng, sự kiện, SNAP, lag 1/7/14/28, rolling mean 7/28 | Chỉ dùng lag 7 dù giao diện nhận cùng bảng đặc trưng |
| Dự báo 28 ngày | Tạo dự báo ngày tiếp theo, đưa dự báo vào lịch sử tạm, lặp lại | Lặp bảy ngày cuối Train bốn lần |
| Phản ứng với lịch | Có khả năng học quan hệ với lịch nếu dữ liệu Train có tín hiệu | Không tự điều chỉnh theo tháng, sự kiện hoặc SNAP |
| Chi phí | Cần xây đặc trưng, huấn luyện, lưu mô hình | Rất đơn giản, dễ kiểm tra, không cần huấn luyện |
| Rủi ro | Có thể quá khớp, làm trơn đỉnh bán và tích lũy sai số đệ quy | Lặp cả nhiễu hoặc tuần bất thường cuối lịch sử |

Ví dụ bảy ngày cuối Train là `[2, 0, 3, 1, 4, 2, 5]`: Seasonal Naive dự báo đúng mẫu đó bốn lần. Ngày thứ 8 lấy dự báo ngày thứ 1, không lấy lượng bán thật của ngày thứ 1 trong holdout. LightGBM không bắt buộc lặp lại mẫu này vì còn xét các đặc trưng khác và các quy luật đã học.

LightGBM là thuật toán học; Tweedie là mục tiêu tối ưu. Với power 1.2 thuộc khoảng (1, 2), mục tiêu phù hợp để mô hình hóa lượng bán không âm có nhiều số 0 và giá trị dương. Điều này không tự bảo đảm mô hình sẽ thắng baseline, tối ưu MAE hay dự báo đúng mọi đỉnh bán. Kết quả phải được kiểm tra thực nghiệm. Project hiện tắt giá bán nên không thể quy mức cải thiện cho thông tin giá.

## 5. Mạnh đến mức nào theo kết quả đã có?

Validation d_1886–d_1913, 10 cửa hàng, 30.490 chuỗi, 853.720 dự báo mỗi mô hình:

| Chỉ số | LightGBM–Tweedie | Seasonal Naive | Mức giảm sai số |
| --- | ---: | ---: | ---: |
| **RMSSE_mean_unweighted — chính** | **0.748373** | **0.975596** | **23,29%** |
| MAE — bổ sung | 1.069718 | 1.232355 | 13,20% |
| RMSE — bổ sung | 2.162939 | 2.759459 | 21,62% |

MAE 1.069718 nghĩa là độ lệch tuyệt đối trung bình khoảng 1,07 đơn vị bán trên một sản phẩm–cửa hàng–ngày của tập này. Không có nghĩa mọi dự báo đều lệch 1,07 hoặc sai 1,07%. MAE lớn/nhỏ còn phụ thuộc quy mô lượng bán.

RMSE cao hơn MAE cho thấy sai số không đồng đều, có những sai số lớn được bình phương và nhấn mạnh hơn. Mức giảm RMSE lớn hơn mức giảm MAE phù hợp với việc mô hình giảm tác động của sai số lớn ở cấp tổng hợp; chưa đủ để khẳng định mọi đỉnh bán đều được dự báo tốt hoặc xác định số lượng ngoại lệ.

RMSSE là sai số đã chuẩn hóa theo biến động sai phân một ngày trong Train. Trung bình 0.748373 không phải tỷ lệ sai 74,84%, cũng không phải độ chính xác 25,16%. RMSSE dưới 1 không tự chứng minh thắng Seasonal Naive 7 ngày, vì mẫu số dùng sai phân một ngày trong Train. Muốn kết luận thắng baseline phải so sánh trực tiếp hai điểm trên cùng holdout, như bảng trên.

LightGBM có cả ba chỉ số thấp hơn Seasonal Naive ở 10/10 cửa hàng. Mức giảm RMSSE từng cửa hàng khoảng 21,99–24,81%. TX_2 có RMSSE LightGBM thấp nhất, 0.689451; WI_2 cao nhất, 0.794073. Điều này không chứng minh LightGBM thắng trên từng sản phẩm hoặc từng ngày.

Thời gian fit đã ghi nhận của 10 mô hình cộng lại khoảng 323,46 giây, tức 5,39 phút; không gồm đọc dữ liệu, xây đặc trưng, dự báo và đánh giá. Baseline ghi thời gian fit bằng 0 vì không fit, không có nghĩa thực thi toàn bộ bằng 0 giây.

## 6. Giới hạn cần trình bày đúng

Đây là một cửa sổ Validation, cấu hình cố định 200 cây, chưa có cơ chế tự động tìm tham số hay chọn best model bằng MAE/RMSSE. Không gọi cấu hình này là tối ưu. Báo cáo dùng RMSSE làm chỉ số so sánh chính; mô hình vẫn học theo loss Tweedie.

RMSSE_mean_unweighted hiện tại chỉ lấy trung bình các chuỗi cấp sản phẩm–cửa hàng, không phải WRMSSE chính thức M5. Chưa đánh giá đủ 12 cấp phân cấp và trọng số doanh thu. Không suy thứ hạng thi M5, hiệu quả tồn kho hoặc mức giảm chi phí từ phần trăm giảm sai số.

Từ ngày dự báo thứ 2 trở đi, các đặc trưng lịch sử có thể chứa dự báo thay vì quan sát thật, nên sai số có thể truyền sang các bước sau. Việc này không bảo đảm sai số tăng đều theo horizon. Muốn kết luận cụ thể phải đọc biểu đồ sai số theo horizon và tính thêm thống kê tương ứng.

## 7. Lời trình bày mẫu

“Em sử dụng mô hình đã huấn luyện trên dữ liệu đến d_1885. Chương trình nạp mô hình và dự báo đệ quy 28 ngày tiếp theo; không cập nhật bằng doanh số thật trong giai đoạn dự báo. Seasonal Naive dùng cùng lịch sử nhưng chỉ lặp tuần cuối. Sau khi cả hai dự báo xong, chương trình mới đọc lượng bán thật để đánh giá. Demo một sản phẩm giúp minh họa cơ chế; kết luận chính dựa trên toàn bộ 30.490 chuỗi: LightGBM–Tweedie giảm RMSSE trung bình 23,29% so với Seasonal Naive trên cửa sổ Validation này. Đây là RMSSE trung bình không trọng số, không phải WRMSSE chính thức.”

## 8. Kết quả chạy kiểm chứng ngày 11/11/2026

Đã chạy thành công luồng demo với sản phẩm mặc định **FOODS_1_001**, cửa hàng **CA_1**. Kết quả tại `results/demo_predictions/20261111T152603575024Z/`. Khoảng dự báo là 28/03/2016–24/04/2016 theo lịch M5; ngày thực hiện demo không phải ngày được dự báo.

| Chỉ số của riêng sản phẩm demo | LightGBM–Tweedie | Seasonal Naive |
| --- | ---: | ---: |
| RMSSE | 0.768426 | 0.921871 |
| MAE | 0.969693 | 1.107143 |
| RMSE | 1.308510 | 1.569804 |

Độ lệch tuyệt đối lớn nhất giữa dự báo LightGBM phát lại và CSV nguồn khoảng `5.07e-08`, phù hợp sai số biểu diễn số khi đọc CSV; Seasonal Naive khớp hoàn toàn. Không fit mô hình và không chạy Test cuối.

Trên hình minh họa này, dự báo LightGBM dao động quanh khoảng 0,7–1,5, trong khi lượng bán thật có những ngày lên 4. Có thể chỉ trực tiếp các đỉnh bị dự báo thấp để giải thích rằng sai số tổng thể tốt hơn không có nghĩa nắm được mọi biến động ngày. Seasonal Naive hiện rõ mẫu lặp theo tuần. Đây là nhận xét của riêng hình demo, không suy rộng thành hình dạng mọi sản phẩm.
