# Kiểm kê repository và chuẩn bị bàn giao

Ngày kiểm tra: 23/11/2026. Phạm vi: 44 file nguồn/cấu hình/tài liệu, gồm 32 file Python sau khi bổ sung công cụ và kiểm thử; 1.729 artifact kết quả/model được kiểm kê, đọc schema/metadata và lập hash. Không coi việc hash một file nhị phân là đã kiểm tra nội dung mô hình của mọi cửa hàng.

## Phân loại file

| Nhóm | Kết luận |
|---|---|
| A — Mã đang dùng | `run.py` và sáu module `src/`; giữ đường dẫn và API |
| B — Hỗ trợ | `tests/`, script kiểm chứng/phân tích/báo cáo, cấu hình, tài liệu, công cụ audit và hình mới |
| C — Thực nghiệm tái lập | C1–C7, C6.1, Stage B và các entrypoint Test; giữ cả λ = 0,50 để truy vết lựa chọn cấu hình |
| D — Lịch sử không còn được gọi | Tài liệu Tweedie và các tiện ích nhánh nền có bối cảnh cũ; không đủ căn cứ xóa chỉ dựa vào tên hay không có import |
| E — Trùng lặp | Đã gom `diagnostics` C1/C2, `require` ở ba script, ghi CSV C7, vòng đệ quy C7 và loader C7 0,50. Không xóa file thực nghiệm |
| F — Kết quả phải giữ | Toàn bộ `results/`, model và dữ liệu gốc trên ổ đĩa; hash xác nhận không đổi |
| G — Cache/tạm | `.cache/`, `__pycache__/`, `.pytest_cache/`, `.venv/`, log, file tạm và model nhị phân: Git bỏ qua, không xóa dữ liệu khỏi đĩa |
| H — Chưa xác định | Không loại bỏ file nào vì chưa xác định được vai trò. Các file giữ lại có danh mục trong `source_inventory.csv` |

Danh mục CSV ghi tham chiếu tĩnh theo tên file/module trong nguồn và tài liệu. Đây là công cụ hỗ trợ rà soát; không suy ra một entrypoint độc lập là code chết khi không có nơi import nó. Các manifest, nguồn model và các kết quả đã hoàn tất xác nhận vai trò tái lập của C1–C7/Stage B.

Cấu trúc giữ nguyên `configs/`, `src/`, `scripts/`, `tests/`, `docs/`, `data/`, `results/`. Bổ sung `reports/refactor_audit/` và `reports/bieu_do_quan_trong/`. Không di chuyển hay xóa output gốc.

## Thay đổi thực hiện

11 file Python hiện có được sửa; 9 file ngắn hơn, giảm ròng 433 dòng ở các file cũ. Số này không trừ đi mã mới của helper, công cụ audit, công cụ vẽ hình và kiểm thử; không dùng số dòng làm thước đo bảo toàn kết quả.

- `scripts/experiment_common.py`: gom helper và vòng C7; lambda vẫn được truyền từ hằng số của từng entrypoint. Giữ phép nhân, chặn không âm, dtype của lịch sử/lag, thứ tự feature/item và cập nhật cửa sổ sau từng bước.
- `scripts/run_c7_lambda075.py`, `scripts/run_c7_lambda050.py`: giữ chữ ký `predict_c7`; script 0,50 dùng lại loader của 0,75. Giữ kiểm tra cutoff, category và đầy đủ feature giá. Kiểm tra hữu hạn `mu` riêng của script 0,50 vẫn được thực hiện.
- `scripts/run_c1_classifier15.py`, `scripts/run_c2_classifier48.py`: dùng lại đúng công thức `diagnostics`, không đổi tham số hay flow Train.
- `scripts/analyze_c5_by_groups.py`, `scripts/diagnose_c5_bias.py`, `scripts/run_c4_with_prices_lagged.py`: dùng chung hàm `require`; không bỏ điều kiện kiểm tra.
- `scripts/run_c7_validation_1885.py`, `scripts/run_c7_final_test.py`: gom ghi CSV UTF-8 BOM; in kết quả hoặc REUSE gọn theo cửa hàng, giữ cache và schema.
- `src/forecasting.py`, `scripts/run_c5_price_features.py`: log từng ngày chuyển INFO sang DEBUG; cảnh báo và lỗi không bị tắt.
- `.gitignore`: bổ sung môi trường, file bí mật, bytecode, log/tạm, joblib/pickle; dữ liệu và `results/` vẫn được giữ cục bộ.
- `requirements.txt`: khai báo trực tiếp SciPy vì script đã import `scipy.special`; không cài lại hoặc thay phiên bản môi trường.
- README được cập nhật theo manifest hiện có; bổ sung script dựng hình chỉ đọc output, công cụ hash baseline và tám trường hợp kiểm thử mới.

Chi tiết fingerprint và số dòng từng file: `source_changes.json`. Các tên hàm đã bỏ thân cục bộ vẫn được import dưới tên cũ để giữ API. Thông báo lỗi loader C7 0,50 dùng câu chữ của loader 0,75; lớp exception và điều kiện từ chối giữ nguyên.

## Validation và pipeline C5/C6/C7

C5 vẫn dùng `p_raw × mu_positive`; C6 dùng Platt từ dự báo lịch sử ngoài mẫu; C7 dùng `(1 − λ) × p_raw + λ × p_cal`, rồi nhân `mu_positive`. C7 chính λ = 0,75, lịch sử 0,50 giữ riêng. C7 chạy đệ quy bằng chính dự báo của nó, không hậu xử lý forecast C6.

Terminal C7 theo cửa hàng gồm trạng thái COMPLETE/REUSE, cấu hình/lambda, origin, cửa hàng, số chuỗi RMSSE hợp lệ, RMSSE, MAE, bias và đường dẫn output. Bảng tổng hợp cuối lượt vẫn được giữ; dữ liệu chi tiết vẫn nằm trong CSV/JSON. Không có thay đổi cơ chế chạy từng cửa hàng, mặc định 10 cửa hàng hoặc resume theo cache cũ.

Không gom toàn bộ C5/C6/Stage B thành một pipeline mới trong đợt này: các nguồn model, cutoff, schema và yêu cầu hiệu chuẩn khác nhau, và chưa chạy lại huấn luyện Full để chứng minh một thay đổi lớn hơn. Các đoạn kiểm tra dài có vai trò kiểm chứng được giữ lại.

## Khác biệt và rủi ro ghi nhận riêng

1. Repository này triển khai nhánh Two-Part và Seasonal Naive, chưa có bộ kết quả hoàn chỉnh của cả sáu phương pháp nghiên cứu trên cùng tập đánh giá. Không tạo hình so sánh sáu mô hình bằng dữ liệu thiếu.
2. YAML mặc định vẫn C0: tắt giá, 200 cây, 31 lá, `last_known`. Manifest C5 ghi classifier 300 cây/48 lá, regressor 200 cây/31 lá và chính sách thực nghiệm `previous_completed_week`; nested config còn trường `last_known`. Script/manifest thực tế quyết định chính sách theo thiết kế hiện có. Không tự sửa YAML hay metadata gốc cho khớp mô tả.
3. README cũ nói chưa chạy Full/Test, nhưng manifest mới xác nhận Validation C5/C6/C7 và Test C6/C7 đã hoàn tất; đã cập nhật phần bàn giao. Manifest C6 Test lưu `c7_test_performed=false` tại thời điểm chỉ chạy C6; manifest so sánh C6/C7 sau đó mới xác nhận C7 hoàn tất, nên phải đọc đúng nguồn/thời điểm.
4. Các cache kiểm tra không đồng nhất: C7 0,75 lịch sử kiểm tra version/lambda và hai bảng chỉ số; C7 0,50 và Test kiểm tra chặt hơn. Không tự nâng kiểm tra cache trong đợt clean code vì có thể thay đổi hành vi resume của kết quả lịch sử. Không coi tồn tại file là đủ để xác minh provenance mới.
5. Xác suất thành phần C7 không được lưu riêng trong các fold gốc; chưa có dữ liệu cho Reliability Diagram C7. Không thay bằng phép pha trộn xác suất trên đường C6.
6. Lỗi import LightGBM/scikit-learn đã có trước refactor; pytest cần nạp scikit-learn trước trong môi trường hiện tại. Cần xử lý môi trường riêng nếu muốn lệnh chuẩn chạy ngay sau khi clone. Không có bằng chứng lỗi này thay đổi kết quả nghiên cứu đã lưu.
7. Nhiều nguồn model trong manifest dùng đường dẫn tuyệt đối của máy hiện tại, batch tag cố định và dữ liệu/model được Git bỏ qua. Clone chỉ gồm code sẽ không tự có model hay kết quả để dựng lại hình. Cần bàn giao artifact riêng và giữ cấu trúc thư mục, hoặc chuẩn bị đợt sửa đường dẫn với kiểm chứng tương ứng. Không tự sửa manifest đã chụp baseline.

## Trước khi push

Đã chuẩn bị để người dùng review; chưa xác nhận khả năng chạy Full từ một máy sạch. Không commit/push và không stage file. Các thay đổi đã tồn tại trước đợt này được ghi trong `baseline_before.json`, không hoàn nguyên hoặc ghi đè.

Nên review và thêm vào Git: code/config/dependency/tài liệu cần thiết đang thay đổi, các script C1–C7/Stage B/kiểm chứng và tests đang untracked, helper/công cụ mới, hai thư mục `reports/`. Các `results/**/.gitkeep` có thể giữ để tạo khung thư mục.

Không thêm dữ liệu M5, `results/` thật, model, môi trường ảo, cache hoặc script tạm trong `.cache/`. Đã xác minh Git bỏ qua các đường dẫn đại diện; không có file tracked hiện tại lớn hơn 5 MB. `baseline_before.json` khoảng 12,4 MB vì chứa metadata chi tiết và cấu hình: đây là báo cáo kiểm toán được yêu cầu, không phải dữ liệu Train/model; cần xem xét dung lượng khi chọn file đưa lên Git.

Lệnh kiểm tra không ghi Git:

```powershell
git status --short
git diff --check
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/refactor_audit.py check
```

Các phần đã kiểm chứng và chưa kiểm chứng được nêu cụ thể trong [regression_check.md](regression_check.md). Không tuyên bố toàn bộ 10 cửa hàng đã chạy lại sau refactor.
