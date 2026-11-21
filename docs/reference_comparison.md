# Đối chiếu với nhánh recursive_store

Đã đọc hai tệp Python chính thức trong kho Mcompetitions/M5-methods, thư mục A1. Bản bàn giao là mã tự triển khai để học và thực nghiệm trên CPU.

## Huấn luyện

[Mã TRAIN chính thức](https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/2.%20train/1-1.%20recursive_store_TRAIN.py) dùng `lgb.train` theo cửa hàng, dữ liệu pickle đã tiền xử lý và đặc trưng giá, lịch, mean encoding, lag. Cấu hình có power 1.1, learning rate 0.015, 3000 cây, 2047 lá. Có lag 28–42 và rolling với shift 1/7/14, cửa sổ 7/14/30/60. END_TRAIN phụ thuộc KKK; không phải thiết kế holdout cố định của nhóm.

Ở đây dùng `LGBMRegressor.fit`, đọc CSV từng cửa hàng, power 1.2, 200 cây, 31 lá; chỉ các đặc trưng được yêu cầu. Không mean encoding, không tái tạo các nhánh hay ensemble. Chia 1885/1913 cố định theo nhóm. Đây là khác biệt có chủ đích để dễ học và giảm tài nguyên.

## Dự báo

[Mã PREDICT chính thức](https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/3.%20predict/1-1.%20recursive_store_PREDICT.py) lặp 28 ngày, tính lại rolling động, gọi mô hình lưu cho từng cửa hàng và ghi dự báo vào lịch sử tạm. Xuất dạng F1–F28 theo mẫu Kaggle.

Ở đây giữ nguyên nguyên lý đệ quy nhưng xử lý hết một cửa hàng rồi giải phóng bộ nhớ. Mô hình được nạp/giữ một lần; đặc trưng từ ma trận 28 ngày, CSV dạng dài tám cột. Giá mặc định tắt; khi bật chỉ giữ giá đã biết. Có kiểm thử leakage, baseline tuần, đánh giá và biểu đồ; chưa có WRMSSE đủ 12 cấp.
