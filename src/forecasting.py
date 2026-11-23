"""Giai đoạn 5: dự báo đệ quy, không nhận doanh số thật của tương lai."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engineering import next_day_features

LOGGER = logging.getLogger("m5")

FORECAST_COLUMNS = [
    "model",
    "split",
    "origin_day",
    "item_id",
    "store_id",
    "target_day",
    "horizon",
    "y_pred",
]


def recursive_forecast(
    model,
    history,
    items,
    store,
    state,
    calendar,
    origin,
    features,
    categories,
    split="validation",
    horizon=28,
    last_prices=None,
    model_name="lightgbm_two_part",
    return_components=False,
):
    """Giữ mô hình cố định, thêm dự báo vào lịch sử trước bước tiếp theo."""
    history = np.asarray(history, dtype=np.float32)
    if (
        history.ndim != 2
        or history.shape[0] != len(items)
        or history.shape[1] != origin
    ):
        raise ValueError("Lịch sử phải có dạng số sản phẩm × origin ngày.")
    if (
        len(set(items)) != len(items)
        or not np.isfinite(history).all()
        or (history < 0).any()
    ):
        raise ValueError("Mã bị trùng hoặc lượng bán lịch sử không hợp lệ.")
    # Chỉ 28 ngày gần nhất cần thiết cho đặc trưng, Train vẫn dùng toàn bộ lịch sử.
    # 5.1. Khởi tạo lịch sử tạm riêng cho mỗi mô hình.
    temporary = history[:, -28:].copy()
    calendar = calendar.set_index("d", verify_integrity=True)
    output = []
    components = []
    for step in range(1, horizon + 1):
        day = f"d_{origin + step}"
        if day not in calendar.index:
            raise ValueError(f"Lịch thiếu {day}.")
        # 5.2. Tạo đặc trưng cho ngày kế tiếp bằng công thức của giai đoạn 3.
        X = next_day_features(
            temporary, items, calendar.loc[day], state, categories, last_prices
        )
        # 5.3. Dự báo cho tất cả sản phẩm, kiểm tra và chặn giá trị âm.
        if return_components:
            probability, conditional = model.predict_components(X[features])
            prediction = probability * conditional
            components.append(pd.DataFrame({
                "item_id": list(items), "store_id": store, "target_day": day,
                "probability_positive": probability,
                "conditional_units": conditional,
            }))
        else:
            prediction = np.asarray(model.predict(X[features]), dtype=np.float64)
        if prediction.shape != (len(items),) or not np.isfinite(prediction).all():
            raise ValueError("Mô hình trả dự báo sai kích thước hoặc NaN/vô hạn.")
        prediction = np.maximum(prediction, 0)
        if not np.isfinite(prediction).all():
            raise ValueError("Dự báo vượt phạm vi số hữu hạn.")
        LOGGER.debug("Dự báo %s | %s: bước %d/%d", model_name, store, step, horizon)
        output.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "split": split,
                    "origin_day": f"d_{origin}",
                    "item_id": list(items),
                    "store_id": store,
                    "target_day": day,
                    "horizon": step,
                    "y_pred": prediction,
                }
            )
        )
        # Ngày sau dùng dự báo vừa tạo; tuyệt đối không thay bằng doanh số thực tế.
        # 5.4. Trượt cửa sổ và thêm dự báo để tính đặc trưng ngày tiếp theo.
        temporary = np.column_stack([temporary[:, 1:], prediction])
    forecast = pd.concat(output, ignore_index=True)[FORECAST_COLUMNS]
    if return_components:
        return forecast, pd.concat(components, ignore_index=True)
    return forecast


class SeasonalNaive:
    """Mốc so sánh lặp lại lượng bán cùng thứ tuần trước, kể cả qua tuần thứ hai."""

    def predict(self, X):
        """lag_7 chứa lịch sử thật hoặc dự báo trước đó của chính baseline."""
        return X["lag_7"].to_numpy()



def validate_forecasts(frame, horizon=28):
    """Kiểm tra định dạng nhóm cho cả đánh giá và xuất CSV, không sửa dữ liệu."""
    if not set(FORECAST_COLUMNS).issubset(frame.columns) or frame.empty:
        raise ValueError("Bảng dự báo rỗng hoặc sai cấu trúc cột.")
    keys = ["model", "split", "origin_day", "item_id", "store_id"]
    if frame.isna().any().any() or frame.duplicated(keys + ["horizon"]).any():
        raise ValueError("Dự báo có giá trị thiếu hoặc khóa bị trùng.")
    if frame.duplicated(["model", "split", "store_id", "item_id", "target_day"]).any():
        raise ValueError("Dự báo trùng mã sản phẩm và ngày đích.")
    if not np.isfinite(frame.y_pred).all() or (frame.y_pred < 0).any():
        raise ValueError("Dự báo phải hữu hạn và không âm.")
    if not np.isfinite(frame.horizon).all() or (frame.horizon % 1 != 0).any():
        raise ValueError("Horizon phải là số nguyên hữu hạn.")
    counts = frame.groupby(keys, observed=True).horizon.agg(["count", "min", "max"])
    if not (
        counts["count"].eq(horizon) & counts["min"].eq(1) & counts["max"].eq(horizon)
    ).all():
        raise ValueError("Mỗi sản phẩm phải có đủ các horizon.")
    if not frame.origin_day.str.fullmatch(r"d_[1-9]\d*").all():
        raise ValueError("origin_day phải có dạng d_<ngày nguyên dương>.")
    expected = frame.origin_day.str[2:].astype(int) + frame.horizon
    if not (frame.target_day == "d_" + expected.astype(int).astype(str)).all():
        raise ValueError("target_day không khớp origin_day và horizon.")


def export_forecasts(frame, path, horizon=28):
    """Kiểm tra trước khi xuất; giữ nguyên thứ tự tám cột đã thống nhất với nhóm."""
    if list(frame.columns) != FORECAST_COLUMNS:
        raise ValueError("Bảng dự báo sai thứ tự cột.")
    validate_forecasts(frame, horizon)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
