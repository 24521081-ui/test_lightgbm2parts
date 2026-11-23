"""Giai đoạn 3: tạo đặc trưng thống nhất cho huấn luyện và dự báo."""

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
HISTORY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    f"rolling_mean_{w}" for w in WINDOWS
]
BASE_FEATURES = [
    "item_id",
    "day_of_week",
    "month",
    "event_flag",
    "snap",
] + HISTORY_FEATURES


def add_history_features(frame):
    """Nhóm theo sản phẩm–cửa hàng; shift trước rolling để không nhìn ngày đích."""
    required = ["store_id", "item_id", "day", "sales"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Lịch sử rỗng hoặc thiếu mã/ngày/lượng bán.")
    # sort_values đã tạo bảng mới; không cần thêm một bản copy toàn bộ bảng dài.
    frame = frame.sort_values(["store_id", "item_id", "day"])
    if not np.isfinite(frame.sales).all() or (frame.sales < 0).any():
        raise ValueError("Lượng bán lịch sử phải hữu hạn và không âm.")
    if (
        not np.isfinite(frame.day).all()
        or (frame.day < 1).any()
        or (frame.day % 1 != 0).any()
    ):
        raise ValueError("Mã ngày phải là số nguyên dương.")
    differences = (
        frame.groupby(["store_id", "item_id"], observed=True).day.diff().dropna()
    )
    if not differences.eq(1).all():
        raise ValueError("Ngày của mỗi chuỗi phải liên tục, không thiếu hoặc trùng.")
    groups = frame.groupby(["store_id", "item_id"], observed=True)["sales"]
    # 3.1. Lag lấy doanh số cách ngày đích 1, 7, 14 và 28 ngày.
    for lag in LAGS:
        frame[f"lag_{lag}"] = groups.shift(lag).astype("float32")
    # 3.2. Shift trước rolling để trung bình không chứa doanh số ngày đích.
    for window in WINDOWS:
        frame[f"rolling_mean_{window}"] = groups.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).mean()
        ).astype("float32")
    return frame


def next_day_features(
    history, items, calendar_row, state, categories, last_prices=None
):
    """Tạo cùng công thức từ ma trận lịch sử tạm thời, không nhận y tương lai."""
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[0] != len(items):
        raise ValueError("Ma trận lịch sử không khớp danh sách sản phẩm.")
    if history.shape[1] < max(LAGS + WINDOWS):
        raise ValueError("Cần ít nhất 28 ngày lịch sử.")
    if state not in ["CA", "TX", "WI"] or not set(items).issubset(set(categories)):
        raise ValueError("Bang không hợp lệ hoặc mã sản phẩm chưa có trong mô hình.")
    frame = pd.DataFrame({"item_id": pd.Categorical(items, categories=categories)})
    for name in ["day_of_week", "month", "event_flag"]:
        frame[name] = np.int8(calendar_row[name])
    frame["snap"] = np.float32(calendar_row[f"snap_{state}"])
    for lag in LAGS:
        frame[f"lag_{lag}"] = history[:, -lag].astype("float32")
    for window in WINDOWS:
        frame[f"rolling_mean_{window}"] = (
            history[:, -window:].mean(axis=1, dtype=np.float64).astype("float32")
        )
    if last_prices is not None:
        frame["sell_price"] = np.asarray(last_prices, dtype=np.float32)
    return frame


def prepare_training_data(history):
    """3. Tạo và kiểm tra đặc trưng trước khi huấn luyện."""
    # Kiểm tra Train chỉ chứa một cửa hàng và cùng khoảng lịch sử cho mọi mã.
    if history.store_id.nunique() != 1:
        raise ValueError("Mỗi mô hình chỉ huấn luyện cho đúng một cửa hàng.")

    bounds = history.groupby(["store_id", "item_id"], observed=True).day.agg(
        ["min", "max"]
    )
    if not bounds["min"].eq(1).all() or bounds["max"].nunique() != 1:
        raise ValueError(
            "Train phải bắt đầu ở d_1 và kết thúc cùng mốc cho mọi sản phẩm."
        )

    frame = add_history_features(history)

    # 28 ngày đầu chưa đủ lịch sử cho lag_28; không phải xóa các ngày bán bằng 0.
    frame = frame.dropna(subset=HISTORY_FEATURES)
    features = BASE_FEATURES + (["sell_price"] if "sell_price" in frame else [])

    if frame.empty:
        raise ValueError("Không đủ Train sau cửa sổ khởi tạo.")

    # 3.3. Tách đầu vào X và mục tiêu y trước khi chuyển sang huấn luyện.
    X = frame[features]
    y = frame.sales
    numeric = X.select_dtypes(include="number")
    required = [name for name in numeric.columns if name != "sell_price"]

    if (
        not np.isfinite(numeric[required].to_numpy()).all()
        or np.isinf(numeric.to_numpy()).any()
    ):
        raise ValueError("Đặc trưng không hợp lệ. Chỉ giá được phép thiếu (NaN).")

    return {
        "X": X,
        "y": y,
        "features": features,
        "categories": list(X.item_id.cat.categories),
        "fit_start_day": 29,
        "fit_end_day": int(history.day.max()),
        "dropped_warmup_rows": len(history) - len(frame),
    }
