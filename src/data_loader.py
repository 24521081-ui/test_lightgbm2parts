"""Giai đoạn 1: đọc dữ liệu M5 theo cửa hàng và khoảng ngày cần sử dụng."""

from pathlib import Path
import csv

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("m5")

SALES_FILE = "sales_train_evaluation.csv"
IDENTIFIERS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def check_files(directory, use_prices=False):
    """Thông báo tất cả file đã có/thiếu; giá chỉ bắt buộc khi bật tính năng."""
    missing = []
    for name in [SALES_FILE, "calendar.csv", "sell_prices.csv"]:
        exists = (Path(directory) / name).is_file()
        LOGGER.info("%s: %s", name, "đã có" if exists else "còn thiếu")
        if not exists and (name != "sell_prices.csv" or use_prices):
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"Không tìm thấy {', '.join(missing)} trong {directory}. "
            "Vui lòng tải dữ liệu M5 và đặt vào thư mục này."
        )
    with (Path(directory) / SALES_FILE).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        first = next(reader, None)
    days = [name for name in header if name.startswith("d_")]
    if (days != [f"d_{day}" for day in range(1, 1942)]
            or len(header) != len(set(header))
            or not set(IDENTIFIERS).issubset(header)
            or first is None or not first[header.index("id")].endswith("_evaluation")):
        raise ValueError("Cần CSV evaluation đầy đủ d_1–d_1941 và ID _evaluation.")


def list_stores(directory):
    """Chỉ đọc cột cửa hàng để liệt kê, không đọc toàn bộ lượng bán."""
    stores = set()
    for chunk in pd.read_csv(
        Path(directory) / SALES_FILE, usecols=["store_id"], chunksize=2000
    ):
        stores.update(chunk.store_id)
    return sorted(stores)


def load_store(
    directory,
    store,
    end_day,
    chunk_size=500,
    sample_items=None,
    seed=42,
    start_day=1,
    item_ids=None,
):
    """Chọn mẫu bằng mã sản phẩm, hoàn toàn không nhìn doanh số tương lai."""
    LOGGER.info("Đọc dữ liệu cửa hàng %s đến d_%d", store, end_day)
    if not 1 <= start_day <= end_day:
        raise ValueError("Khoảng ngày đọc dữ liệu không hợp lệ.")
    days = [f"d_{d}" for d in range(start_day, end_day + 1)]
    dtypes = {**{c: "string" for c in IDENTIFIERS}, **{d: "float32" for d in days}}
    # 1.1. Đọc từng khối, chỉ giữ cửa hàng và sản phẩm cần xử lý.
    parts = []
    for chunk in pd.read_csv(
        Path(directory) / SALES_FILE,
        usecols=IDENTIFIERS + days,
        dtype=dtypes,
        chunksize=chunk_size,
    ):
        selected = chunk.loc[chunk.store_id == store]
        if item_ids is not None:
            selected = selected.loc[selected.item_id.isin(item_ids)]
        if not selected.empty:
            parts.append(selected.copy())
    if not parts:
        raise ValueError(f"Không có cửa hàng {store} trong dữ liệu.")
    # 1.2. Cố định thứ tự sản phẩm và kiểm tra doanh số đầu vào.
    wide = (
        pd.concat(parts, ignore_index=True)
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    if (
        wide[IDENTIFIERS].isna().any().any()
        or wide.duplicated(["item_id", "store_id"]).any()
        or not wide.id.str.endswith("_evaluation").all()
    ):
        raise ValueError("Mã sản phẩm bị thiếu hoặc chuỗi bị trùng.")
    values = wide[days].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Lượng bán bị thiếu, vô hạn hoặc âm.")
    # 1.3. Smoke chọn mẫu có seed, không chọn theo doanh số tương lai.
    if sample_items is not None:
        wide = wide.sample(n=min(sample_items, len(wide)), random_state=seed)
        wide = wide.sort_values("item_id").reset_index(drop=True)
    return wide


def load_calendar(directory):
    """Lịch tương lai là biến ngoại sinh; ngày và sự kiện được biết trước."""
    calendar = pd.read_csv(Path(directory) / "calendar.csv")
    required = [
        "d",
        "date",
        "wm_yr_wk",
        "event_name_1",
        "event_name_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    if not set(required).issubset(calendar.columns):
        raise ValueError("calendar.csv thiếu cột bắt buộc.")
    if (
        calendar.d.duplicated().any()
        or calendar[required[:3] + required[5:]].isna().any().any()
    ):
        raise ValueError("Lịch có ngày trùng hoặc giá trị bắt buộc bị thiếu.")
    dates = pd.to_datetime(calendar.date, errors="raise")
    day_numbers = calendar.d.str.removeprefix("d_").astype(int)
    ordered = pd.DataFrame({"day": day_numbers, "date": dates}).sort_values("day")
    if (
        not (ordered.day.diff().dropna() == 1).all()
        or not (ordered.date.diff().dropna().dt.days == 1).all()
    ):
        raise ValueError("Lịch phải liên tục theo ngày.")
    calendar["day_of_week"] = dates.dt.dayofweek.astype("int8")
    calendar["month"] = dates.dt.month.astype("int8")
    calendar["event_flag"] = (
        calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8")
    )
    for state in ["CA", "TX", "WI"]:
        if not calendar[f"snap_{state}"].isin([0, 1]).all():
            raise ValueError("SNAP chỉ được nhận 0 hoặc 1.")
        calendar[f"snap_{state}"] = calendar[f"snap_{state}"].astype("int8")
    calendar["wm_yr_wk"] = calendar.wm_yr_wk.astype("int32")
    return calendar


def load_prices(directory, store, chunk_size=100000):
    """Chỉ giữ giá của cửa hàng hiện tại; NaN được giữ, không biến thành 0 USD."""
    parts = []
    for chunk in pd.read_csv(
        Path(directory) / "sell_prices.csv",
        chunksize=chunk_size,
        dtype={"sell_price": "float32", "wm_yr_wk": "int32"},
    ):
        parts.append(chunk.loc[chunk.store_id == store].copy())
    prices = pd.concat(parts, ignore_index=True)
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Giá bị trùng khóa cửa hàng/sản phẩm/tuần.")
    known = prices.sell_price.dropna()
    if not np.isfinite(known).all() or (known <= 0).any():
        raise ValueError("Giá đã biết phải hữu hạn và lớn hơn 0.")
    return prices
