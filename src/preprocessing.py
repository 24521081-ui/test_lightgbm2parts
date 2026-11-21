"""Giai đoạn 2: chuẩn bị lịch sử Train, giữ nguyên các ngày bán bằng 0."""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("m5")


def prepare_history(wide, calendar, origin, prices=None):
    """Chỉ chuyển các ngày Train thành dạng dài để chặn rò rỉ từ gốc."""
    # 2.1. Giới hạn dữ liệu ở origin, không đưa nhãn holdout vào Train.
    days = [f"d_{d}" for d in range(1, origin + 1)]
    metadata = ["item_id", "store_id", "state_id"]
    selected = wide[metadata + days].copy()
    # Mã lặp lại hàng triệu lần: category lưu mã số nhỏ thay cho chuỗi ở mỗi hàng.
    for column in metadata:
        selected[column] = selected[column].astype("category")
    # 2.2. Chuyển bảng rộng thành một dòng cho mỗi sản phẩm và ngày.
    frame = selected.melt(id_vars=metadata, var_name="d", value_name="sales")
    del selected
    frame["day"] = frame.d.str[2:].astype("int16")
    # 2.3. Ghép lịch, sự kiện và SNAP của đúng bang.
    frame = frame.merge(
        calendar[
            [
                "d",
                "wm_yr_wk",
                "day_of_week",
                "month",
                "event_flag",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
        ],
        on="d",
        how="left",
        validate="many_to_one",
    )
    if frame.wm_yr_wk.isna().any():
        raise ValueError("Thiếu ngày Train trong lịch.")
    frame["snap"] = np.select(
        [frame.state_id == state for state in ["CA", "TX", "WI"]],
        [frame[f"snap_{state}"] for state in ["CA", "TX", "WI"]],
        default=np.nan,
    )
    if frame.snap.isna().any():
        raise ValueError("Không xác định được SNAP của bang.")
    frame = frame.sort_values(["store_id", "item_id", "day"]).reset_index(drop=True)
    # 2.4. Nếu bật giá, chỉ điền tiếp từ giá đã biết trong quá khứ.
    if prices is not None:
        frame = frame.merge(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
            validate="many_to_one",
        )
        # Chỉ truyền giá quá khứ về phía trước trong cùng chuỗi, tuyệt đối không bfill.
        frame["sell_price"] = frame.groupby(
            ["store_id", "item_id"], observed=True
        ).sell_price.ffill()
        LOGGER.info(
            "Tỷ lệ thiếu giá sau ffill: %.2f%%", frame.sell_price.isna().mean() * 100
        )
    keep = [
        "item_id",
        "store_id",
        "day",
        "sales",
        "day_of_week",
        "month",
        "event_flag",
        "snap",
    ]
    if prices is not None:
        keep.append("sell_price")
    frame = frame[keep].copy()
    for column in ["item_id", "store_id"]:
        frame[column] = frame[column].astype("category")
    for column in ["sales", "snap"]:
        frame[column] = frame[column].astype("float32")
    return frame


def history_matrix(wide, origin):
    """Ma trận sản phẩm × ngày, giữ cùng thứ tự với wide."""
    return wide[[f"d_{d}" for d in range(1, origin + 1)]].to_numpy(dtype=np.float32)
