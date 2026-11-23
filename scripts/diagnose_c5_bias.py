
"""Chan doan sai lech du bao C5 - LightGBM Two-Part.

Khong Train lai mo hinh.
Khong sua cac ket qua C0-C5.

Phan ra:
    error = (p - z) * mu + z * (mu - y)

p  : xac suat co ban
mu : luong ban co dieu kien
z  : 1 neu thuc te co ban, nguoc lai 0
y  : luong ban thuc te
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from experiment_common import require


ROOT = Path(__file__).resolve().parents[1]

C5_TAG = (
    "validation_C5_full_10_stores_"
    "20261121T140034335643Z"
)

METRICS = (
    ROOT / "results" / "metrics" / C5_TAG
)

OUTPUT = (
    ROOT / "results" / "analysis"
    / C5_TAG / "bias_diagnostics"
)

COLUMNS = [
    "store_id",
    "item_id",
    "target_day",
    "horizon",
    "y_actual",
    "y_pred_two_part",
    "probability_positive",
    "conditional_units",
]


# ============================================================
# 1. TINH CHI SO CHAN DOAN
# ============================================================


def summarize(frame):

    actual = float(
        frame["y_actual"].sum()
    )

    forecast = float(
        frame["y_pred_two_part"].sum()
    )

    positive = (
        frame["positive"].astype(bool)
    )

    actual_positive = float(
        frame.loc[
            positive, "y_actual"
        ].sum()
    )

    mu_positive = float(
        frame.loc[
            positive, "conditional_units"
        ].sum()
    )

    return pd.Series({

        "n_observations": len(frame),

        "actual_units": actual,

        "forecast_units": forecast,

        "over_forecast_units": (
            forecast - actual
        ),

        "bias_pct": (
            100 * (forecast / actual - 1)
            if actual else np.nan
        ),

        "classifier_term_units": float(
            frame["classifier_term"].sum()
        ),

        "regressor_term_units": float(
            frame["regressor_term"].sum()
        ),

        "positive_rate_pct": (
            100 * frame["positive"].mean()
        ),

        "mean_probability_pct": (
            100 * frame[
                "probability_positive"
            ].mean()
        ),

        "probability_gap_pp": (
            100 * (
                frame["probability_positive"]
                - frame["positive"]
            ).mean()
        ),

        "brier_score": float(
            frame["brier"].mean()
        ),

        "conditional_mu_bias_pct_on_selling_days": (
            100 * (
                mu_positive / actual_positive - 1
            )
            if actual_positive else np.nan
        ),
    })


# ============================================================
# 2. TINH HIEU CHUAN XAC SUAT THEO 10 KHOANG
# ============================================================

def probability_calibration(
    frame,
    group_columns,
):

    table = (
        frame.groupby(
            group_columns,
            observed=True,
        )
        .agg(
            n=("positive", "size"),

            mean_probability=(
                "probability_positive", "mean"
            ),

            observed_positive_rate=(
                "positive", "mean"
            ),
        )
        .reset_index()
    )

    table["probability_gap_pp"] = (
        100 * (
            table["mean_probability"]
            - table["observed_positive_rate"]
        )
    )

    table["probability_interval"] = (
        table["probability_bin"].map(
            lambda i: (
                f"[{i / 10:.1f}, "
                f"{(i + 1) / 10:.1f}"
                f"{']' if i == 9 else ')'}"
            )
        )
    )

    return table


# ============================================================
# 3. CHUONG TRINH CHINH
# ============================================================

def main():

    manifest_path = (
        METRICS / "manifest.json"
    )

    require(
        manifest_path.is_file(),
        f"Thieu manifest: {manifest_path}",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        manifest.get("configuration") == "C5"
        and manifest.get("status") == "completed",
        "C5 chua hoan tat.",
    )

    sources = manifest["sources"]

    require(
        len(sources) == 10,
        "Thieu ket qua 10 cua hang.",
    )

    frames = []

    # --------------------------------------------------------
    # BƯỚC 1: ĐỌC DỰ BÁO CHI TIẾT 10 CỬA HÀNG
    # --------------------------------------------------------

    for source in sources:

        store = source["store_id"]

        path = (
            ROOT
            / "results"
            / "validation"
            / source["run_tag"]
            / "actual_vs_predictions.csv"
        )

        if not path.is_file():
            path = Path(
                source["actual_vs_predictions"]
            )

        require(
            path.is_file(),
            f"Thieu du bao cua {store}: {path}",
        )

        frame = pd.read_csv(
            path,
            usecols=COLUMNS,
            float_precision="round_trip",
        )

        require(
            len(frame) == 3049 * 28,
            f"Sai so dong: {store}",
        )

        require(
            frame["store_id"].eq(store).all(),
            f"Sai store_id: {store}",
        )

        require(
            not frame.duplicated(
                [
                    "store_id",
                    "item_id",
                    "target_day",
                ]
            ).any(),
            f"Trung khoa: {store}",
        )

        require(
            set(frame["horizon"])
            == set(range(1, 29)),
            f"Thieu horizon: {store}",
        )

        expected_day = (
            "d_"
            + (
                1885 + frame["horizon"]
            ).astype(str)
        )

        require(
            frame["target_day"]
            .eq(expected_day)
            .all(),
            f"Sai target_day: {store}",
        )

        # ----------------------------------------------------
        # BƯỚC 2: KIỂM TRA DỮ LIỆU
        # ----------------------------------------------------

        numeric_columns = [
            "y_actual",
            "y_pred_two_part",
            "probability_positive",
            "conditional_units",
        ]

        numeric = frame[
            numeric_columns
        ].to_numpy(dtype=float)

        require(
            np.isfinite(numeric).all(),
            f"Du lieu khong huu han: {store}",
        )

        require(
            (
                frame[
                    [
                        "y_actual",
                        "y_pred_two_part",
                        "conditional_units",
                    ]
                ] >= 0
            ).all().all(),
            f"Du lieu am: {store}",
        )

        require(
            frame["probability_positive"]
            .between(0, 1)
            .all(),
            f"Xac suat khong hop le: {store}",
        )

        # ----------------------------------------------------
        # BƯỚC 3: LẤY CÁC THÀNH PHẦN TWO-PART
        # ----------------------------------------------------

        p = frame[
            "probability_positive"
        ].to_numpy(dtype=float)

        mu = frame[
            "conditional_units"
        ].to_numpy(dtype=float)

        y = frame[
            "y_actual"
        ].to_numpy(dtype=float)

        yhat = frame[
            "y_pred_two_part"
        ].to_numpy(dtype=float)

        z = (
            y > 0
        ).astype(float)

        # ----------------------------------------------------
        # BƯỚC 4: PHÂN RÃ SAI LỆCH
        # ----------------------------------------------------

        classifier_term = (
            (p - z) * mu
        )

        regressor_term = (
            z * (mu - y)
        )

        # Kiểm tra: dự báo = p * mu.
        np.testing.assert_allclose(
            yhat,
            p * mu,
            rtol=1e-10,
            atol=1e-9,
        )

        # Kiểm tra hai thành phần cộng đúng sai lệch.
        np.testing.assert_allclose(
            yhat - y,
            classifier_term + regressor_term,
            rtol=1e-10,
            atol=1e-9,
        )

        frame["positive"] = z

        frame["classifier_term"] = (
            classifier_term
        )

        frame["regressor_term"] = (
            regressor_term
        )

        # Brier score: sai số bình phương xác suất.
        frame["brier"] = (
            p - z
        ) ** 2

        # 10 khoảng xác suất từ 0 đến 1.
        frame["probability_bin"] = (
            np.minimum(
                np.floor(p * 10).astype(int),
                9,
            )
        )

        frames.append(frame)

        print(
            f"Da kiem tra {store}: "
            f"{len(frame):,} dong"
        )

    # --------------------------------------------------------
    # BƯỚC 5: GỘP 10 CỬA HÀNG
    # --------------------------------------------------------

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    require(
        len(data) == 853720,
        "Tong du bao khong du 853720 dong.",
    )

    require(
        data["item_id"]
        .groupby(data["store_id"])
        .nunique()
        .eq(3049)
        .all(),
        "Khong du 3049 san pham moi cua hang.",
    )

    # --------------------------------------------------------
    # BƯỚC 6: THỐNG KÊ TỔNG THỂ
    # --------------------------------------------------------

    overall = (
        summarize(data)
        .to_frame()
        .T
    )

    # --------------------------------------------------------
    # BƯỚC 7: THỐNG KÊ TỪNG CỬA HÀNG
    # --------------------------------------------------------

    by_store = pd.DataFrame({
        store: summarize(group)

        for store, group in data.groupby(
            "store_id",
            sort=True,
        )
    }).T

    by_store.index.name = "store_id"

    # --------------------------------------------------------
    # BƯỚC 8: THỐNG KÊ TỪNG NGÀY DỰ BÁO
    # --------------------------------------------------------

    by_horizon = pd.DataFrame({
        day: summarize(group)

        for day, group in data.groupby(
            "horizon",
            sort=True,
        )
    }).T

    by_horizon.index.name = "horizon"

    # --------------------------------------------------------
    # BƯỚC 9: KIỂM TRA HIỆU CHUẨN XÁC SUẤT
    # --------------------------------------------------------

    by_probability = (
        probability_calibration(
            data,
            ["probability_bin"],
        )
    )

    by_store_probability = (
        probability_calibration(
            data,
            [
                "store_id",
                "probability_bin",
            ],
        )
    )

    # --------------------------------------------------------
    # BƯỚC 10: ĐỐI CHIẾU VỚI KẾT QUẢ C5 GỐC
    # --------------------------------------------------------

    reference = pd.read_csv(
        METRICS / "comparison_overall.csv",
        encoding="utf-8-sig",
    )

    reference = reference.loc[
        reference["configuration"].eq("C5")
    ]

    require(
        len(reference) == 1,
        "Thieu ket qua tong hop C5.",
    )

    np.testing.assert_allclose(
        [
            overall.loc[0, "actual_units"],
            overall.loc[0, "forecast_units"],
        ],
        [
            float(reference.iloc[0]["actual"]),
            float(reference.iloc[0]["prediction"]),
        ],
        atol=1e-5,
        rtol=0,
    )

    np.testing.assert_allclose(
        (
            overall["classifier_term_units"]
            + overall["regressor_term_units"]
        ),
        overall["over_forecast_units"],
        atol=1e-5,
        rtol=0,
    )

    # --------------------------------------------------------
    # BƯỚC 11: XUẤT BÁO CÁO
    # --------------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall.to_csv(
        OUTPUT / "01_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_store.reset_index().to_csv(
        OUTPUT / "02_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_horizon.reset_index().to_csv(
        OUTPUT / "03_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_probability.to_csv(
        OUTPUT / "04_probability_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_store_probability.to_csv(
        OUTPUT / "05_probability_bins_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # BƯỚC 12: IN KẾT QUẢ NHANH
    # --------------------------------------------------------

    print(
        "\n=== C5: CHAN DOAN TONG THE ==="
    )

    print(
        overall.T.to_string(
            header=False,
            float_format=lambda v: f"{v:,.4f}",
        )
    )

    print(
        "\n=== C5: CHAN DOAN THEO CUA HANG ==="
    )

    selected = [
        "bias_pct",
        "probability_gap_pp",
        "conditional_mu_bias_pct_on_selling_days",
        "classifier_term_units",
        "regressor_term_units",
    ]

    print(
        by_store[selected]
        .round(3)
        .to_string()
    )

    print(
        "\nC5 DIAGNOSTICS COMPLETE"
    )

    print(
        f"Ket qua luu tai: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
