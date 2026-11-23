
"""Phan tich ket qua C5 so voi C4 theo nhom san pham.

Khong huan luyen lai mo hinh.
Khong chay Validation hoac Test moi.
Chi su dung du bao va diem RMSSE da luu.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common import require


ROOT = Path(__file__).resolve().parents[1]

C5_BATCH = (
    "validation_C5_full_10_stores_"
    "20261121T140034335643Z"
)

ORIGIN = 1885
N_SERIES = 30490

TRAIN_DAYS = [
    f"d_{i}" for i in range(1, ORIGIN + 1)
]

KEYS = ["store_id", "item_id"]


# ========================================================
# 1. DOC MANIFEST C4 VA C5
# ========================================================


def read_manifest(path):
    require(
        path.is_file(),
        f"Khong tim thay manifest: {path}",
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    require(
        data.get("status") == "completed",
        f"Batch chua hoan tat: {path}",
    )

    return data


# ========================================================
# 2. PHAN NHOM SAN PHAM CHI BANG TRAIN
# ========================================================

def read_training_groups(sales_path):

    require(
        sales_path.is_file(),
        f"Thieu sales Train: {sales_path}",
    )

    ids = [
        "item_id",
        "store_id",
        "cat_id",
        "dept_id",
    ]

    dtypes = {
        name: "string" for name in ids
    }

    dtypes.update({
        day: "float32"
        for day in TRAIN_DAYS
    })

    frames = []

    for chunk in pd.read_csv(
        sales_path,
        usecols=ids + TRAIN_DAYS,
        dtype=dtypes,
        chunksize=500,
    ):

        values = chunk[
            TRAIN_DAYS
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        require(
            np.isfinite(values).all()
            and (values >= 0).all(),
            "Sales Train co du lieu khong hop le.",
        )

        # Ty le ngay co luong ban bang 0.
        zero_fraction = (
            values == 0
        ).mean(axis=1)

        # Dung nguong giong training_patterns().
        patterns = np.select(
            [
                zero_fraction < 0.25,
                zero_fraction < 0.75,
            ],
            [
                "it_gian_doan",
                "gian_doan",
            ],
            default="rat_gian_doan",
        )

        part = chunk[ids].copy()

        part["zero_fraction_train"] = (
            zero_fraction
        )

        part["sales_pattern"] = patterns

        frames.append(part)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    require(
        len(result) == N_SERIES,
        "Du lieu khong du 30490 chuoi.",
    )

    require(
        not result.duplicated(KEYS).any(),
        "Trung khoa cua hang - san pham.",
    )

    return result


# ========================================================
# 3. DOC RMSSE TUNG SAN PHAM
# ========================================================

def read_saved_scores(
    manifest,
    label,
    metrics_dir,
):

    sources = manifest["sources"]

    require(
        len(sources) == 10,
        f"{label} thieu cua hang.",
    )

    frames = []

    for source in sources:

        store = source["store_id"]
        run_tag = source["run_tag"]

        path = (
            metrics_dir
            / run_tag
            / "lightgbm_two_part_per_item.csv"
        )

        require(
            path.is_file(),
            f"Thieu file: {path}",
        )

        frame = pd.read_csv(
            path,
            usecols=[
                "item_id",
                "RMSSE",
                "MAE",
                "RMSE",
                "scale",
            ],
        )

        require(
            len(frame) == 3049,
            f"{store}/{label}: thieu san pham.",
        )

        require(
            not frame.item_id.duplicated().any(),
            f"{store}/{label}: trung item.",
        )

        frame["store_id"] = store

        frame = frame.rename(
            columns={
                col: f"{col}_{label}"
                for col in [
                    "RMSSE",
                    "MAE",
                    "RMSE",
                    "scale",
                ]
            }
        )

        frames.append(frame)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    require(
        len(result) == N_SERIES,
        f"{label} khong du 30490 chuoi.",
    )

    return result


# ========================================================
# 4. DOI CHIEU C4 VA C5
# ========================================================

def attach_scores(groups, c4, c5):

    result = groups.merge(
        c4,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    require(
        result["_merge"].eq("both").all(),
        "C4 thieu chuoi khi JOIN.",
    )

    result = result.drop(
        columns="_merge"
    )

    result = result.merge(
        c5,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    require(
        result["_merge"].eq("both").all(),
        "C5 thieu chuoi khi JOIN.",
    )

    result = result.drop(
        columns="_merge"
    )

    # C4 va C5 phai co cung mau so RMSSE.
    np.testing.assert_allclose(
        result["scale_C4"],
        result["scale_C5"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    valid = (
        np.isfinite(result["RMSSE_C4"])
        & np.isfinite(result["RMSSE_C5"])
    )

    require(
        valid.all(),
        "Co chuoi khong co du RMSSE C4/C5.",
    )

    result["delta_RMSSE"] = (
        result["RMSSE_C5"]
        - result["RMSSE_C4"]
    )

    result["improved"] = (
        result["delta_RMSSE"] < -1e-12
    )

    result["worsened"] = (
        result["delta_RMSSE"] > 1e-12
    )

    return result


# ========================================================
# 5. TONG HOP RMSSE THEO NHOM
# ========================================================

def summarize(frame, columns):

    report = (
        frame.groupby(
            columns,
            dropna=False,
            observed=True,
        )
        .agg(
            n_series=("item_id", "size"),

            RMSSE_C4=("RMSSE_C4", "mean"),
            RMSSE_C5=("RMSSE_C5", "mean"),

            MAE_C4=("MAE_C4", "mean"),
            MAE_C5=("MAE_C5", "mean"),

            n_improved=("improved", "sum"),
            n_worsened=("worsened", "sum"),

            zero_fraction_mean=(
                "zero_fraction_train",
                "mean",
            ),
        )
        .reset_index()
    )

    report["delta_RMSSE_C5_minus_C4"] = (
        report["RMSSE_C5"]
        - report["RMSSE_C4"]
    )

    report["change_pct"] = (
        100
        * report["delta_RMSSE_C5_minus_C4"]
        / report["RMSSE_C4"]
    )

    report["pct_improved"] = (
        100
        * report["n_improved"]
        / report["n_series"]
    )

    report["n_unchanged"] = (
        report["n_series"]
        - report["n_improved"]
        - report["n_worsened"]
    )

    return report


# ========================================================
# 6. CHUONG TRINH CHINH
# ========================================================

def main():

    metrics_dir = (
        ROOT / "results" / "metrics"
    )

    c5_dir = (
        metrics_dir / C5_BATCH
    )

    c5_manifest = read_manifest(
        c5_dir / "manifest.json"
    )

    c4_tag = c5_manifest["c4_batch"]

    c4_manifest = read_manifest(
        metrics_dir
        / c4_tag
        / "manifest.json"
    )

    print(
        "Dang doc RMSSE tung san pham C4/C5..."
    )

    c4 = read_saved_scores(
        c4_manifest,
        "C4",
        metrics_dir,
    )

    c5 = read_saved_scores(
        c5_manifest,
        "C5",
        metrics_dir,
    )

    print(
        "Dang phan nhom du lieu Train..."
    )

    groups = read_training_groups(
        ROOT
        / "data"
        / "raw"
        / "sales_train_evaluation.csv"
    )

    combined = attach_scores(
        groups,
        c4,
        c5,
    )

    # Kiem tra trung voi RMSSE tong the da bao cao.
    reference = pd.read_csv(
        c5_dir / "comparison_overall.csv"
    )

    for label in ["C4", "C5"]:

        expected = float(
            reference.loc[
                reference["configuration"].eq(label),
                "RMSSE_mean_unweighted",
            ].iloc[0]
        )

        actual = float(
            combined[f"RMSSE_{label}"].mean()
        )

        require(
            np.isclose(
                expected,
                actual,
                atol=1e-8,
                rtol=0,
            ),
            f"RMSSE {label} khong trung bao cao.",
        )

    # ----------------------------------------------------
    # 7. TAO BAO CAO
    # ----------------------------------------------------

    output = (
        ROOT
        / "results"
        / "analysis"
        / C5_BATCH
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "by_category": ["cat_id"],

        "by_department": ["dept_id"],

        "by_sales_pattern": ["sales_pattern"],

        "by_category_and_pattern": [
            "cat_id",
            "sales_pattern",
        ],

        "by_store": ["store_id"],
    }

    for name, columns in reports.items():

        summary = summarize(
            combined,
            columns,
        )

        summary.to_csv(
            output / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if name in [
            "by_category",
            "by_sales_pattern",
        ]:

            print(f"\n=== {name} ===")

            print(
                summary[
                    columns + [
                        "n_series",
                        "RMSSE_C4",
                        "RMSSE_C5",
                        "change_pct",
                        "pct_improved",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    combined.to_csv(
        output / "per_item_C4_vs_C5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wins = int(
        combined["improved"].sum()
    )

    losses = int(
        combined["worsened"].sum()
    )

    print(
        f"\nDa phan tich {len(combined):,} chuoi."
    )

    print(
        f"C5 tot hon: {wins:,} chuoi."
    )

    print(
        f"C5 kem hon: {losses:,} chuoi."
    )

    print(
        f"Ket qua tai: {output}"
    )


if __name__ == "__main__":
    main()

