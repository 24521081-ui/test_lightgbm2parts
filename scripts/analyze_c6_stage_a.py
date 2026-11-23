
"""Chan doan C5 vs C6 - Giai doan A.

KHONG TRAIN LAI.
KHONG SUA MO HINH.

Danh gia:
1. Brier Score va hieu chuan xac suat.
2. Du bao du/thieu theo cua hang.
3. Sai lech theo horizon.
4. Ket qua theo nhom san pham.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]

C5_TAG = (
    "validation_C5_full_10_stores_"
    "20261121T140034335643Z"
)

N_ITEMS_PER_STORE = 3049
N_STORES = 10
HORIZON = 28

KEYS = [
    "store_id",
    "item_id",
    "horizon",
    "target_day",
]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, columns=None):
    check(
        path.is_file(),
        f"Thieu file: {path}",
    )

    return pd.read_csv(
        path,
        usecols=columns,
        float_precision="round_trip",
    )


# ======================================================
# 1. TIM BATCH C6 FULL
# ======================================================

def find_c6_batch(metrics, requested):

    if requested:
        return requested

    eligible = []

    for folder in metrics.glob(
        "validation_C6_platt_10_stores_*"
    ):

        path = folder / "manifest.json"

        if not path.is_file():
            continue

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            data.get("configuration") == "C6"
            and data.get("status") == "completed"
            and data.get("c5_batch") == C5_TAG
            and data.get("n_items")
            == N_STORES * N_ITEMS_PER_STORE
        ):
            eligible.append(folder.name)

    check(
        len(eligible) == 1,
        (
            f"Tim thay {len(eligible)} batch C6: "
            f"{eligible}. "
            "Neu co nhieu batch, su dung --c6-batch."
        ),
    )

    return eligible[0]


# ======================================================
# 2. DOC VA KIEM TRA DU BAO
# ======================================================

def read_detail(path, label):

    columns = KEYS + [
        "y_actual",
        "y_pred_two_part",
        "probability_positive",
        "conditional_units",
    ]

    if label == "C6":
        columns.append("probability_raw")

    df = read_csv(path, columns)

    check(
        len(df) == N_ITEMS_PER_STORE * HORIZON,
        f"{label}: Sai so dong {path}",
    )

    check(
        not df.duplicated(KEYS).any(),
        f"{label}: Trung khoa.",
    )

    check(
        df["item_id"].nunique()
        == N_ITEMS_PER_STORE,
        f"{label}: Thieu san pham.",
    )

    check(
        df["horizon"].between(1, HORIZON).all(),
        f"{label}: Horizon khong hop le.",
    )

    expected_day = (
        "d_"
        + (1885 + df["horizon"]).astype(str)
    )

    check(
        df["target_day"].eq(expected_day).all(),
        f"{label}: Sai target_day.",
    )

    numeric = (
        df.select_dtypes(include="number")
        .to_numpy()
    )

    check(
        np.isfinite(numeric).all(),
        f"{label}: Chua NaN hoac Inf.",
    )

    for name in [
        "y_actual",
        "y_pred_two_part",
        "conditional_units",
    ]:
        check(
            (df[name] >= 0).all(),
            f"{label}: Gia tri am: {name}",
        )

    p_columns = ["probability_positive"]

    if label == "C6":
        p_columns.append("probability_raw")

    for name in p_columns:
        check(
            df[name].between(0, 1).all(),
            f"{label}: Xac suat khong hop le.",
        )

    np.testing.assert_allclose(
        df["y_pred_two_part"],
        df["probability_positive"]
        * df["conditional_units"],
        rtol=1e-10,
        atol=1e-8,
    )

    return df


# ======================================================
# 3. TINH CHI SO DANH GIA
# ======================================================

def stats(df):

    y = df["y_actual"].to_numpy()
    z = (y > 0).astype(float)

    y5 = df[
        "y_pred_two_part_C5"
    ].to_numpy()

    y6 = df[
        "y_pred_two_part_C6"
    ].to_numpy()

    p5 = df[
        "probability_positive_C5"
    ].to_numpy()

    # Xac suat goc tai moi buoc cua C6.
    pr = df[
        "probability_raw"
    ].to_numpy()

    # Xac suat da hieu chuan tai moi buoc cua C6.
    pc = df[
        "probability_positive_C6"
    ].to_numpy()

    total = y.sum()

    def bias(predicted):
        return (
            100 * (predicted / total - 1)
            if total > 0
            else np.nan
        )

    return {
        "n_rows": len(df),

        "actual": float(total),

        "forecast_C5": float(y5.sum()),
        "forecast_C6": float(y6.sum()),

        "bias_C5_pct": bias(y5.sum()),
        "bias_C6_pct": bias(y6.sum()),

        "MAE_C5": float(
            np.abs(y5 - y).mean()
        ),

        "MAE_C6": float(
            np.abs(y6 - y).mean()
        ),

        "positive_rate_pct": (
            100 * float(z.mean())
        ),

        "p_C5_mean_pct": (
            100 * float(p5.mean())
        ),

        "p_raw_C6_path_pct": (
            100 * float(pr.mean())
        ),

        "p_cal_C6_path_pct": (
            100 * float(pc.mean())
        ),

        "p_gap_C5_pp": (
            100 * float((p5 - z).mean())
        ),

        "p_raw_C6_gap_pp": (
            100 * float((pr - z).mean())
        ),

        "p_cal_C6_gap_pp": (
            100 * float((pc - z).mean())
        ),

        "Brier_C5_path": float(
            ((p5 - z) ** 2).mean()
        ),

        "Brier_raw_C6_path": float(
            ((pr - z) ** 2).mean()
        ),

        "Brier_cal_C6_path": float(
            ((pc - z) ** 2).mean()
        ),
    }


def table_by(df, columns):

    rows = []

    grouping = (
        columns[0]
        if len(columns) == 1
        else columns
    )

    for key, group in df.groupby(
        grouping,
        sort=True,
        observed=True,
    ):

        if len(columns) == 1:
            key = (key,)

        rows.append({
            **dict(zip(columns, key)),
            **stats(group),
        })

    return pd.DataFrame(rows)


# ======================================================
# 4. KIEM TRA HIEU CHUAN XAC SUAT
# ======================================================

def reliability(df, scope_cols=None):

    scope_cols = scope_cols or []

    probability_types = [
        (
            "C5_own_path",
            "probability_positive_C5",
        ),
        (
            "raw_on_C6_path",
            "probability_raw",
        ),
        (
            "cal_on_C6_path",
            "probability_positive_C6",
        ),
    ]

    pieces = []

    for label, column in probability_types:

        tmp = df[
            scope_cols + ["y_actual", column]
        ].copy()

        tmp["bin"] = np.minimum(
            (
                tmp[column].to_numpy() * 10
            ).astype(int),
            9,
        )

        tmp["positive"] = (
            tmp["y_actual"] > 0
        ).astype(int)

        result = (
            tmp.groupby(
                scope_cols + ["bin"],
                observed=True,
                sort=True,
            )
            .agg(
                n=("positive", "size"),

                mean_probability=(
                    column,
                    "mean",
                ),

                actual_positive_rate=(
                    "positive",
                    "mean",
                ),
            )
            .reset_index()
        )

        result["gap_pp"] = (
            100
            * (
                result["mean_probability"]
                - result["actual_positive_rate"]
            )
        )

        result["probability_type"] = label

        pieces.append(result)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def calculate_ece(table, label):

    selected = table.loc[
        table["probability_type"].eq(label)
    ]

    return float(
        np.average(
            np.abs(selected["gap_pp"]) / 100,
            weights=selected["n"],
        )
    )


# ======================================================
# 5. DOC RMSSE THEO SAN PHAM
# ======================================================

def load_item_groups(root, c6_batch, stores):

    group_path = (
        root / "results"
        / "analysis"
        / C5_TAG
        / "per_item_C4_vs_C5.csv"
    )

    groups = read_csv(
        group_path,
        [
            "store_id",
            "item_id",
            "cat_id",
            "dept_id",
            "sales_pattern",
            "zero_fraction_train",
            "RMSSE_C5",
            "scale_C5",
        ],
    )

    check(
        len(groups)
        == N_STORES * N_ITEMS_PER_STORE,
        "Thieu nhom san pham C5.",
    )

    check(
        not groups.duplicated(
            ["store_id", "item_id"]
        ).any(),
        "Trung item trong C5.",
    )

    parts = []

    for store in stores:

        path = (
            root / "results"
            / "metrics"
            / f"{c6_batch}_{store}"
            / "lightgbm_two_part_per_item.csv"
        )

        frame = read_csv(
            path,
            [
                "store_id",
                "item_id",
                "RMSSE",
                "scale",
            ],
        )

        check(
            len(frame) == N_ITEMS_PER_STORE
            and frame["store_id"].eq(store).all(),
            f"Sai RMSSE C6: {store}",
        )

        parts.append(
            frame.rename(
                columns={
                    "RMSSE": "RMSSE_C6",
                    "scale": "scale_C6",
                }
            )
        )

    combined = groups.merge(
        pd.concat(parts, ignore_index=True),
        on=["store_id", "item_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    check(
        combined["_merge"].eq("both").all(),
        "C5/C6 khong cung san pham.",
    )

    combined = combined.drop(
        columns="_merge"
    )

    np.testing.assert_allclose(
        combined["scale_C5"].to_numpy(dtype=float),
    combined["scale_C6"].to_numpy(dtype=float),
    rtol=1e-12,
    atol=1e-12,
    equal_nan=True,
    err_msg="Mau so RMSSE C5/C6 sai lech vuot nguong cho phep.",
    )

    check(
        np.isfinite(
            combined[
                ["RMSSE_C5", "RMSSE_C6"]
            ].to_numpy()
        ).all(),
        "RMSSE khong hop le.",
    )

    combined["delta_RMSSE"] = (
        combined["RMSSE_C6"]
        - combined["RMSSE_C5"]
    )

    combined["C6_improves"] = (
        combined["delta_RMSSE"] < 0
    )

    return combined


# ======================================================
# 6. TONG HOP THEO NHOM
# ======================================================

def group_report(items, columns):

    rows = []

    grouping = (
        columns[0]
        if len(columns) == 1
        else columns
    )

    for key, group in items.groupby(
        grouping,
        observed=True,
        sort=True,
    ):

        if len(columns) == 1:
            key = (key,)

        actual = group["actual_units"].sum()

        p5 = group[
            "forecast_units_C5"
        ].sum()

        p6 = group[
            "forecast_units_C6"
        ].sum()

        row = dict(
            zip(columns, key)
        )

        row.update({
            "n_series": len(group),

            "actual_units": float(actual),

            "forecast_units_C5": float(p5),
            "forecast_units_C6": float(p6),

            "bias_C5_pct": (
                100 * (p5 / actual - 1)
                if actual else np.nan
            ),

            "bias_C6_pct": (
                100 * (p6 / actual - 1)
                if actual else np.nan
            ),

            "RMSSE_C5": float(
                group["RMSSE_C5"].mean()
            ),

            "RMSSE_C6": float(
                group["RMSSE_C6"].mean()
            ),

            "RMSSE_change": float(
                group["delta_RMSSE"].mean()
            ),

            "percent_series_improved": (
                100 * float(
                    group["C6_improves"].mean()
                )
            ),
        })

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================
# 7. CHUONG TRINH CHINH
# ======================================================

def main(requested_batch=None):

    metrics = (
        ROOT / "results" / "metrics"
    )

    c6_batch = find_c6_batch(
        metrics,
        requested_batch,
    )

    c6_dir = metrics / c6_batch

    c6_manifest = json.loads(
        (
            c6_dir / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    check(
        c6_manifest.get("status") == "completed"
        and c6_manifest.get("c5_batch")
        == C5_TAG
        and len(c6_manifest.get("stores", []))
        == N_STORES,
        "C6 chua FULL hoac khong dung C5.",
    )

    c5_manifest = json.loads(
        (
            metrics / C5_TAG / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    c5_sources = {
        entry["store_id"]: entry
        for entry in c5_manifest["sources"]
    }

    stores = sorted(
        c6_manifest["stores"]
    )

    check(
        set(stores) == set(c5_sources),
        "Danh sach cua hang khac nhau.",
    )

    calibration = read_csv(
        c6_dir / "calibration_by_store.csv"
    ).set_index("store_id")

    check(
        set(stores) == set(calibration.index),
        "Thieu he so Platt.",
    )

    frames = []

    # --------------------------------------------------
    # 8. DOC C5 VA C6 TUNG CUA HANG
    # --------------------------------------------------

    for store in stores:

        path5 = (
            ROOT / "results" / "validation"
            / c5_sources[store]["run_tag"]
            / "actual_vs_predictions.csv"
        )

        path6 = (
            ROOT / "results" / "validation"
            / f"{c6_batch}_{store}"
            / "actual_vs_predictions.csv"
        )

        c5 = read_detail(path5, "C5")
        c6 = read_detail(path6, "C6")

        df = c5.merge(
            c6,
            on=KEYS,
            how="outer",
            suffixes=("_C5", "_C6"),
            validate="one_to_one",
            indicator=True,
        )

        check(
            df["_merge"].eq("both").all(),
            f"Khac khoa C5/C6: {store}",
        )

        df = df.drop(columns="_merge")

        np.testing.assert_array_equal(
            df["y_actual_C5"],
            df["y_actual_C6"],
        )

        df["y_actual"] = (
            df["y_actual_C5"]
        )

        df = df.drop(
            columns=[
                "y_actual_C5",
                "y_actual_C6",
            ]
        )

        # Kiem tra p_cal dung he so Platt da hoc.
        p = np.clip(
            df["probability_raw"].to_numpy(),
            1e-6,
            1 - 1e-6,
        )

        cal = calibration.loc[store]

        expected = expit(
            float(cal["platt_slope"])
            * np.log(p / (1 - p))
            + float(cal["platt_intercept"])
        )

        np.testing.assert_allclose(
            expected,
            df["probability_positive_C6"],
            atol=1e-9,
            rtol=1e-9,
        )

        frames.append(df)

        print(
            f"Da kiem tra {store}: "
            f"{len(df):,} du bao"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    check(
        len(data)
        == N_STORES * N_ITEMS_PER_STORE * HORIZON,
        "Tong so du bao khong du.",
    )

    # --------------------------------------------------
    # 9. DOC RMSSE C5/C6
    # --------------------------------------------------

    items = load_item_groups(
        ROOT,
        c6_batch,
        stores,
    )

    item_totals = (
        data.groupby(
            ["store_id", "item_id"],
            sort=False,
        )
        .agg(
            actual_units=("y_actual", "sum"),

            forecast_units_C5=(
                "y_pred_two_part_C5",
                "sum",
            ),

            forecast_units_C6=(
                "y_pred_two_part_C6",
                "sum",
            ),
        )
        .reset_index()
    )

    items = items.merge(
        item_totals,
        on=["store_id", "item_id"],
        how="left",
        validate="one_to_one",
    )

    check(
        items["actual_units"].notna().all(),
        "Thieu du bao theo item.",
    )

    # --------------------------------------------------
    # 10. BAO CAO TONG THE
    # --------------------------------------------------

    overall = pd.DataFrame([
        stats(data)
    ])

    overall["RMSSE_C5"] = (
        items["RMSSE_C5"].mean()
    )

    overall["RMSSE_C6"] = (
        items["RMSSE_C6"].mean()
    )

    reference = read_csv(
        c6_dir / "comparison_overall.csv"
    ).set_index("configuration")

    for label in ["C5", "C6"]:

        np.testing.assert_allclose(
            overall[f"forecast_{label}"],
            reference.loc[label, "prediction"],
            atol=1e-5,
            rtol=0,
        )

        np.testing.assert_allclose(
            overall[f"RMSSE_{label}"],
            reference.loc[
                label,
                "RMSSE_mean_unweighted",
            ],
            atol=1e-9,
            rtol=0,
        )

    # --------------------------------------------------
    # 11. BAO CAO THEO CUA HANG VA HORIZON
    # --------------------------------------------------

    by_store = table_by(
        data,
        ["store_id"],
    )

    by_horizon = table_by(
        data,
        ["horizon"],
    )

    data["week_block"] = (
        (data["horizon"] - 1) // 7 + 1
    )

    by_week = table_by(
        data,
        ["week_block"],
    )

    store_horizon = table_by(
        data,
        ["store_id", "horizon"],
    )

    # --------------------------------------------------
    # 12. HIEU CHUAN XAC SUAT
    # --------------------------------------------------

    reliability_overall = reliability(
        data
    )

    reliability_stores = reliability(
        data,
        ["store_id"],
    )

    for label in [
        "C5_own_path",
        "raw_on_C6_path",
        "cal_on_C6_path",
    ]:

        overall[f"ECE_{label}"] = (
            calculate_ece(
                reliability_overall,
                label,
            )
        )

    # --------------------------------------------------
    # 13. XUAT KET QUA
    # --------------------------------------------------

    reports = {
        "01_overall.csv": overall,

        "02_by_store.csv": by_store,

        "03_by_horizon.csv": by_horizon,

        "04_by_week.csv": by_week,

        "05_by_store_horizon.csv": store_horizon,

        "06_reliability_overall.csv":
            reliability_overall,

        "07_reliability_by_store.csv":
            reliability_stores,

        "08_by_category.csv":
            group_report(items, ["cat_id"]),

        "09_by_sales_pattern.csv":
            group_report(items, ["sales_pattern"]),

        "10_by_category_pattern.csv":
            group_report(
                items,
                ["cat_id", "sales_pattern"],
            ),

        "11_per_item.csv": items,
    }

    output = (
        ROOT / "results" / "analysis"
        / c6_batch
        / "stage_A"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, frame in reports.items():

        frame.to_csv(
            output / filename,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------
    # 14. IN THONG TIN NGAN GON
    # --------------------------------------------------

    print(
        "\n=== C5 VS C6 - TONG QUAN ==="
    )

    columns = [
        "RMSSE_C5",
        "RMSSE_C6",
        "bias_C5_pct",
        "bias_C6_pct",
        "Brier_raw_C6_path",
        "Brier_cal_C6_path",
        "ECE_raw_on_C6_path",
        "ECE_cal_on_C6_path",
    ]

    print(
        overall[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\n=== CUA HANG ==="
    )

    selected = [
        "store_id",
        "bias_C5_pct",
        "bias_C6_pct",
        "p_raw_C6_gap_pp",
        "p_cal_C6_gap_pp",
        "Brier_raw_C6_path",
        "Brier_cal_C6_path",
    ]

    print(
        by_store[selected].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nC6 STAGE A COMPLETE"
    )

    print(
        "Ket qua tai:",
        output,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c6-batch",
        default=None,
        help="Ten batch C6 Full neu co nhieu batch.",
    )

    args = parser.parse_args()

    main(args.c6_batch)
