
"""C7 - Partial Platt Calibration (lambda = 0.50).

Muc tieu:
    So sanh C5, C6 va C7 (lambda=0.50).

Du lieu:
    Origin d_1829: du bao d_1830..d_1857.
    Origin d_1857: du bao d_1858..d_1885.

Nguyen tac:
    - Su dung lai LightGBM cua Stage B.
    - Su dung lai Platt Scaling cua Stage B.
    - Khong train lai.
    - Khong dung validation d_1885 de chon lambda.
    - Khong dung Test.
    - Khong ghi de C7 lambda=0.75.

Cong thuc:
    p_C7 = 0.50 * p_raw + 0.50 * p_platt
    y_C7 = p_C7 * mu

Ket qua:
    results/analysis/stage_C7_lambda_050/
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml
from experiment_common import predict_blended, print_validation_summary

from src.data_loader import (
    check_files,
    load_calendar,
    load_prices,
    load_store,
)

from src.preprocessing import history_matrix


from src.reporting import evaluate_forecast
from run_c7_lambda075 import load_model

from run_c5_price_features import (
    EXTRAS,
    build_weekly_features,
)


from run_stage_b_robustness import (
    H,
    FOLDS,
    STORE_CODES,
    historical_price_arrays,
    read_json,
    require,
    write_json,
)


# ============================================================
# 1. CAU HINH C7
# ============================================================

LAM = 0.50

B_VERSION = "B1_20261121"
C7_VERSION = "C7_050_v1"

OUTPUT_DIR = "stage_C7_lambda_050"


# ============================================================
# 2. NAP MO HINH LIGHTGBM DA TRAIN
# ============================================================


# ============================================================
# 3. DU BAO DE QUY C7
# ============================================================

def predict_c7(bundle, platt, matrix, items, store, state, calendar, origin, prices):
    """Forecast recursively using this entrypoint's frozen lambda."""
    return predict_blended(
        bundle, platt, matrix, items, store, state, calendar, origin, prices,
        LAM, require_finite_mu=True,
    )


# ============================================================
# 4. CHAY MOT CUA HANG TAI MOT GOC
# ============================================================

def run_store(
    store,
    origin,
    data_dir,
    config,
    calendar,
    results,
    stage_b,
    c6_tag,
    output,
):

    source = (
        stage_b
        / "folds"
        / f"d_{origin}"
        / store
    )

    previous = read_json(
        source / "completed.json"
    )

    require(
        previous["version"] == B_VERSION
        and previous["origin"] == origin
        and previous["store_id"] == store,
        f"Stage B fold khong hop le: {store}",
    )

    old = previous["record"]

    folder = (
        output
        / "folds"
        / f"d_{origin}"
        / store
    )

    done = folder / "completed.json"

    # --------------------------------------------
    # 4.1. Dung lai ket qua neu da hoan thanh
    # --------------------------------------------

    if done.is_file():

        saved = read_json(done)

        require(
            saved.get("version") == C7_VERSION
            and saved.get("lambda") == LAM
            and saved.get("origin") == origin
            and saved.get("store_id") == store,
            "Cache C7 khong tuong thich.",
        )

        require(
            all(
                (folder / filename).is_file()
                for filename in [
                    "per_item.csv",
                    "by_week.csv",
                    "forecast_C7.csv",
                ]
            ),
            "Cache C7 thieu file.",
        )

        print_validation_summary(
            saved["record"], f"C7 lambda={LAM:.2f}", origin, store, folder, status="REUSE"
        )

        return (
            saved["record"],
            pd.read_csv(folder / "by_week.csv"),
            pd.read_csv(folder / "per_item.csv"),
        )

    # --------------------------------------------
    # 4.2. Doc du lieu lich su
    # --------------------------------------------

    wide = load_store(
        data_dir,
        store,
        origin,
        chunk_size=config["chunk_size"],
    )

    items = wide.item_id.tolist()

    require(
        len(items) == 3049
        and len(set(items)) == 3049,
        f"{store}: expected 3049 items.",
    )

    state = str(
        wide.state_id.iloc[0]
    )

    matrix = history_matrix(
        wide,
        origin,
    )

    # --------------------------------------------
    # 4.3. Nap LightGBM va Platt
    # --------------------------------------------

    bundle = load_model(
        results,
        stage_b,
        c6_tag,
        store,
        origin,
        items,
    )

    platt_path = (
        source / "platt.joblib"
    )

    require(
        platt_path.is_file(),
        f"Thieu Platt: {platt_path}",
    )

    platt = joblib.load(
        platt_path
    )

    # --------------------------------------------
    # 4.4. Tao dac trung gia
    # --------------------------------------------

    weekly_prices = build_weekly_features(
        load_prices(
            data_dir,
            store,
        ),
        calendar,
    )

    prices = historical_price_arrays(
        weekly_prices,
        calendar,
        items,
        origin,
    )

    del weekly_prices
    gc.collect()

    # --------------------------------------------
    # 4.5. Du bao C7
    # --------------------------------------------

    prediction, components = predict_c7(
        bundle,
        platt,
        matrix,
        items,
        store,
        state,
        calendar,
        origin,
        prices,
    )

    # --------------------------------------------
    # 4.6. Doc nhan thuc te sau du bao
    # --------------------------------------------

    actual = load_store(
        data_dir,
        store,
        origin + H,
        start_day=origin + 1,
        item_ids=items,
        chunk_size=config["chunk_size"],
    )

    require(
        actual.item_id.tolist() == items,
        "Thu tu actual items khong khop.",
    )

    # --------------------------------------------
    # 4.7. Danh gia C7
    # --------------------------------------------

    score, item_scores, truth, yhat = (
        evaluate_forecast(
            prediction,
            actual,
            matrix,
            items,
        )
    )

    # --------------------------------------------
    # 4.8. Doc ket qua C5/C6 cua Stage B
    # --------------------------------------------

    base = (
        pd.read_csv(
            source / "per_item.csv"
        )
        .set_index("item_id")
        .reindex(items)
    )

    current = (
        item_scores
        .set_index("item_id")
        .reindex(items)
    )

    require(
        not base.index.has_duplicates
        and not current.index.has_duplicates,
        "Duplicate item_id.",
    )

    require(
        base["actual_units"].notna().all(),
        "Thieu Stage B per-item.",
    )

    np.testing.assert_allclose(
        base["scale"].to_numpy(dtype=float),
        current["scale"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        base["actual_units"].to_numpy(dtype=float),
        truth.sum(axis=1),
        rtol=0,
        atol=1e-9,
    )

    require(
        int(score["rmsse_valid_items"])
        == int(old["valid_rmsse_items"]),
        "So series co RMSSE hop le khong khop.",
    )

    # --------------------------------------------
    # 4.9. Ket qua theo tung san pham
    # --------------------------------------------

    item_out = pd.DataFrame({
        "origin": origin,
        "store_id": store,
        "item_id": items,

        "RMSSE_C5":
            base["RMSSE_C5"].to_numpy(),

        "RMSSE_C6":
            base["RMSSE_C6"].to_numpy(),

        "RMSSE_C7":
            current["RMSSE"].to_numpy(),

        "actual":
            truth.sum(axis=1),

        "predicted_C5":
            base["predicted_C5"].to_numpy(),

        "predicted_C6":
            base["predicted_C6"].to_numpy(),

        "predicted_C7":
            yhat.sum(axis=1),
    })

    item_out["C7_better_C6"] = (
        item_out["RMSSE_C7"]
        < item_out["RMSSE_C6"]
    )

    # --------------------------------------------
    # 4.10. Ket qua cua hang
    # --------------------------------------------

    total = float(
        truth.sum()
    )

    predicted = float(
        yhat.sum()
    )

    require(
        total > 0,
        "Actual total must be positive.",
    )

    # Component rows theo horizon-major.
    z = (
        (truth > 0)
        .astype(float)
        .T.reshape(-1)
    )

    p = components[
        "p_C7"
    ].to_numpy(dtype=float)

    require(
        len(p) == len(z),
        "Probability and actual count mismatch.",
    )

    np.testing.assert_allclose(
        total,
        old["actual"],
        rtol=0,
        atol=0,
    )

    row = {
        "origin": origin,
        "store_id": store,
        "lambda": LAM,
        "n_items": len(items),

        "valid_rmsse_items":
            int(score["rmsse_valid_items"]),

        "actual": total,

        "predicted_C5":
            float(old["predicted_C5"]),

        "predicted_C6":
            float(old["predicted_C6"]),

        "predicted_C7": predicted,

        "RMSSE_C5":
            float(old["RMSSE_C5"]),

        "RMSSE_C6":
            float(old["RMSSE_C6"]),

        "RMSSE_C7":
            float(score["RMSSE_mean_unweighted"]),

        "MAE_C5":
            float(old["MAE_C5"]),

        "MAE_C6":
            float(old["MAE_C6"]),

        "MAE_C7":
            float(score["MAE"]),

        "bias_C5_pct":
            float(old["bias_C5_pct"]),

        "bias_C6_pct":
            float(old["bias_C6_pct"]),

        "bias_C7_pct":
            100 * (predicted / total - 1),

        "Brier_C7_path":
            float(
                np.mean((p - z) ** 2)
            ),

        "mean_probability_C7":
            float(p.mean()),

        "positive_rate":
            float(z.mean()),

        "n_series_C7_better_C6":
            int(
                item_out["C7_better_C6"].sum()
            ),
    }

    # --------------------------------------------
    # 4.11. Ket qua theo 4 tuan
    # --------------------------------------------

    previous_weeks = (
        pd.read_csv(
            source / "weekly.csv"
        )
        .set_index("week")
    )

    scale = current[
        "scale"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(scale)
        & (scale > 0)
    )

    require(
        valid.any(),
        "No valid RMSSE scales.",
    )

    week_rows = []

    for start in range(0, H, 7):

        week = start // 7 + 1

        a = truth[
            :,
            start:start + 7,
        ]

        f = yhat[
            :,
            start:start + 7,
        ]

        actual_week = float(
            a.sum()
        )

        predicted_week = float(
            f.sum()
        )

        mse = np.mean(
            (a - f) ** 2,
            axis=1,
        )

        rmsse_week = float(
            np.sqrt(
                mse[valid] / scale[valid]
            ).mean()
        )

        week_rows.append({
            "origin": origin,
            "store_id": store,
            "week": week,

            "actual": actual_week,

            "predicted_C5":
                float(
                    previous_weeks.loc[
                        week,
                        "predicted_C5",
                    ]
                ),

            "predicted_C6":
                float(
                    previous_weeks.loc[
                        week,
                        "predicted_C6",
                    ]
                ),

            "predicted_C7": predicted_week,

            "RMSSE_C5":
                float(
                    previous_weeks.loc[
                        week,
                        "RMSSE_C5",
                    ]
                ),

            "RMSSE_C6":
                float(
                    previous_weeks.loc[
                        week,
                        "RMSSE_C6",
                    ]
                ),

            "RMSSE_C7": rmsse_week,

            "MAE_C7":
                float(
                    np.abs(a - f).mean()
                ),

            "bias_C7_pct": (
                100 * (
                    predicted_week
                    / actual_week - 1
                )
                if actual_week > 0
                else np.nan
            ),
        })

    # --------------------------------------------
    # 4.12. Luu ket qua tung fold
    # --------------------------------------------

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    item_out.to_csv(
        folder / "per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        week_rows
    ).to_csv(
        folder / "by_week.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction.to_csv(
        folder / "forecast_C7.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Luu completed cuoi cung.
    write_json(
        done,
        {
            "version": C7_VERSION,
            "lambda": LAM,
            "origin": origin,
            "store_id": store,
            "record": row,
        },
    )

    print_validation_summary(row, f"C7 lambda={LAM:.2f}", origin, store, folder)

    del (
        wide,
        matrix,
        bundle,
        prediction,
        components,
        actual,
        truth,
        yhat,
    )

    gc.collect()

    return (
        row,
        pd.DataFrame(week_rows),
        item_out,
    )


# ============================================================
# 5. TONG HOP KET QUA
# ============================================================

def main(args):

    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "config.yaml"
        ).read_text(encoding="utf-8-sig")
    )

    results = (
        ROOT / config["results_dir"]
    )

    data_dir = (
        ROOT / config["data_dir"]
    )

    stage_b = (
        results
        / "analysis"
        / "stage_B_C5_C6_full"
    )

    # --------------------------------------------
    # 5.1. Kiem tra Stage B
    # --------------------------------------------

    manifest_b = read_json(
        stage_b / "manifest.json"
    )

    require(
        manifest_b["status"] == "complete",
        "Stage B chua hoan thanh.",
    )

    require(
        set(
            manifest_b["origins_requested"]
        ) == set(FOLDS),
        "Stage B khong du 2 historical origins.",
    )

    require(
        set(
            manifest_b["stores"]
        ) == set(STORE_CODES),
        "Stage B khong du 10 stores.",
    )

    c6_tag = manifest_b["C6_batch"]

    # --------------------------------------------
    # 5.2. Chon stores va origins
    # --------------------------------------------

    stores = sorted(
        args.stores or STORE_CODES
    )

    origins = sorted(
        set(args.origins)
    )

    require(
        len(stores) == len(set(stores))
        and bool(stores)
        and set(stores).issubset(STORE_CODES),
        "Danh sach stores khong hop le.",
    )

    require(
        bool(origins)
        and set(origins).issubset(FOLDS),
        "Chi ho tro d_1829 va d_1857.",
    )

    # --------------------------------------------
    # 5.3. Nap du lieu nen
    # --------------------------------------------

    check_files(
        data_dir,
        use_prices=True,
    )

    calendar = load_calendar(
        data_dir
    )

    output = (
        results
        / "analysis"
        / OUTPUT_DIR
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    week_parts = []
    item_parts = []

    # --------------------------------------------
    # 5.4. Chay tung store va origin
    # --------------------------------------------

    for store in stores:

        for origin in origins:

            row, weeks, items = run_store(
                store,
                origin,
                data_dir,
                config,
                calendar,
                results,
                stage_b,
                c6_tag,
                output,
            )

            records.append(row)
            week_parts.append(weeks)
            item_parts.append(items)

    # --------------------------------------------
    # 5.5. Tong hop theo store-origin
    # --------------------------------------------

    by_store = (
        pd.DataFrame(records)
        .sort_values(
            ["origin", "store_id"]
        )
        .reset_index(drop=True)
    )

    by_store["delta_RMSSE_C7_C6"] = (
        by_store["RMSSE_C7"]
        - by_store["RMSSE_C6"]
    )

    by_store["C7_beats_C6"] = (
        by_store["delta_RMSSE_C7_C6"] < 0
    )

    # --------------------------------------------
    # 5.6. Tong hop theo origin
    # --------------------------------------------

    origin_rows = []

    for origin, group in by_store.groupby(
        "origin",
        sort=True,
    ):

        actual = float(
            group["actual"].sum()
        )

        weights = group[
            "valid_rmsse_items"
        ].to_numpy(dtype=float)

        record = {
            "origin": int(origin),
            "actual": actual,
            "n_stores": len(group),
        }

        for name in ["C5", "C6", "C7"]:

            predicted = float(
                group[
                    f"predicted_{name}"
                ].sum()
            )

            record[f"RMSSE_{name}"] = float(
                np.average(
                    group[f"RMSSE_{name}"],
                    weights=weights,
                )
            )

            record[f"MAE_{name}"] = float(
                np.average(
                    group[f"MAE_{name}"],
                    weights=group["n_items"],
                )
            )

            record[f"predicted_{name}"] = predicted

            record[f"bias_{name}_pct"] = (
                100 * (
                    predicted / actual - 1
                )
            )

        origin_rows.append(record)

    by_origin = (
        pd.DataFrame(origin_rows)
        .sort_values("origin")
        .reset_index(drop=True)
    )

    # --------------------------------------------
    # 5.7. Bang tong hop lambda = 0.50
    # --------------------------------------------

    summary_row = {
        "lambda": LAM,
        "origins": len(by_origin),
        "stores": len(stores),
    }

    for name in ["C5", "C6", "C7"]:

        summary_row[
            f"mean_RMSSE_{name}"
        ] = float(
            by_origin[
                f"RMSSE_{name}"
            ].mean()
        )

        summary_row[
            f"mean_MAE_{name}"
        ] = float(
            by_origin[
                f"MAE_{name}"
            ].mean()
        )

        summary_row[
            f"mean_abs_store_bias_{name}_pct"
        ] = float(
            by_store[
                f"bias_{name}_pct"
            ].abs().mean()
        )

    summary_row[
        "C7_beats_C6_origin_count"
    ] = int(
        (
            by_origin["RMSSE_C7"]
            < by_origin["RMSSE_C6"]
        ).sum()
    )

    summary_row[
        "C7_beats_C6_store_origin_count"
    ] = int(
        by_store["C7_beats_C6"].sum()
    )

    summary = pd.DataFrame(
        [summary_row]
    )

    # --------------------------------------------
    # 5.8. Luu cac bang
    # --------------------------------------------

    by_origin.to_csv(
        output / "01_by_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_store.to_csv(
        output / "02_by_store_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        output / "03_lambda_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        week_parts,
        ignore_index=True,
    ).to_csv(
        output / "04_by_week_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        item_parts,
        ignore_index=True,
    ).to_csv(
        output / "05_per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------
    # 5.9. Manifest
    # --------------------------------------------

    status = (
        "complete"
        if (
            set(stores) == set(STORE_CODES)
            and set(origins) == set(FOLDS)
        )
        else "partial"
    )

    write_json(
        output / "manifest.json",
        {
            "status": status,
            "version": C7_VERSION,
            "lambda": LAM,
            "origins": origins,
            "stores": stores,

            "stage_b_source":
                "stage_B_C5_C6_full",

            "C6_batch": c6_tag,

            "lightgbm_retrained": False,
            "platt_retrained": False,

            "d_1885_not_used_for_lambda_selection":
                True,

            "test_performed": False,
        },
    )

    # --------------------------------------------
    # 5.10. In bao cao ngan gon
    # --------------------------------------------

    print(
        "\n=== C5 vs C6 vs C7 "
        f"(lambda={LAM:.2f}) ==="
    )

    print(
        by_origin[
            [
                "origin",
                "RMSSE_C5",
                "RMSSE_C6",
                "RMSSE_C7",
                "MAE_C5",
                "MAE_C6",
                "MAE_C7",
                "bias_C5_pct",
                "bias_C6_pct",
                "bias_C7_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== LAMBDA SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nKet qua duoc luu tai: {output}"
    )


# ============================================================
# 6. DIEM CHAY CHUONG TRINH
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stores",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--origins",
        nargs="+",
        type=int,
        default=[1829, 1857],
    )

    args = parser.parse_args()

    main(args)
