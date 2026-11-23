
"""C7 Validation - d_1886 to d_1913.

Origin: d_1885
Lambda: 0.75

Reuse:
- C5 LightGBM main model trained through d_1885.
- Original C6 Platt coefficients.
- Existing C5/C6 validation metrics from Stage B.

No retraining.
No Test.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from experiment_common import csv_save as save_csv, print_validation_summary

from scipy.special import expit

from src.data_loader import (
    check_files,
    load_calendar,
    load_prices,
    load_store,
)

from src.preprocessing import history_matrix
from src.reporting import evaluate_forecast
from src.training import verify_saved_models

from run_c5_price_features import (
    EXTRAS,
    build_weekly_features,
)

from run_stage_b_robustness import (
    H,
    STORE_CODES,
    historical_price_arrays,
    read_json,
    require,
    write_json,
)

from run_c7_lambda075 import (
    LAM,
    predict_c7,
)


# ======================================================
# 1. CONFIGURATION
# ======================================================

ORIGIN = 1885
VERSION = "C7_VALIDATION_1885_L075_V1"

OUTPUT_DIR = "stage_C7_validation_d1885_lambda075"


# ======================================================
# 2. RECONSTRUCT ORIGINAL C6 PLATT
# ======================================================


class FrozenPlatt:

    def __init__(self, slope, intercept):

        self.slope = float(slope)
        self.intercept = float(intercept)

        self.coef_ = np.array([
            [self.slope]
        ])

        self.intercept_ = np.array([
            self.intercept
        ])

        self.classes_ = np.array([0, 1])

        require(
            np.isfinite([
                self.slope,
                self.intercept,
            ]).all(),
            "Invalid Platt coefficients.",
        )

        require(
            self.slope > 0,
            "Nonpositive Platt slope.",
        )

    def predict_proba(self, x):

        x = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        p = expit(
            self.slope * x
            + self.intercept
        )

        return np.column_stack([
            1.0 - p,
            p,
        ])


# ======================================================
# 3. VALIDATE ONE STORE
# ======================================================

def run_one(
    store,
    *,
    data_dir,
    results,
    calendar,
    config,
    c5_manifest,
    cal_table,
    reference,
    output,
):

    folder = (
        output
        / "folds"
        / store
    )

    completed = (
        folder / "completed.json"
    )

    # --------------------------------------------------
    # 3.1. REUSE COMPLETED C7 VALIDATION
    # --------------------------------------------------

    if completed.is_file():

        obj = read_json(completed)

        require(
            obj.get("version") == VERSION
            and obj.get("store_id") == store,
            f"Invalid cache: {store}",
        )

        for name in [
            "forecast_C7.csv",
            "per_item_C7.csv",
            "by_week.csv",
        ]:
            require(
                (folder / name).is_file(),
                f"Missing cached file: {name}",
            )

        print_validation_summary(
            obj["record"], "C7 lambda=0.75", ORIGIN, store, folder, status="REUSE"
        )

        return (
            obj["record"],
            pd.read_csv(
                folder / "by_week.csv"
            ),
            pd.read_csv(
                folder / "per_item_C7.csv"
            ),
        )

    # --------------------------------------------------
    # 3.2. LOAD HISTORY THROUGH d_1885
    # --------------------------------------------------

    wide = load_store(
        data_dir,
        store,
        ORIGIN,
        chunk_size=config["chunk_size"],
    )

    items = wide.item_id.to_list()

    require(
        len(items) == 3049
        and len(set(items)) == 3049,
        f"Expected 3049 series: {store}",
    )

    state = str(
        wide.state_id.iloc[0]
    )

    matrix = history_matrix(
        wide,
        ORIGIN,
    )

    # --------------------------------------------------
    # 3.3. LOAD TRAINED C5 LIGHTGBM
    # --------------------------------------------------

    sources = [
        source
        for source in c5_manifest["sources"]
        if source["store_id"] == store
    ]

    require(
        len(sources) == 1,
        f"Missing C5 model: {store}",
    )

    main_path = (
        results
        / "models"
        / sources[0]["run_tag"]
        / "model.joblib"
    )

    require(
        main_path.is_file(),
        f"Missing model: {main_path}",
    )

    bundle = verify_saved_models(
        main_path
    )

    require(
        bundle["origin_day"] == ORIGIN
        and bundle["categories"] == items,
        "Wrong training cutoff or item order.",
    )

    require(
        all(
            name in bundle["features"]
            for name in [
                "sell_price",
                *EXTRAS,
            ]
        ),
        "Missing price features.",
    )

    # --------------------------------------------------
    # 3.4. BUILD CAUSAL PRICE FEATURES
    # --------------------------------------------------

    weekly = build_weekly_features(
        load_prices(
            data_dir,
            store,
        ),
        calendar,
    )

    prices = historical_price_arrays(
        weekly,
        calendar,
        items,
        ORIGIN,
    )

    del weekly
    gc.collect()

    # --------------------------------------------------
    # 3.5. RESTORE C6 PLATT COEFFICIENTS
    # --------------------------------------------------

    cr = cal_table.loc[
        cal_table["store_id"].eq(store)
    ]

    require(
        len(cr) == 1,
        f"Missing Platt: {store}",
    )

    c = cr.iloc[0]

    require(
        int(c["calibration_train_end"]) == 1857,
        "Incorrect C6 calibration cutoff.",
    )

    platt = FrozenPlatt(
        c["platt_slope"],
        c["platt_intercept"],
    )

    # --------------------------------------------------
    # 3.6. RECURSIVE C7 FORECAST
    # --------------------------------------------------

    # Uses lambda=0.75 from run_c7_lambda075.
    # Every next day uses previous C7 forecasts.

    forecast, comps = predict_c7(
        bundle,
        platt,
        matrix,
        items,
        store,
        state,
        calendar,
        ORIGIN,
        prices,
    )

    # --------------------------------------------------
    # 3.7. LOAD VALIDATION ACTUALS AFTER FORECAST
    # --------------------------------------------------

    actual = load_store(
        data_dir,
        store,
        ORIGIN + H,
        start_day=ORIGIN + 1,
        item_ids=items,
        chunk_size=config["chunk_size"],
    )

    require(
        actual.item_id.to_list() == items,
        "Validation item order mismatch.",
    )

    # --------------------------------------------------
    # 3.8. CALCULATE RMSSE / MAE
    # --------------------------------------------------

    score, item_scores, truth, yhat = (
        evaluate_forecast(
            forecast,
            actual,
            matrix,
            items,
        )
    )

    require(
        item_scores["item_id"].is_unique,
        "Duplicate item metrics.",
    )

    item_scores = (
        item_scores
        .set_index("item_id")
        .reindex(items)
        .reset_index()
    )

    # --------------------------------------------------
    # 3.9. VERIFY SAME ACTUALS AS C6
    # --------------------------------------------------

    ref = reference.loc[
        reference["store_id"].eq(store)
    ]

    require(
        len(ref) == 1,
        f"Missing Stage B metrics: {store}",
    )

    ref = ref.iloc[0]

    np.testing.assert_allclose(
        truth.sum(),
        float(ref["actual"]),
        rtol=0,
        atol=0,
    )

    require(
        int(score["rmsse_valid_items"])
        == int(ref["valid_rmsse_items"]),
        "RMSSE population mismatch.",
    )

    # --------------------------------------------------
    # 3.10. STORE METRICS
    # --------------------------------------------------

    total = float(truth.sum())
    total_hat = float(yhat.sum())

    require(
        total > 0,
        "Validation actual total must be positive.",
    )

    p = comps[
        "p_C7"
    ].to_numpy(dtype=float)

    z = (
        (truth > 0)
        .astype(float)
        .T.reshape(-1)
    )

    require(
        len(p) == len(z) == len(items) * H,
        "Invalid probability count.",
    )

    record = {
        "origin": ORIGIN,
        "store_id": store,
        "n_items": len(items),

        "valid_rmsse_items":
            int(score["rmsse_valid_items"]),

        "actual": total,

        "predicted_C5":
            float(ref["predicted_C5"]),

        "predicted_C6":
            float(ref["predicted_C6"]),

        "predicted_C7": total_hat,

        "RMSSE_C5":
            float(ref["RMSSE_C5"]),

        "RMSSE_C6":
            float(ref["RMSSE_C6"]),

        "RMSSE_C7":
            float(score["RMSSE_mean_unweighted"]),

        "MAE_C5":
            float(ref["MAE_C5"]),

        "MAE_C6":
            float(ref["MAE_C6"]),

        "MAE_C7":
            float(score["MAE"]),

        "bias_C5_pct":
            float(ref["bias_C5_pct"]),

        "bias_C6_pct":
            float(ref["bias_C6_pct"]),

        "bias_C7_pct":
            100 * (
                total_hat / total - 1.0
            ),

        "Brier_C7_path":
            float(np.mean((p - z) ** 2)),

        "positive_rate":
            float(z.mean()),

        "mean_probability_C7":
            float(p.mean()),

        "platt_slope":
            platt.slope,

        "platt_intercept":
            platt.intercept,

        "missing_origin_sell_price":
            int(
                np.isnan(
                    prices["sell_price"]
                ).sum()
            ),
    }

    # --------------------------------------------------
    # 3.11. PER-ITEM METRICS
    # --------------------------------------------------

    per_item = item_scores[
        [
            "item_id",
            "RMSSE",
            "scale",
        ]
    ].copy()

    per_item.rename(
        columns={
            "RMSSE": "RMSSE_C7",
        },
        inplace=True,
    )

    per_item.insert(
        0,
        "store_id",
        store,
    )

    per_item.insert(
        0,
        "origin",
        ORIGIN,
    )

    per_item["actual_units"] = (
        truth.sum(axis=1)
    )

    per_item["predicted_C7"] = (
        yhat.sum(axis=1)
    )

    # --------------------------------------------------
    # 3.12. WEEKLY METRICS
    # --------------------------------------------------

    scale = item_scores[
        "scale"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(scale)
        & (scale > 0)
    )

    by_week = []

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

        a_sum = float(a.sum())
        f_sum = float(f.sum())

        mse = np.mean(
            (a - f) ** 2,
            axis=1,
        )

        rmsse = float(
            np.sqrt(
                mse[valid] / scale[valid]
            ).mean()
        )

        by_week.append({
            "origin": ORIGIN,
            "store_id": store,
            "week": week,

            "actual": a_sum,
            "predicted_C7": f_sum,

            "RMSSE_C7": rmsse,

            "MAE_C7":
                float(np.abs(a - f).mean()),

            "bias_C7_pct": (
                100 * (f_sum / a_sum - 1)
                if a_sum > 0
                else np.nan
            ),
        })

    # --------------------------------------------------
    # 3.13. SAVE FOLD
    # --------------------------------------------------

    save_csv(
        forecast,
        folder / "forecast_C7.csv",
    )

    save_csv(
        per_item,
        folder / "per_item_C7.csv",
    )

    save_csv(
        pd.DataFrame(by_week),
        folder / "by_week.csv",
    )

    write_json(
        completed,
        {
            "version": VERSION,
            "store_id": store,
            "origin": ORIGIN,
            "record": record,
        },
    )

    print_validation_summary(record, "C7 lambda=0.75", ORIGIN, store, folder)

    del (
        wide,
        matrix,
        bundle,
        forecast,
        comps,
        actual,
        truth,
        yhat,
    )

    gc.collect()

    return (
        record,
        pd.DataFrame(by_week),
        per_item,
    )


# ======================================================
# 4. MAIN
# ======================================================

def main(args):

    require(
        abs(LAM - 0.75) < 1e-12,
        "Expected C7 lambda = 0.75.",
    )

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

    manifest_b = read_json(
        stage_b / "manifest.json"
    )

    require(
        manifest_b["status"] == "complete",
        "Stage B is incomplete.",
    )

    require(
        manifest_b["origins_in_summary"]
        == [1829, 1857, 1885],
        "Stage B must include d_1885.",
    )

    c5_tag = manifest_b["C5_batch"]
    c6_tag = manifest_b["C6_batch"]

    # --------------------------------------------------
    # 4.1. VALIDATE EXISTING C5/C6
    # --------------------------------------------------

    c5_manifest = read_json(
        results
        / "metrics"
        / c5_tag
        / "manifest.json"
    )

    c6_manifest = read_json(
        results
        / "metrics"
        / c6_tag
        / "manifest.json"
    )

    require(
        c5_manifest["status"] == "completed"
        and c5_manifest["origin_day"] == ORIGIN,
        "Invalid C5 validation source.",
    )

    require(
        c5_manifest.get("price_policy")
        == "previous_completed_week",
        "Incorrect price policy.",
    )

    require(
        c6_manifest["status"] == "completed"
        and c6_manifest["validation_origin"] == ORIGIN
        and c6_manifest["calibration_train_end"] == 1857
        and c6_manifest[
            "validation_not_used_for_calibration"
        ] is True,
        "Invalid C6 calibration period.",
    )

    # --------------------------------------------------
    # 4.2. STORES
    # --------------------------------------------------

    stores = sorted(
        args.stores or STORE_CODES
    )

    require(
        stores
        and len(stores) == len(set(stores))
        and set(stores).issubset(STORE_CODES),
        "Invalid stores.",
    )

    # --------------------------------------------------
    # 4.3. EXISTING VALIDATION REFERENCE
    # --------------------------------------------------

    base = pd.read_csv(
        stage_b
        / "02_by_store_origin.csv"
    )

    base = base.loc[
        base["origin"].eq(ORIGIN)
        & base["store_id"].isin(stores)
    ]

    require(
        len(base) == len(stores),
        "Missing C5/C6 validation metrics.",
    )

    cal_table = pd.read_csv(
        results
        / "metrics"
        / c6_tag
        / "calibration_by_store.csv"
    )

    require(
        cal_table["store_id"].is_unique,
        "Duplicate Platt records.",
    )

    # --------------------------------------------------
    # 4.4. LOAD DATA AND RUN
    # --------------------------------------------------

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

    records = []
    weeks = []
    items = []

    for store in stores:

        r, w, i = run_one(
            store,
            data_dir=data_dir,
            results=results,
            calendar=calendar,
            config=config,
            c5_manifest=c5_manifest,
            cal_table=cal_table,
            reference=base,
            output=output,
        )

        records.append(r)
        weeks.append(w)
        items.append(i)

    # --------------------------------------------------
    # 4.5. AGGREGATE ALL STORES
    # --------------------------------------------------

    by_store = (
        pd.DataFrame(records)
        .sort_values("store_id")
        .reset_index(drop=True)
    )

    actual_total = float(
        by_store["actual"].sum()
    )

    overall = {
        "origin": ORIGIN,
        "n_stores": len(stores),
        "actual": actual_total,
    }

    for name in [
        "C5",
        "C6",
        "C7",
    ]:

        units = float(
            by_store[
                f"predicted_{name}"
            ].sum()
        )

        overall[
            f"predicted_{name}"
        ] = units

        overall[
            f"bias_{name}_pct"
        ] = 100 * (
            units / actual_total - 1
        )

        overall[
            f"RMSSE_{name}"
        ] = float(
            np.average(
                by_store[
                    f"RMSSE_{name}"
                ],
                weights=by_store[
                    "valid_rmsse_items"
                ],
            )
        )

        overall[
            f"MAE_{name}"
        ] = float(
            np.average(
                by_store[
                    f"MAE_{name}"
                ],
                weights=by_store[
                    "n_items"
                ],
            )
        )

    # --------------------------------------------------
    # 4.6. SAVE REPORTS
    # --------------------------------------------------

    save_csv(
        pd.DataFrame([overall]),
        output / "01_by_origin.csv",
    )

    save_csv(
        by_store,
        output / "02_by_store.csv",
    )

    save_csv(
        pd.concat(
            weeks,
            ignore_index=True,
        ),
        output / "03_by_week_C7.csv",
    )

    save_csv(
        pd.concat(
            items,
            ignore_index=True,
        ),
        output / "04_per_item_C7.csv",
    )

    status = (
        "complete"
        if set(stores) == set(STORE_CODES)
        else "partial"
    )

    write_json(
        output / "manifest.json",
        {
            "status": status,
            "version": VERSION,

            "origin": ORIGIN,
            "horizon": [1886, 1913],

            "stores": stores,
            "lambda": LAM,

            "C5_batch": c5_tag,
            "C6_batch": c6_tag,

            "C5_C6_reused_from_StageB": True,

            "main_LightGBM_retrained": False,
            "Platt_retrained": False,

            "price_policy":
                "previous_completed_week",

            "test_performed": False,
        },
    )

    # --------------------------------------------------
    # 4.7. TERMINAL SUMMARY
    # --------------------------------------------------

    print(
        "\n=== VALIDATION d_1886 - d_1913 ==="
    )

    print(
        pd.DataFrame([overall])[
            [
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
        ].to_string(index=False)
    )

    print(
        f"\nResults saved: {output}"
    )


# ======================================================
# 5. EXECUTE
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stores",
        nargs="+",
        default=None,
    )

    main(
        parser.parse_args()
    )
