
"""Final frozen C7 (lambda=0.75) Test on M5 d_1914..d_1941.

Prerequisite: scripts/run_c6_final_test.py has completed all ten stores.
Uses the *same* d_1913 LightGBM and Platt as C6, rerunning a separate
28-step recursive C7 forecast. Test labels used for scoring only.
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
from experiment_common import csv_save, print_validation_summary

from src.data_loader import check_files, load_calendar, load_prices, load_store
from src.preprocessing import history_matrix
from src.reporting import evaluate_forecast
from src.training import verify_saved_models
from src.forecasting import validate_forecasts
from run_c5_price_features import EXTRAS, build_weekly_features
from run_c7_lambda075 import LAM, predict_c7
from run_stage_b_robustness import (
    H, STORE_CODES, historical_price_arrays, read_json, require, write_json,
)

ORIGIN = 1913
VERSION = "FINAL_TEST_C7_L075_FROZEN_V1"
C6_VERSION = "FINAL_TEST_C6_FROZEN_V1"


def run_store(store: str, config: dict, calendar: pd.DataFrame, base: Path):
    c6_folder = base / "folds" / store
    c7_folder = base / "C7" / "folds" / store
    done = c7_folder / "completed.json"

    c6_fold = read_json(c6_folder / "completed.json")
    require(
        c6_fold.get("version") == C6_VERSION
        and c6_fold.get("origin") == ORIGIN
        and c6_fold.get("store_id") == store,
        f"Invalid C6 Test fold for {store}",
    )
    c6 = c6_fold["record"]

    if done.is_file():
        saved = read_json(done)
        require(
            saved.get("version") == VERSION
            and saved.get("store_id") == store
            and saved.get("origin") == ORIGIN
            and abs(float(saved.get("lambda", -1)) - LAM) < 1e-12,
            f"Stale C7 cache: {store}",
        )
        for name in ("forecast_C7.csv", "per_item.csv", "by_week.csv"):
            require((c7_folder / name).is_file(), f"Missing C7 cache {name}")
        print_validation_summary(
            saved["record"], "C7 lambda=0.75", ORIGIN, store, c7_folder, status="REUSE"
        )
        return (
            saved["record"],
            pd.read_csv(c7_folder / "by_week.csv"),
            pd.read_csv(c7_folder / "per_item.csv"),
        )

    data_dir = ROOT / config["data_dir"]
    wide = load_store(
        data_dir, store, ORIGIN,
        chunk_size=config["chunk_size"],
    )
    items = wide.item_id.tolist()
    require(len(items) == 3049 and len(set(items)) == 3049, f"Wrong items: {store}")
    require(
        all(f"d_{d}" in wide.columns for d in range(1, ORIGIN + 1)),
        "Missing historical sales columns",
    )
    require(
        not any(f"d_{d}" in wide.columns for d in range(ORIGIN + 1, ORIGIN + H + 1)),
        "Test actuals were loaded into training history",
    )
    state = str(wide.state_id.iloc[0])
    history = history_matrix(wide, ORIGIN)

    # Exact main model and calibrator that C6 used at this origin.
    model_path = base / "models" / store / "model.joblib"
    platt_path = c6_folder / "platt.joblib"
    require(model_path.is_file() and platt_path.is_file(), f"C6 model missing: {store}")
    bundle = verify_saved_models(model_path)
    require(bundle["origin_day"] == ORIGIN, f"Incorrect model cutoff {store}")
    require(bundle["categories"] == items, f"Incorrect item order {store}")
    require(
        all(feature in bundle["features"] for feature in ("sell_price", *EXTRAS)),
        f"Missing C5/C6 price features {store}",
    )
    platt = joblib.load(platt_path)
    require(hasattr(platt, "predict_proba"), f"Invalid Platt model {store}")

    weekly_prices = build_weekly_features(load_prices(data_dir, store), calendar)
    prices = historical_price_arrays(weekly_prices, calendar, items, ORIGIN)
    missing_price = int(np.isnan(prices["sell_price"]).sum())
    require(missing_price == int(c6["missing_origin_sell_price"]), "Price mismatch C6/C7")
    del weekly_prices
    gc.collect()

    # Recursive C7: own lag/rolling updates, not post-processed C6 forecasts.
    forecast, components = predict_c7(
        bundle, platt, history, items, store, state,
        calendar, ORIGIN, prices,
    )
    forecast["split"] = "test"
    validate_forecasts(forecast)
    expected_days = {f"d_{d}" for d in range(ORIGIN + 1, ORIGIN + H + 1)}
    require(
        len(forecast) == len(items) * H
        and set(forecast["target_day"]) == expected_days
        and (forecast["origin_day"] == f"d_{ORIGIN}").all()
        and (forecast["store_id"] == store).all(),
        f"Invalid horizon or forecast keys: {store}",
    )
    require(np.isfinite(forecast["y_pred"]).all(), "Invalid C7 forecast values")

    # Write forecasts before accessing test targets.
    csv_save(forecast, c7_folder / "forecast_C7.csv")

    actual = load_store(
        data_dir, store, ORIGIN + H,
        start_day=ORIGIN + 1, item_ids=items,
        chunk_size=config["chunk_size"],
    )
    require(actual.item_id.tolist() == items, "Test item order mismatch")
    score, item_scores, truth, yhat = evaluate_forecast(
        forecast, actual, history, items,
    )

    # Verify paired scoring and identical RMSSE scales.
    ref_items = pd.read_csv(c6_folder / "per_item.csv").set_index("item_id").reindex(items)
    new_items = item_scores.set_index("item_id").reindex(items)
    require(not ref_items.index.has_duplicates and not new_items.index.has_duplicates,
            "Duplicate item_id in scoring")
    np.testing.assert_allclose(
        ref_items["actual_units"].to_numpy(), truth.sum(axis=1), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        ref_items["scale"].to_numpy(), new_items["scale"].to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    np.testing.assert_allclose(truth.sum(), c6["actual"], rtol=0, atol=0)
    require(int(score["rmsse_valid_items"]) == int(c6["valid_rmsse_items"]),
            "RMSSE series population changed")

    y_total = float(truth.sum())
    p_total = float(yhat.sum())
    z = (truth > 0).astype(float).T.reshape(-1)
    p_raw = components["p_raw"].to_numpy(dtype=float)
    p_c7 = components["p_C7"].to_numpy(dtype=float)
    require(len(z) == len(p_raw) == len(p_c7), "Invalid probability shapes")

    row = {
        "store_id": store, "origin": ORIGIN,
        "n_items": len(items), "valid_rmsse_items": int(score["rmsse_valid_items"]),
        "actual": y_total,
        "predicted_C6": float(c6["predicted_C6"]), "predicted_C7": p_total,
        "RMSSE_C6": float(c6["RMSSE_C6"]),
        "RMSSE_C7": float(score["RMSSE_mean_unweighted"]),
        "MAE_C6": float(c6["MAE_C6"]), "MAE_C7": float(score["MAE"]),
        "bias_C6_pct": float(c6["bias_C6_pct"]),
        "bias_C7_pct": 100 * (p_total / y_total - 1),
        "Brier_raw_C7_path": float(np.mean((p_raw - z) ** 2)),
        "Brier_C7_path": float(np.mean((p_c7 - z) ** 2)),
        "positive_rate": float(z.mean()),
        "mean_p_C7": float(p_c7.mean()),
        "missing_origin_sell_price": missing_price,
    }

    per_item = pd.DataFrame({
        "store_id": store, "item_id": items,
        "RMSSE_C6": ref_items["RMSSE_C6"].to_numpy(),
        "RMSSE_C7": new_items["RMSSE"].to_numpy(),
        "scale": new_items["scale"].to_numpy(),
        "actual_units": truth.sum(axis=1),
        "predicted_C6": ref_items["predicted_C6"].to_numpy(),
        "predicted_C7": yhat.sum(axis=1),
    })
    per_item["C7_better_RMSSE"] = per_item.RMSSE_C7 < per_item.RMSSE_C6
    row["n_series_C7_better_RMSSE"] = int(per_item.C7_better_RMSSE.sum())

    old_weeks = pd.read_csv(c6_folder / "by_week.csv").set_index("week")
    scale = new_items["scale"].to_numpy(dtype=float)
    valid = np.isfinite(scale) & (scale > 0)
    require(valid.any(), "No valid RMSSE scales")
    week_rows = []
    for start in range(0, H, 7):
        week = start // 7 + 1
        a = truth[:, start:start + 7]
        f = yhat[:, start:start + 7]
        units = float(a.sum())
        pred_units = float(f.sum())
        np.testing.assert_allclose(units, old_weeks.loc[week, "actual"], rtol=0, atol=0)
        rmsse = float(np.sqrt(np.mean((a - f) ** 2, axis=1)[valid] / scale[valid]).mean())
        week_rows.append({
            "store_id": store, "week": week,
            "start_day": ORIGIN + start + 1,
            "end_day": ORIGIN + start + 7,
            "actual": units,
            "predicted_C6": float(old_weeks.loc[week, "predicted_C6"]),
            "predicted_C7": pred_units,
            "RMSSE_C6": float(old_weeks.loc[week, "RMSSE_C6"]),
            "RMSSE_C7": rmsse,
            "MAE_C6": float(old_weeks.loc[week, "MAE_C6"]),
            "MAE_C7": float(np.mean(np.abs(a - f))),
            "bias_C6_pct": float(old_weeks.loc[week, "bias_C6_pct"]),
            "bias_C7_pct": 100 * (pred_units / units - 1) if units > 0 else np.nan,
        })

    csv_save(per_item, c7_folder / "per_item.csv")
    csv_save(pd.DataFrame(week_rows), c7_folder / "by_week.csv")
    write_json(done, {
        "version": VERSION, "store_id": store,
        "origin": ORIGIN, "lambda": LAM,
        "record": row,
    })
    print_validation_summary(row, f"C7 lambda={LAM:.2f}", ORIGIN, store, c7_folder)
    del wide, history, bundle, forecast, components, actual, truth, yhat
    gc.collect()
    return row, pd.DataFrame(week_rows), per_item


def main(args):
    require(abs(LAM - 0.75) < 1e-12, "Final C7 lambda must remain 0.75")
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8-sig"))
    results = ROOT / config["results_dir"]
    data_dir = ROOT / config["data_dir"]
    base = results / "analysis" / "final_test_C6_C7_frozen"
    c6_manifest = read_json(base / "manifest_C6.json")
    require(
        c6_manifest.get("status") == "complete"
        and c6_manifest.get("version") == C6_VERSION
        and c6_manifest.get("origin") == ORIGIN
        and c6_manifest.get("test_days") == [1914, 1941]
        and c6_manifest.get("calibration_window") == [1886, 1913]
        and c6_manifest.get("main_training_cutoff") == ORIGIN
        and c6_manifest.get("price_policy") == "previous_completed_week"
        and c6_manifest.get("lightgbm_hyperparams_frozen") is True
        and set(c6_manifest.get("stores", [])) == set(STORE_CODES),
        "Complete frozen C6 Test is required before C7",
    )
    require(
        config["test_origin"] == ORIGIN
        and config["validation_origin"] == 1885,
        "Incorrect origins in config.yaml",
    )
    stores = sorted(args.stores or STORE_CODES)
    require(stores and len(stores) == len(set(stores)) and set(stores).issubset(STORE_CODES),
            "Invalid stores")
    check_files(data_dir, use_prices=True)
    calendar = load_calendar(data_dir)
    require(
        {f"d_{x}" for x in range(1914, 1942)}.issubset(set(calendar["d"])),
        "Missing Test calendar days",
    )

    store_rows, weekly_tables, item_tables = [], [], []
    for store in stores:
        row, weekly, item = run_store(store, config, calendar, base)
        store_rows.append(row)
        weekly_tables.append(weekly)
        item_tables.append(item)

    by_store = pd.DataFrame(store_rows).sort_values("store_id").reset_index(drop=True)
    by_week = pd.concat(weekly_tables, ignore_index=True)
    by_item = pd.concat(item_tables, ignore_index=True)
    weights = by_store["valid_rmsse_items"].to_numpy(dtype=float)
    total = float(by_store["actual"].sum())
    summary = {
        "origin": ORIGIN, "start_day": 1914, "end_day": 1941,
        "n_stores": len(stores), "n_items": int(by_store["n_items"].sum()),
        "valid_rmsse_items": int(weights.sum()), "actual": total,
    }
    for m in ("C6", "C7"):
        pred = float(by_store[f"predicted_{m}"].sum())
        summary[f"predicted_{m}"] = pred
        summary[f"bias_{m}_pct"] = 100 * (pred / total - 1)
        summary[f"RMSSE_{m}"] = float(np.average(by_store[f"RMSSE_{m}"], weights=weights))
        summary[f"MAE_{m}"] = float(np.average(by_store[f"MAE_{m}"], weights=by_store.n_items))
        summary[f"mean_abs_store_bias_{m}_pct"] = float(by_store[f"bias_{m}_pct"].abs().mean())
    summary["n_store_C7_better_RMSSE"] = int((by_store.RMSSE_C7 < by_store.RMSSE_C6).sum())
    summary["n_store_C7_better_abs_bias"] = int((by_store.bias_C7_pct.abs() < by_store.bias_C6_pct.abs()).sum())
    summary["n_item_C7_better_RMSSE"] = int(by_item.C7_better_RMSSE.sum())
    summary["lambda_C7"] = LAM

    reports = base / "comparison_C6_C7"
    csv_save(pd.DataFrame([summary]), reports / "01_by_origin.csv")
    csv_save(by_store, reports / "02_by_store.csv")
    csv_save(by_week, reports / "03_by_week.csv")
    csv_save(by_item, reports / "04_per_item.csv")
    status = "complete" if set(stores) == set(STORE_CODES) else "partial"
    write_json(reports / "manifest_C6_C7.json", {
        "status": status, "version": VERSION, "origin": ORIGIN,
        "test_days": [1914, 1941], "stores": stores, "lambda_C7": LAM,
        "C6_source": "final_test_C6_C7_frozen",
        "reuse_C6_main_model_and_platt": True,
        "lightgbm_retrained_for_C7": False,
        "platt_retrained_for_C7": False,
        "recursive_forecasts_separate": True,
        "price_policy": "previous_completed_week",
    })
    print("\n=== FINAL TEST: C6 vs C7 (lambda=0.75) ===")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved reports to: {reports}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", nargs="+", default=None)
    main(parser.parse_args())
