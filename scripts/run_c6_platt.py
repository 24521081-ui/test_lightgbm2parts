
"""C6 - LightGBM Two-Part + Platt Scaling.

Calibration:
    Train auxiliary model through d_1857.
    Generate out-of-sample predictions d_1858-d_1885.
    Fit Platt Scaling on historical predictions.

Validation:
    Reuse saved C5 model.
    Predict recursively d_1886-d_1913 using p_cal * mu.

No validation information is used to fit Platt Scaling.
"""

import argparse
import gc
import logging
import sys

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression

from src.data_loader import (
    check_files,
    load_calendar,
    load_prices,
    load_store,
)

from src.feature_engineering import next_day_features

from src.forecasting import (
    FORECAST_COLUMNS,
    export_forecasts,
    validate_forecasts,
)

from src.preprocessing import (
    history_matrix,
    prepare_history,
)

from src.reporting import (
    evaluate_and_export,
    save_json,
    training_patterns,
)

from src.training import (
    train_model,
    verify_saved_models,
)

from run_c5_price_features import (
    EXTRAS,
    KEYS,
    attach_price_features,
    build_weekly_features,
    forecast_c5,
    make_training_data,
)

from run_c4_with_prices_lagged import (
    check_join,
    diagnostics,
    read_json,
    require,
)


LOG = logging.getLogger("m5")

C5_TAG = (
    "validation_C5_full_10_stores_"
    "20261121T140034335643Z"
)

CAL_ORIGIN = 1857
VALID_ORIGIN = 1885
HORIZON = 28

PRICE_FEATURES = ["sell_price"] + EXTRAS


# ==========================================================
# 1. HAM HIEU CHUAN XAC SUAT
# ==========================================================

def logit_probability(values):

    p = np.asarray(
        values,
        dtype=np.float64,
    )

    require(
        np.isfinite(p).all()
        and ((p >= 0) & (p <= 1)).all(),
        "Classifier probability invalid.",
    )

    p = np.clip(
        p,
        1e-6,
        1 - 1e-6,
    )

    return np.log(
        p / (1 - p)
    ).reshape(-1, 1)


def apply_platt(calibrator, p):

    calibrated = calibrator.predict_proba(
        logit_probability(p)
    )[:, 1]

    require(
        np.isfinite(calibrated).all()
        and (calibrated >= 0).all()
        and (calibrated <= 1).all(),
        "Calibrated probability invalid.",
    )

    return calibrated


# ==========================================================
# 2. CHUAN BI DAC TRUNG GIA C5
# ==========================================================

def prepare_c5_features(
    wide,
    calendar,
    weekly,
    origin,
):

    history = prepare_history(
        wide,
        calendar,
        origin,
        weekly[KEYS + ["sell_price"]],
    )

    return attach_price_features(
        history,
        weekly,
        calendar,
    )


def origin_price_arrays(
    history,
    items,
    day,
):

    rows = (
        history.loc[
            history.day.eq(day)
        ]
        .set_index("item_id")
        .reindex(items)
    )

    require(
        len(rows) == len(items),
        "Origin item count mismatch.",
    )

    arrays = {
        name: rows[name].to_numpy(
            dtype="float32"
        )
        for name in PRICE_FEATURES
    }

    require(
        np.isfinite(
            arrays["sell_price"]
        ).all(),
        "Missing origin prices.",
    )

    for name in PRICE_FEATURES:
        require(
            not np.isinf(arrays[name]).any(),
            f"Invalid feature: {name}",
        )

    return arrays


# ==========================================================
# 3. HOC PLATT TREN DU BAO LICH SU NGOAI MAU
# ==========================================================

def fit_historical_platt(
    aux_bundle,
    matrix,
    items,
    store,
    state,
    calendar,
    prices,
    actual,
):

    _, components = forecast_c5(
        aux_bundle,
        matrix,
        items,
        store,
        state,
        calendar,
        CAL_ORIGIN,
        prices,
    )

    require(
        len(components)
        == len(items) * HORIZON,
        "Calibration forecast count mismatch.",
    )

    # forecast_c5 tra ve theo thu tu:
    # ngay 1: toan bo items
    # ngay 2: toan bo items
    # ...
    y = actual.T.reshape(-1).astype(
        np.float64
    )

    z = (y > 0).astype("int8")

    p = components[
        "probability_positive"
    ].to_numpy(
        dtype=np.float64
    )

    require(
        z.min() == 0
        and z.max() == 1,
        "Calibration has only one class.",
    )

    # Platt Scaling:
    # p_cal = sigmoid(a * logit(p) + b)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
    )

    calibrator.fit(
        logit_probability(p),
        z,
    )

    require(
        calibrator.coef_[0, 0] > 0,
        "Historical Platt slope is not positive.",
    )

    p_cal = apply_platt(
        calibrator,
        p,
    )

    metadata = {
        "store_id": store,

        "calibration_train_end": CAL_ORIGIN,

        "calibration_days": [
            CAL_ORIGIN + 1,
            VALID_ORIGIN,
        ],

        "n_calibration_rows": len(z),

        "observed_positive_rate": float(
            z.mean()
        ),

        "mean_probability_raw": float(
            p.mean()
        ),

        "mean_probability_calibrated": float(
            p_cal.mean()
        ),

        "platt_slope": float(
            calibrator.coef_[0, 0]
        ),

        "platt_intercept": float(
            calibrator.intercept_[0]
        ),

        "brier_raw": float(
            np.mean((p - z) ** 2)
        ),

        "brier_cal_in_sample": float(
            np.mean((p_cal - z) ** 2)
        ),
    }

    return calibrator, metadata


# ==========================================================
# 4. DU BAO DE QUY BANG XAC SUAT HIEU CHUAN
# ==========================================================

def forecast_c6(
    bundle,
    calibrator,
    matrix,
    items,
    store,
    state,
    calendar,
    prices,
):

    model = bundle["model"]
    features = bundle["features"]
    categories = bundle["categories"]

    for name in PRICE_FEATURES:
        require(
            name in features,
            f"C5 model missing {name}",
        )

    cal = calendar.set_index(
        "d",
        verify_integrity=True,
    )

    temporary = np.asarray(
        matrix[:, -28:],
        dtype="float32",
    ).copy()

    predictions = []
    components = []

    for step in range(1, HORIZON + 1):

        day = (
            f"d_{VALID_ORIGIN + step}"
        )

        X = next_day_features(
            temporary,
            items,
            cal.loc[day],
            state,
            categories,
            last_prices=prices["sell_price"],
        )

        for name in EXTRAS:
            X[name] = prices[name]

        require(
            X[features].columns.to_list()
            == features,
            "C6 feature order mismatch.",
        )

        # Xac suat goc va luong ban co dieu kien.
        p_raw, mu = model.predict_components(
            X[features]
        )

        # Hieu chuan xac suat.
        p_cal = apply_platt(
            calibrator,
            p_raw,
        )

        # Du bao C6.
        y_hat = np.maximum(
            p_cal * mu,
            0,
        )

        require(
            np.isfinite(y_hat).all(),
            "C6 prediction invalid.",
        )

        predictions.append(
            pd.DataFrame({
                "model": "lightgbm_two_part",
                "split": "validation",
                "origin_day": f"d_{VALID_ORIGIN}",
                "item_id": items,
                "store_id": store,
                "target_day": day,
                "horizon": step,
                "y_pred": y_hat,
            })
        )

        components.append(
            pd.DataFrame({
                "item_id": items,
                "store_id": store,
                "target_day": day,
                "probability_positive": p_cal,
                "conditional_units": mu,
                "probability_raw": p_raw,
            })
        )

        # Quan trong:
        # Ngay sau su dung du bao C6,
        # khong su dung du bao C5 hoac y thuc te.
        temporary = np.column_stack(
            (
                temporary[:, 1:],
                y_hat,
            )
        )

    forecast = pd.concat(
        predictions,
        ignore_index=True,
    )[FORECAST_COLUMNS]

    validate_forecasts(forecast)

    return (
        forecast,
        pd.concat(
            components,
            ignore_index=True,
        ),
    )


# ==========================================================
# 5. TINH CHI SO TONG HOP
# ==========================================================

def compute_summary(
    scores,
    records,
    name,
):

    scores = pd.DataFrame(scores)
    records = pd.DataFrame(records)

    valid = scores[
        "rmsse_valid_items"
    ].astype(float)

    actual = records[
        f"actual_{name}"
    ].sum()

    predicted = records[
        f"prediction_{name}"
    ].sum()

    return {
        "configuration": name,

        "actual": actual,

        "prediction": predicted,

        "total_difference": (
            predicted - actual
        ),

        "bias_pct": (
            100 * (
                predicted / actual - 1
            )
        ),

        "MAE": float(
            np.average(
                scores["MAE"],
                weights=scores["n_items"],
            )
        ),

        "RMSSE_mean_unweighted": float(
            np.average(
                scores["RMSSE_mean_unweighted"],
                weights=valid,
            )
        ),
    }


# ==========================================================
# 6. CHUONG TRINH CHINH
# ==========================================================

def main(selected_stores=None):

    config = yaml.safe_load(
        (
            ROOT / "configs/config.yaml"
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    results = (
        ROOT / config["results_dir"]
    )

    c5_dir = (
        results / "metrics" / C5_TAG
    )

    c5_manifest = read_json(
        c5_dir / "manifest.json"
    )

    require(
        c5_manifest["status"] == "completed"
        and c5_manifest["configuration"] == "C5"
        and c5_manifest["origin_day"] == VALID_ORIGIN,
        "C5 manifest invalid.",
    )

    sources = {
        item["store_id"]: item
        for item in c5_manifest["sources"]
    }

    stores = sorted(sources)

    if selected_stores:

        require(
            set(selected_stores).issubset(
                sources
            ),
            "Unknown store.",
        )

        stores = [
            store for store in stores
            if store in selected_stores
        ]

    require(
        bool(stores),
        "No stores selected.",
    )

    data_dir = (
        ROOT / config["data_dir"]
    )

    check_files(
        data_dir,
        use_prices=True,
    )

    calendar = load_calendar(
        data_dir
    )

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    batch = (
        f"validation_C6_platt_"
        f"{len(stores)}_stores_{run_id}"
    )

    metrics_dir = (
        results / "metrics" / batch
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    save_json(
        {
            "status": "running",
            "configuration": "C6",
            "c5_batch": C5_TAG,
            "calibration_origin": CAL_ORIGIN,
            "validation_origin": VALID_ORIGIN,
            "stores": stores,
        },
        metrics_dir / "manifest.json",
    )

    c5_scores = pd.read_csv(
        c5_dir / "C5_metrics_by_store.csv"
    )

    scores5 = []
    scores6 = []

    comparisons = []
    calibration_reports = []

    # ------------------------------------------------------
    # 7. LAP QUA TUNG CUA HANG
    # ------------------------------------------------------

    for store in stores:

        LOG.info(
            "C6 START | %s",
            store,
        )

        source = sources[store]

        tag = f"{batch}_{store}"

        paths = {
            name: results / name / tag
            for name in [
                "models",
                "metrics",
                "validation",
            ]
        }

        # Nap C5 da train.
        c5_model_path = (
            results
            / "models"
            / source["run_tag"]
            / "model.joblib"
        )

        require(
            c5_model_path.exists(),
            f"Missing C5 model: {c5_model_path}",
        )

        c5_bundle = verify_saved_models(
            c5_model_path
        )

        require(
            c5_bundle["origin_day"]
            == VALID_ORIGIN,
            "C5 origin mismatch.",
        )

        items = list(
            c5_bundle["categories"]
        )

        require(
            len(items) == 3049,
            "Expected 3049 items.",
        )

        # --------------------------------------------------
        # 8. DOC LICH SU DEN d_1885
        # --------------------------------------------------

        wide = load_store(
            data_dir,
            store,
            VALID_ORIGIN,
            chunk_size=config["chunk_size"],
            item_ids=items,
        )

        require(
            wide.item_id.to_list() == items,
            "Item order mismatch.",
        )

        state = wide.state_id.iloc[0]

        full_matrix = history_matrix(
            wide,
            VALID_ORIGIN,
        )

        _, selected, component_item = (
            training_patterns(
                full_matrix,
                items,
            )
        )

        prices = load_prices(
            data_dir,
            store,
        )

        weekly = build_weekly_features(
            prices,
            calendar,
        )

        del prices

        # --------------------------------------------------
        # 9. TRAIN MO HINH PHU DEN d_1857
        # --------------------------------------------------

        historical = prepare_c5_features(
            wide,
            calendar,
            weekly,
            CAL_ORIGIN,
        )

        calibration_prices = origin_price_arrays(
            historical,
            items,
            CAL_ORIGIN,
        )

        train = make_training_data(
            historical
        )

        require(
            train["fit_end_day"]
            == CAL_ORIGIN,
            "Calibration data leakage.",
        )

        del historical
        gc.collect()

        params = {
            name: {
                **c5_manifest["config"][name],
                "importance_type": "gain",
            }
            for name in [
                "classifier",
                "regressor",
            ]
        }

        auxiliary_bundle, train_info = (
            train_model(
                train,
                params,
                paths["models"]
                / "aux_model.joblib",
            )
        )

        del train
        gc.collect()

        # --------------------------------------------------
        # 10. TAO DU BAO NGOAI MAU LICH SU
        # --------------------------------------------------

        actual_calibration = wide[
            [
                f"d_{day}"
                for day in range(
                    CAL_ORIGIN + 1,
                    VALID_ORIGIN + 1,
                )
            ]
        ].to_numpy()

        calibrator, cal_info = fit_historical_platt(
            auxiliary_bundle,
            full_matrix[:, :CAL_ORIGIN],
            items,
            store,
            state,
            calendar,
            calibration_prices,
            actual_calibration,
        )

        cal_info[
            "aux_training_seconds"
        ] = train_info["training_time_seconds"]

        calibration_reports.append(
            cal_info
        )

        joblib.dump(
            calibrator,
            paths["models"] / "platt.joblib",
        )

        save_json(
            cal_info,
            paths["models"] / "calibration.json",
        )

        del auxiliary_bundle
        del actual_calibration

        gc.collect()

        # --------------------------------------------------
        # 11. CHUAN BI FEATURE TAI d_1885
        # --------------------------------------------------

        final_history = prepare_c5_features(
            wide,
            calendar,
            weekly,
            VALID_ORIGIN,
        )

        final_prices = origin_price_arrays(
            final_history,
            items,
            VALID_ORIGIN,
        )

        del final_history
        del weekly
        del wide

        gc.collect()

        # --------------------------------------------------
        # 12. DU BAO C6 BANG p_cal
        # --------------------------------------------------

        prediction, components = forecast_c6(
            c5_bundle,
            calibrator,
            full_matrix,
            items,
            store,
            state,
            calendar,
            final_prices,
        )

        paths["validation"].mkdir(
            parents=True,
            exist_ok=False,
        )

        paths["metrics"].mkdir(
            parents=True,
            exist_ok=False,
        )

        export_forecasts(
            prediction,
            paths["validation"]
            / "lightgbm_two_part.csv",
        )

        # --------------------------------------------------
        # 13. CHAM DIEM VALIDATION
        # --------------------------------------------------

        actual = load_store(
            data_dir,
            store,
            VALID_ORIGIN + HORIZON,
            start_day=VALID_ORIGIN + 1,
            item_ids=items,
            chunk_size=config["chunk_size"],
        )

        baseline = pd.read_csv(
            results
            / "validation"
            / source["run_tag"]
            / "seasonal_naive.csv",
            float_precision="round_trip",
        )

        validate_forecasts(
            baseline
        )

        export_forecasts(
            baseline,
            paths["validation"]
            / "seasonal_naive.csv",
        )

        context = {
            "config": c5_manifest["config"],
            "configuration_name": "C6",
            "store_id": store,
            "scope": "full",
            "split": "validation",
            "origin_day": VALID_ORIGIN,
            "run_id": run_id,
            "batch_tag": batch,
            "illustration_items": selected,
            "component_item": component_item,
            "save_figures": False,
            "mode": "validation",
            "status": "completed",
            "regressor_retrained": False,
            "forecast_performed": True,
            "evaluation_performed": True,
            "price_policy": "previous_completed_week",
            "features_added": EXTRAS,
            "probability_calibration": "historical_platt",
            "calibration_days": [1858, 1885],
            "model_source": "C5",
        }

        info = {
            "training_time_seconds":
                train_info["training_time_seconds"],
            "calibration_source":
                "out_of_sample_aux_d1858_d1885",
        }

        scores = evaluate_and_export(
            prediction,
            baseline,
            actual,
            full_matrix,
            items,
            c5_bundle,
            info,
            paths,
            context,
            components=components,
        )

        score6 = scores.loc[
            scores.model.eq(
                "lightgbm_two_part"
            )
        ]

        require(
            len(score6) == 1,
            "C6 RMSSE missing.",
        )

        scores6.append(
            score6.iloc[0].to_dict()
        )

        score5 = c5_scores.loc[
            c5_scores.model.eq("lightgbm_two_part")
            & c5_scores.store_id.eq(store)
        ]

        require(
            len(score5) == 1,
            "C5 RMSSE missing.",
        )

        scores5.append(
            score5.iloc[0].to_dict()
        )

        # --------------------------------------------------
        # 14. DOI CHIEU C5 VA C6
        # --------------------------------------------------

        detail5 = pd.read_csv(
            source["actual_vs_predictions"],
            float_precision="round_trip",
        )

        detail6 = pd.read_csv(
            paths["validation"]
            / "actual_vs_predictions.csv",
            float_precision="round_trip",
        )

        check_join(
            detail5,
            detail6,
            store,
        )

        row = {
            "store_id": store,
        }

        for name, detail in [
            ("C5", detail5),
            ("C6", detail6),
        ]:

            row.update({
                f"{key}_{name}": value

                for key, value
                in diagnostics(detail).items()
            })

        row["RMSSE_C5"] = float(
            score5.iloc[0][
                "RMSSE_mean_unweighted"
            ]
        )

        row["RMSSE_C6"] = float(
            score6.iloc[0][
                "RMSSE_mean_unweighted"
            ]
        )

        row["delta_RMSSE"] = (
            row["RMSSE_C6"]
            - row["RMSSE_C5"]
        )

        comparisons.append(row)

        LOG.info(
            "C6 COMPLETE | %s | RMSSE: %.6f -> %.6f"
            " | Bias: %.2f%% -> %.2f%%",
            store,
            row["RMSSE_C5"],
            row["RMSSE_C6"],
            row["bias_pct_C5"],
            row["bias_pct_C6"],
        )

        del (
            c5_bundle,
            prediction,
            components,
            actual,
            baseline,
            detail5,
            detail6,
        )

        gc.collect()

    # ------------------------------------------------------
    # 15. TONG HOP KET QUA
    # ------------------------------------------------------

    pd.DataFrame(
        comparisons
    ).to_csv(
        metrics_dir / "comparison_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        scores6
    ).to_csv(
        metrics_dir / "C6_metrics_by_store.csv",
        index=False,
    )

    pd.DataFrame(
        calibration_reports
    ).to_csv(
        metrics_dir / "calibration_by_store.csv",
        index=False,
    )

    overall = pd.DataFrame([
        compute_summary(
            scores5,
            comparisons,
            "C5",
        ),

        compute_summary(
            scores6,
            comparisons,
            "C6",
        ),
    ])

    overall.to_csv(
        metrics_dir / "comparison_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        {
            "status": (
                "completed"
                if len(stores) == 10
                else "completed_smoke"
            ),

            "configuration": "C6",
            "c5_batch": C5_TAG,

            "calibration_train_end": CAL_ORIGIN,
            "calibration_days": [1858, 1885],

            "validation_origin": VALID_ORIGIN,
            "validation_days": [1886, 1913],

            "stores": stores,
            "n_items": len(stores) * 3049,
            "n_predictions":
                len(stores) * 3049 * HORIZON,

            "c5_models_reused": True,
            "auxiliary_models_trained": len(stores),
            "platt_per_store": True,

            "validation_not_used_for_calibration": True,
            "test_performed": False,
        },
        metrics_dir / "manifest.json",
    )

    print("\n=== C5 VS C6 ===")

    print(
        overall.to_string(index=False)
    )

    print(
        "\nResults:",
        metrics_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stores",
        nargs="*",
        default=None,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args.stores)
