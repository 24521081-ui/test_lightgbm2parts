
"""FINAL TEST C6 | M5 d_1914..d_1941.

Origin: d_1913.

- Giữ nguyên cấu hình C5/C6 đã chốt.
- Dùng lại C5 model d_1885 làm mô hình phụ hiệu chuẩn.
- Học Platt trên d_1886..d_1913.
- Train lại LightGBM chính đến d_1913.
- Dự báo đệ quy d_1914..d_1941.
- Chỉ đọc dữ liệu Test để chấm sau khi đã dự báo.
- Lưu model và Platt để C7 sử dụng lại.
"""

from __future__ import annotations

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

from src.data_loader import (
    check_files,
    load_calendar,
    load_prices,
    load_store,
)

from src.preprocessing import history_matrix
from src.reporting import evaluate_forecast
from src.training import train_model, verify_saved_models
from src.forecasting import validate_forecasts

from run_c5_price_features import (
    EXTRAS,
    build_weekly_features,
    make_training_data,
    forecast_c5,
)

from run_c6_platt import prepare_c5_features

from run_stage_b_robustness import (
    C5_TAG,
    H,
    STORE_CODES,
    check_prediction_order,
    find_c6_batch,
    fit_platt,
    forecast_c6_at_origin,
    historical_price_arrays,
    read_json,
    require,
    write_json,
)


# ============================================================
# 1. CAU HINH
# ============================================================

LOG = logging.getLogger("final_test_c6")

ORIGIN = 1913
AUX_ORIGIN = 1885

VERSION = "FINAL_TEST_C6_FROZEN_V1"
OUTPUT_NAME = "final_test_C6_C7_frozen"


# ============================================================
# 2. XAC NHAN CAU HINH DA CHOT
# ============================================================

def load_reference(results):

    c5 = read_json(
        results
        / "metrics"
        / C5_TAG
        / "manifest.json"
    )

    require(
        c5.get("configuration") == "C5"
        and c5.get("status") == "completed"
        and c5.get("origin_day") == AUX_ORIGIN,
        "C5 Validation reference khong hop le.",
    )

    require(
        c5.get("price_policy")
        == "previous_completed_week",
        "C5 phai su dung previous_completed_week.",
    )

    stage_b = read_json(
        results
        / "analysis"
        / "stage_B_C5_C6_full"
        / "manifest.json"
    )

    require(
        stage_b.get("status") == "complete"
        and stage_b.get("C5_batch") == C5_TAG,
        "Stage B chua complete hoac khac C5.",
    )

    c6_tag, c6 = find_c6_batch(
        results,
        stage_b["C6_batch"],
    )

    require(
        c6.get("status") == "completed"
        and c6.get("configuration") == "C6"
        and c6.get("calibration_train_end") == 1857
        and c6.get("validation_origin") == 1885,
        "C6 reference khong hop le.",
    )

    classifier = c5["config"]["classifier"]
    regressor = c5["config"]["regressor"]

    require(
        classifier["num_leaves"] == 48
        and classifier["n_estimators"] == 300
        and regressor["num_leaves"] == 31
        and regressor["n_estimators"] == 200,
        "Sai cau hinh LightGBM da chot.",
    )

    sources = {
        x["store_id"]: x
        for x in c5["sources"]
    }

    require(
        set(sources) == set(STORE_CODES),
        "C5 thieu cua hang.",
    )

    params = {
        key: {
            **c5["config"][key],
            "importance_type": "gain",
        }
        for key in ("classifier", "regressor")
    }

    return c5, c6_tag, sources, params


# ============================================================
# 3. NAP LIGHTGBM PHU TAI d_1885
# ============================================================

def load_auxiliary(
    store,
    results,
    sources,
    items,
):

    path = (
        results
        / "models"
        / sources[store]["run_tag"]
        / "model.joblib"
    )

    require(
        path.is_file(),
        f"Thieu C5 model d_1885: {path}",
    )

    bundle = verify_saved_models(path)

    require(
        bundle["origin_day"] == AUX_ORIGIN
        and bundle["categories"] == items,
        "Auxiliary sai ngay Train hoac item order.",
    )

    require(
        all(
            f in bundle["features"]
            for f in ("sell_price", *EXTRAS)
        ),
        "Auxiliary thieu features gia C5.",
    )

    return bundle, path


# ============================================================
# 4. TRAIN LIGHTGBM CHINH DEN d_1913
# ============================================================

def get_main(
    store,
    wide,
    calendar,
    weekly,
    items,
    params,
    output,
):

    path = (
        output
        / "models"
        / store
        / "model.joblib"
    )

    # Khong Train lai neu model d_1913 da luu.
    if path.is_file():

        model = verify_saved_models(path)

        require(
            model["origin_day"] == ORIGIN
            and model["categories"] == items,
            "Cached main model khong dung d_1913.",
        )

        require(
            all(
                f in model["features"]
                for f in ("sell_price", *EXTRAS)
            ),
            "Saved model thieu features gia.",
        )

        LOG.info(
            "REUSE main d_%d: %s",
            ORIGIN,
            store,
        )

        return model, "cached_main_d1913"

    # Tao features chi tu du lieu den d_1913.
    hist = prepare_c5_features(
        wide,
        calendar,
        weekly,
        ORIGIN,
    )

    training = make_training_data(hist)

    require(
        training["fit_end_day"] == ORIGIN
        and training["categories"] == items,
        "Training data sai origin hoac item order.",
    )

    require(
        len(training["X"]) > 0
        and len(training["X"]) == len(training["y"]),
        "Training data khong hop le.",
    )

    del hist
    gc.collect()

    bundle, info = train_model(
        training,
        params,
        path,
    )

    require(
        bundle["origin_day"] == ORIGIN,
        "Model Train sai ngay ket thuc.",
    )

    write_json(
        path.parent / "training_info.json",
        info,
    )

    LOG.info(
        "TRAIN main %s to d_%d: %.1f seconds",
        store,
        ORIGIN,
        info["training_time_seconds"],
    )

    del training
    gc.collect()

    return bundle, "trained_main_d1913"


# ============================================================
# 5. CHAY C6 TEST CHO MOT CUA HANG
# ============================================================

def score_one_store(
    store,
    config,
    calendar,
    results,
    sources,
    params,
    output,
):

    folder = (
        output
        / "folds"
        / store
    )

    done = folder / "completed.json"

    # --------------------------------------------------------
    # 5.1. TAI SU DUNG KET QUA DA HOAN THANH
    # --------------------------------------------------------

    if done.is_file():

        cached = read_json(done)

        require(
            cached.get("version") == VERSION
            and cached.get("store_id") == store,
            "Cached C6 Test khong tuong thich.",
        )

        for name in (
            "forecast_C6.csv",
            "per_item.csv",
            "by_week.csv",
            "platt.joblib",
        ):
            require(
                (folder / name).is_file(),
                f"Cache thieu: {folder / name}",
            )

        require(
            (
                output
                / "models"
                / store
                / "model.joblib"
            ).is_file(),
            "Thieu LightGBM d_1913 cho C7.",
        )

        LOG.info(
            "REUSE C6 Test %s",
            store,
        )

        return (
            cached["record"],
            pd.read_csv(folder / "by_week.csv"),
            pd.read_csv(folder / "per_item.csv"),
            cached["calibration"],
        )

    # --------------------------------------------------------
    # 5.2. DOC LICH SU CHI DEN d_1913
    # --------------------------------------------------------

    data_dir = ROOT / config["data_dir"]

    wide = load_store(
        data_dir,
        store,
        ORIGIN,
        chunk_size=config["chunk_size"],
    )

    items = wide["item_id"].tolist()

    require(
        len(items) == 3049
        and len(set(items)) == 3049,
        f"{store}: phai co 3049 series.",
    )

    require(
        all(
            f"d_{d}" in wide.columns
            for d in range(1, ORIGIN + 1)
        ),
        "Thieu cot du lieu Training.",
    )

    require(
        not any(
            f"d_{d}" in wide.columns
            for d in range(ORIGIN + 1, 1942)
        ),
        "Training dataframe chua du lieu Test!",
    )

    state = str(
        wide.state_id.iloc[0]
    )

    matrix = history_matrix(
        wide,
        ORIGIN,
    )

    weekly = build_weekly_features(
        load_prices(data_dir, store),
        calendar,
    )

    # --------------------------------------------------------
    # 5.3. DU BAO NGOAI MAU CHO PLATT
    # --------------------------------------------------------

    auxiliary, aux_path = load_auxiliary(
        store,
        results,
        sources,
        items,
    )

    aux_prices = historical_price_arrays(
        weekly,
        calendar,
        items,
        AUX_ORIGIN,
    )

    _, aux_comps = forecast_c5(
        auxiliary,
        matrix[:, :AUX_ORIGIN],
        items,
        store,
        state,
        calendar,
        AUX_ORIGIN,
        aux_prices,
    )

    check_prediction_order(
        aux_comps,
        AUX_ORIGIN,
        items,
    )

    # Actual chi thuoc calibration d_1886..d_1913.
    cal_actual = wide[
        [
            f"d_{d}"
            for d in range(
                AUX_ORIGIN + 1,
                ORIGIN + 1,
            )
        ]
    ].to_numpy()

    # --------------------------------------------------------
    # 5.4. HOC LAI PLATT SCALING
    # --------------------------------------------------------

    platt, diag = fit_platt(
        aux_comps,
        cal_actual,
        AUX_ORIGIN,
        ORIGIN,
        items,
    )

    require(
        diag["n_calibration_rows"]
        == len(items) * H,
        "So quan sat calibration khong dung.",
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        platt,
        folder / "platt.joblib",
    )

    del (
        auxiliary,
        aux_comps,
        aux_prices,
        cal_actual,
    )

    gc.collect()

    # --------------------------------------------------------
    # 5.5. TRAIN LIGHTGBM CHINH d_1913
    # --------------------------------------------------------

    main, source = get_main(
        store,
        wide,
        calendar,
        weekly,
        items,
        params,
        output,
    )

    main_prices = historical_price_arrays(
        weekly,
        calendar,
        items,
        ORIGIN,
    )

    del weekly
    gc.collect()

    # --------------------------------------------------------
    # 5.6. DU BAO C6 DE QUY 28 NGAY
    # --------------------------------------------------------

    pred, comp = forecast_c6_at_origin(
        main,
        platt,
        matrix,
        items,
        store,
        state,
        calendar,
        ORIGIN,
        main_prices,
    )

    check_prediction_order(
        comp,
        ORIGIN,
        items,
    )

    pred["split"] = "test"

    validate_forecasts(pred)

    require(
        pred["target_day"].nunique() == H
        and set(pred["target_day"])
        == {
            f"d_{d}"
            for d in range(1914, 1942)
        },
        "Sai ngay du bao Test.",
    )

    require(
        len(pred) == 3049 * H,
        "Thieu du bao Test.",
    )

    # Luu du bao truoc khi doc actual Test.
    pred.to_csv(
        folder / "forecast_C6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 5.7. DOC ACTUAL TEST SAU KHI DU BAO
    # --------------------------------------------------------

    holdout = load_store(
        data_dir,
        store,
        ORIGIN + H,
        chunk_size=config["chunk_size"],
        start_day=ORIGIN + 1,
        item_ids=items,
    )

    require(
        holdout["item_id"].tolist() == items,
        "Test item order khong khop.",
    )

    # --------------------------------------------------------
    # 5.8. TINH RMSSE / MAE / RMSE
    # --------------------------------------------------------

    score, item_scores, truth, yhat = (
        evaluate_forecast(
            pred,
            holdout,
            matrix,
            items,
        )
    )

    z = (
        (truth > 0)
        .astype(float)
        .T.reshape(-1)
    )

    praw = comp[
        "probability_raw"
    ].to_numpy(dtype=float)

    pcal = comp[
        "probability_positive"
    ].to_numpy(dtype=float)

    require(
        len(z) == len(praw) == len(pcal),
        "Sai so luong xac suat/actual.",
    )

    actual_total = float(
        truth.sum()
    )

    forecast_total = float(
        yhat.sum()
    )

    require(
        actual_total > 0,
        "Actual Test total khong hop le.",
    )

    # --------------------------------------------------------
    # 5.9. BAO CAO CAP CUA HANG
    # --------------------------------------------------------

    record = {
        "store_id": store,
        "origin": ORIGIN,
        "start_day": 1914,
        "end_day": 1941,

        "n_items": len(items),

        "valid_rmsse_items":
            int(score["rmsse_valid_items"]),

        "actual": actual_total,

        "predicted_C6": forecast_total,

        "bias_C6_pct":
            float(
                100 * (
                    forecast_total / actual_total - 1
                )
            ),

        "RMSSE_C6":
            float(score["RMSSE_mean_unweighted"]),

        "MAE_C6":
            float(score["MAE"]),

        "RMSE_C6":
            float(score["RMSE"]),

        "Brier_raw_C6_path":
            float(np.square(praw - z).mean()),

        "Brier_cal_C6_path":
            float(np.square(pcal - z).mean()),

        "mean_p_raw_C6_path":
            float(praw.mean()),

        "mean_p_cal_C6_path":
            float(pcal.mean()),

        "positive_rate":
            float(z.mean()),

        "missing_origin_sell_price":
            int(
                np.isnan(
                    main_prices["sell_price"]
                ).sum()
            ),

        "model_source": source,

        "auxiliary_model":
            str(aux_path),
    }

    # --------------------------------------------------------
    # 5.10. BAO CAO CAP SAN PHAM
    # --------------------------------------------------------

    per_item = item_scores[
        ["item_id", "RMSSE", "scale"]
    ].rename(
        columns={"RMSSE": "RMSSE_C6"}
    ).copy()

    per_item.insert(
        0,
        "store_id",
        store,
    )

    per_item["actual_units"] = (
        truth.sum(axis=1)
    )

    per_item["predicted_C6"] = (
        yhat.sum(axis=1)
    )

    # --------------------------------------------------------
    # 5.11. BAO CAO THEO 4 TUAN
    # --------------------------------------------------------

    scales = per_item[
        "scale"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(scales)
        & (scales > 0)
    )

    weekly_rows = []

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

        mse = np.square(
            a - f
        ).mean(axis=1)

        weekly_rows.append({
            "store_id": store,
            "week": week,

            "start_day":
                ORIGIN + start + 1,

            "end_day":
                ORIGIN + start + 7,

            "actual": a_sum,
            "predicted_C6": f_sum,

            "bias_C6_pct": (
                float(
                    100 * (f_sum / a_sum - 1)
                )
                if a_sum > 0
                else np.nan
            ),

            "RMSSE_C6":
                float(
                    np.sqrt(
                        mse[valid] / scales[valid]
                    ).mean()
                ),

            "MAE_C6":
                float(
                    np.abs(a - f).mean()
                ),
        })

    # --------------------------------------------------------
    # 5.12. LUU KET QUA
    # --------------------------------------------------------

    per_item.to_csv(
        folder / "per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(weekly_rows).to_csv(
        folder / "by_week.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Chi ghi completed khi da co du file.
    write_json(
        done,
        {
            "version": VERSION,
            "store_id": store,
            "origin": ORIGIN,
            "record": record,
            "calibration": diag,
        },
    )

    LOG.info(
        "%s Test: RMSSE=%.6f MAE=%.6f bias=%+.2f%%",
        store,
        record["RMSSE_C6"],
        record["MAE_C6"],
        record["bias_C6_pct"],
    )

    del (
        wide,
        matrix,
        main,
        pred,
        comp,
        holdout,
        truth,
        yhat,
    )

    gc.collect()

    return (
        record,
        pd.DataFrame(weekly_rows),
        per_item,
        diag,
    )


# ============================================================
# 6. MAIN - CHAY VA TONG HOP
# ============================================================

def run(args):

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

    # Kiem tra bo M5 evaluation day du.
    check_files(
        data_dir,
        use_prices=True,
    )

    c5, c6_tag, sources, params = (
        load_reference(results)
    )

    require(
        config["test_origin"] == ORIGIN
        and config["validation_origin"] == AUX_ORIGIN,
        "config.yaml thay doi origin.",
    )

    stores = sorted(
        args.stores or STORE_CODES
    )

    require(
        bool(stores)
        and len(stores) == len(set(stores))
        and set(stores).issubset(STORE_CODES),
        "Store khong hop le.",
    )

    calendar = load_calendar(
        data_dir
    )

    require(
        {
            f"d_{d}"
            for d in range(1914, 1942)
        }.issubset(set(calendar["d"])),
        "Calendar thieu Test horizon.",
    )

    output = (
        results
        / "analysis"
        / OUTPUT_NAME
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    weeks = []
    items = []
    diagnostics = []

    # --------------------------------------------------------
    # 6.1. CHAY TUNG CUA HANG
    # --------------------------------------------------------

    for store in stores:

        row, week, item, cal = score_one_store(
            store,
            config,
            calendar,
            results,
            sources,
            params,
            output,
        )

        records.append(row)
        weeks.append(week)
        items.append(item)

        diagnostics.append({
            "store_id": store,
            **cal,
        })

    # --------------------------------------------------------
    # 6.2. TONG HOP THEO CUA HANG
    # --------------------------------------------------------

    by_store = (
        pd.DataFrame(records)
        .sort_values("store_id")
        .reset_index(drop=True)
    )

    actual = float(
        by_store.actual.sum()
    )

    forecast = float(
        by_store.predicted_C6.sum()
    )

    total_valid = by_store[
        "valid_rmsse_items"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 6.3. TONG HOP TOAN BO 10 CUA HANG
    # --------------------------------------------------------

    overall = pd.DataFrame([
        {
            "origin": ORIGIN,
            "start_day": 1914,
            "end_day": 1941,

            "n_stores": len(stores),

            "n_items":
                int(by_store.n_items.sum()),

            "valid_rmsse_items":
                int(total_valid.sum()),

            "actual": actual,

            "predicted_C6": forecast,

            "bias_C6_pct":
                float(
                    100 * (
                        forecast / actual - 1
                    )
                ),

            "RMSSE_C6":
                float(
                    np.average(
                        by_store.RMSSE_C6,
                        weights=total_valid,
                    )
                ),

            "MAE_C6":
                float(
                    np.average(
                        by_store.MAE_C6,
                        weights=by_store.n_items,
                    )
                ),

            "RMSE_C6":
                float(
                    np.sqrt(
                        np.average(
                            np.square(
                                by_store.RMSE_C6
                            ),
                            weights=by_store.n_items,
                        )
                    )
                ),

            "Brier_raw_C6_path":
                float(
                    np.average(
                        by_store.Brier_raw_C6_path,
                        weights=by_store.n_items,
                    )
                ),

            "Brier_cal_C6_path":
                float(
                    np.average(
                        by_store.Brier_cal_C6_path,
                        weights=by_store.n_items,
                    )
                ),

            "mean_abs_store_bias_C6_pct":
                float(
                    by_store.bias_C6_pct.abs().mean()
                ),
        }
    ])

    # --------------------------------------------------------
    # 6.4. XUAT FILE BAO CAO
    # --------------------------------------------------------

    overall.to_csv(
        output / "01_by_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_store.to_csv(
        output / "02_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        weeks,
        ignore_index=True,
    ).to_csv(
        output / "03_by_week.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        items,
        ignore_index=True,
    ).to_csv(
        output / "04_per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(diagnostics).to_csv(
        output / "05_calibration_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6.5. MANIFEST
    # --------------------------------------------------------

    status = (
        "complete"
        if set(stores) == set(STORE_CODES)
        else "partial"
    )

    write_json(
        output / "manifest_C6.json",
        {
            "version": VERSION,
            "status": status,

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "model": "C6",
            "stores": stores,

            "origin": ORIGIN,
            "test_days": [1914, 1941],

            "calibration_training_cutoff":
                AUX_ORIGIN,

            "calibration_window":
                [1886, 1913],

            "main_training_cutoff":
                ORIGIN,

            "source_C5_batch": C5_TAG,
            "source_C6_batch": c6_tag,

            "price_policy":
                "previous_completed_week",

            "lightgbm_hyperparams_frozen": True,
            "c6_platt_method_frozen": True,

            "c7_lambda_frozen": 0.75,
            "c7_test_performed": False,
            "c6_test_performed": True,
        },
    )

    # --------------------------------------------------------
    # 6.6. BAO CAO TERMINAL
    # --------------------------------------------------------

    print(
        "\n=== FINAL TEST C6 / d_1914..d_1941 ==="
    )

    print(
        overall.to_string(index=False)
    )

    print(
        "\nSaved:",
        output,
    )

    if status != "complete":
        print(
            "PARTIAL: Chua du 10 cua hang."
        )


# ============================================================
# 7. EXECUTE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stores",
        nargs="+",
        default=None,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run(args)
