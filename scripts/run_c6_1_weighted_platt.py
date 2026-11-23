
"""C6.1 - Weighted Platt Scaling (1x / 2x).

Goc du bao:
    d_1829 -> d_1830 ... d_1857
    d_1857 -> d_1858 ... d_1885

Cau hinh:
    - LightGBM: giu nguyen Stage B.
    - Platt Scaling: hoc lai voi trong so.
    - 14 ngay dau: 1x.
    - 14 ngay cuoi: 2x.
    - Chuan hoa trong so ve mean = 1.
    - LogisticRegression(C=1, solver='lbfgs').
    - Du bao de quy 28 ngay.

Khong train lai LightGBM.
Khong su dung Validation d_1885 de chon cau hinh.
Khong chay Test.

Output:
    results/analysis/stage_C6_1_weighted_platt_1x2x/
"""

from __future__ import annotations

import argparse
import gc
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

from src.preprocessing import history_matrix
from src.reporting import evaluate_forecast
from src.training import verify_saved_models

from run_c5_price_features import (
    build_weekly_features,
    forecast_c5,
)

from run_c6_platt import (
    apply_platt,
    logit_probability,
)

from run_stage_b_robustness import (
    FOLDS,
    H,
    STORE_CODES,
    VERSION as STAGE_B_VERSION,
    check_prediction_order,
    forecast_c6_at_origin,
    historical_price_arrays,
    read_json,
    require,
    write_json,
)


# ============================================================
# 1. CAU HINH C6.1
# ============================================================

VERSION = "C6_1_WEIGHTED_PLATT_1X2X_V1"

OUTPUT_DIR = "stage_C6_1_weighted_platt_1x2x"

FIRST_HALF_WEIGHT = 1.0
SECOND_HALF_WEIGHT = 2.0


# ============================================================
# 2. NAP LIGHTGBM DA TRAIN TU STAGE B
# ============================================================

def load_saved_model(
    results,
    stage_b,
    c6_tag,
    store,
    fit_origin,
    items,
):

    path = (
        stage_b
        / "models"
        / store
        / f"d_{fit_origin}"
        / "model.joblib"
    )

    if not path.is_file() and fit_origin == 1857:
        path = (
            results
            / "models"
            / f"{c6_tag}_{store}"
            / "aux_model.joblib"
        )

    require(
        path.is_file(),
        f"Missing LightGBM: {path}",
    )

    bundle = verify_saved_models(path)

    require(
        int(bundle["origin_day"]) == fit_origin,
        f"Incorrect training cutoff: {path}",
    )

    require(
        bundle["categories"] == items,
        f"Incorrect item ordering: {path}",
    )

    return bundle


# ============================================================
# 3. HOC WEIGHTED PLATT SCALING
# ============================================================

def learn_weighted_platt(
    cal_comps,
    cal_actual,
    aux_origin,
    origin,
    items,
    original_calibrator,
    original_record,
):

    require(
        aux_origin + H == origin and H == 28,
        "Wrong calibration windows.",
    )

    check_prediction_order(
        cal_comps,
        aux_origin,
        items,
    )

    # Xac suat goc tren 28 ngay calibration
    raw = cal_comps[
        "probability_positive"
    ].to_numpy(dtype=np.float64)

    # Luong ban thuc te tren calibration
    y = np.asarray(
        cal_actual,
        dtype=np.float64,
    ).T.reshape(-1)

    require(
        len(raw) == len(y) == len(items) * H,
        "Wrong calibration length.",
    )

    require(
        np.isfinite(raw).all(),
        "Non-finite calibration probabilities.",
    )

    # Nhan 1 = co ban, 0 = khong ban
    z = (y > 0).astype(np.int8)

    require(
        z.min() == 0 and z.max() == 1,
        "Only one target class.",
    )

    # --------------------------------------------------------
    # 3.1. Doi chieu voi Stage B
    # --------------------------------------------------------

    np.testing.assert_allclose(
        z.mean(),
        original_record["cal_positive_rate"],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        raw.mean(),
        original_record["cal_mean_p_raw"],
        rtol=1e-8,
        atol=1e-8,
    )

    original_p = apply_platt(
        original_calibrator,
        raw,
    )

    np.testing.assert_allclose(
        np.mean(original_p),
        original_record["cal_mean_p_platt"],
        rtol=1e-8,
        atol=1e-8,
    )

    # --------------------------------------------------------
    # 3.2. Tao trong so 1x / 2x
    # --------------------------------------------------------

    day_weights = np.r_[
        np.full(14, FIRST_HALF_WEIGHT),
        np.full(14, SECOND_HALF_WEIGHT),
    ]

    # Moi ngay co 3049 series.
    # Du lieu sap theo ngay truoc, san pham sau.
    weights = np.repeat(
        day_weights,
        len(items),
    )

    # Chuan hoa de mean(weights) = 1
    # Van giu ti le 1:2
    weights = weights / weights.mean()

    require(
        np.isclose(weights.mean(), 1.0),
        "Invalid mean weight.",
    )

    require(
        np.isclose(
            weights.max() / weights.min(),
            2.0,
        ),
        "Invalid weight ratio.",
    )

    # --------------------------------------------------------
    # 3.3. Fit Weighted Platt
    # --------------------------------------------------------

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
    )

    model.fit(
        logit_probability(raw),
        z,
        sample_weight=weights,
    )

    require(
        float(model.coef_[0, 0]) > 0,
        "Nonpositive weighted Platt slope.",
    )

    weighted_p = apply_platt(
        model,
        raw,
    )

    # --------------------------------------------------------
    # 3.4. Diagnostics
    # --------------------------------------------------------

    diagnostics = {
        "aux_origin": aux_origin,
        "calibration_start": aux_origin + 1,
        "calibration_end": origin,
        "n_calibration_rows": len(raw),

        "weight_first_14": FIRST_HALF_WEIGHT,
        "weight_last_14": SECOND_HALF_WEIGHT,
        "weights_normalized_mean_1": True,

        "actual_positive_rate": float(z.mean()),
        "p_raw_mean": float(raw.mean()),

        "p_original_platt_mean":
            float(original_p.mean()),

        "p_weighted_platt_mean":
            float(weighted_p.mean()),

        "original_platt_slope":
            float(original_calibrator.coef_[0, 0]),

        "original_platt_intercept":
            float(original_calibrator.intercept_[0]),

        "weighted_platt_slope":
            float(model.coef_[0, 0]),

        "weighted_platt_intercept":
            float(model.intercept_[0]),

        "cal_brier_original_platt":
            float(np.mean((original_p - z) ** 2)),

        "cal_brier_weighted_platt":
            float(np.mean((weighted_p - z) ** 2)),
    }

    return model, diagnostics


# ============================================================
# 4. CHAY MOT CUA HANG TAI MOT GOC
# ============================================================

def evaluate_one(
    store,
    origin,
    data_dir,
    chunk_size,
    calendar,
    results,
    stage_b,
    c6_tag,
    output_dir,
):

    stage_folder = (
        stage_b
        / "folds"
        / f"d_{origin}"
        / store
    )

    previous = read_json(
        stage_folder / "completed.json"
    )

    require(
        previous.get("version") == STAGE_B_VERSION
        and previous.get("origin") == origin
        and previous.get("store_id") == store,
        "Mismatched Stage B fold.",
    )

    baseline = previous["record"]

    folder = (
        output_dir
        / "folds"
        / f"d_{origin}"
        / store
    )

    done_file = folder / "completed.json"

    # --------------------------------------------------------
    # 4.1. Tai su dung cache neu da chay
    # --------------------------------------------------------

    if done_file.is_file():

        cached = read_json(done_file)

        require(
            cached.get("version") == VERSION
            and cached.get("origin") == origin
            and cached.get("store_id") == store,
            "Incompatible C6.1 cache.",
        )

        for name in (
            "per_item.csv",
            "by_week.csv",
            "platt_weighted.joblib",
            "forecast_C6_1.csv",
        ):
            require(
                (folder / name).is_file(),
                f"Incomplete C6.1 fold: {name}",
            )

        print(
            f"REUSE {store} d_{origin}",
            flush=True,
        )

        return (
            cached["record"],
            pd.read_csv(folder / "by_week.csv"),
            pd.read_csv(folder / "per_item.csv"),
            cached["calibration"],
        )

    # --------------------------------------------------------
    # 4.2. Doc lich su den origin
    # --------------------------------------------------------

    aux_origin = origin - H

    wide = load_store(
        data_dir,
        store,
        origin,
        chunk_size=chunk_size,
    )

    items = wide.item_id.to_list()

    require(
        len(items) == 3049
        and len(set(items)) == 3049,
        f"Expected 3049 items at {store}.",
    )

    state = str(
        wide.state_id.iloc[0]
    )

    history = history_matrix(
        wide,
        origin,
    )

    require(
        history.shape[1] == origin,
        f"Incorrect history length: {store}",
    )

    weekly = build_weekly_features(
        load_prices(data_dir, store),
        calendar,
    )

    # --------------------------------------------------------
    # 4.3. Tao du bao xac suat goc tren calibration
    # --------------------------------------------------------

    aux = load_saved_model(
        results,
        stage_b,
        c6_tag,
        store,
        aux_origin,
        items,
    )

    aux_prices = historical_price_arrays(
        weekly,
        calendar,
        items,
        aux_origin,
    )

    _, cal_comps = forecast_c5(
        aux,
        history[:, :aux_origin],
        items,
        store,
        state,
        calendar,
        aux_origin,
        aux_prices,
    )

    # Chi su dung actual cua calibration
    actual_cal = wide[
        [
            f"d_{d}"
            for d in range(
                aux_origin + 1,
                origin + 1,
            )
        ]
    ].to_numpy()

    original_platt = joblib.load(
        stage_folder / "platt.joblib"
    )

    # --------------------------------------------------------
    # 4.4. Fit Weighted Platt
    # --------------------------------------------------------

    weighted_platt, cal_diagnostics = (
        learn_weighted_platt(
            cal_comps,
            actual_cal,
            aux_origin,
            origin,
            items,
            original_platt,
            baseline,
        )
    )

    del (
        aux,
        aux_prices,
        cal_comps,
        actual_cal,
    )

    gc.collect()

    # --------------------------------------------------------
    # 4.5. Tai LightGBM chinh
    # --------------------------------------------------------

    main = load_saved_model(
        results,
        stage_b,
        c6_tag,
        store,
        origin,
        items,
    )

    main_prices = historical_price_arrays(
        weekly,
        calendar,
        items,
        origin,
    )

    del weekly
    gc.collect()

    # --------------------------------------------------------
    # 4.6. Du bao de quy 28 ngay
    # --------------------------------------------------------

    pred, comps = forecast_c6_at_origin(
        main,
        weighted_platt,
        history,
        items,
        store,
        state,
        calendar,
        origin,
        main_prices,
    )

    check_prediction_order(
        comps,
        origin,
        items,
    )

    # --------------------------------------------------------
    # 4.7. Doc actual chi SAU khi du bao xong
    # --------------------------------------------------------

    holdout = load_store(
        data_dir,
        store,
        origin + H,
        chunk_size=chunk_size,
        start_day=origin + 1,
        item_ids=items,
    )

    require(
        holdout.item_id.to_list() == items,
        "Evaluation item order differs.",
    )

    score, scores_by_item, truth, yhat = (
        evaluate_forecast(
            pred,
            holdout,
            history,
            items,
        )
    )

    # --------------------------------------------------------
    # 4.8. Doi chieu voi Stage B
    # --------------------------------------------------------

    old_items = (
        pd.read_csv(
            stage_folder / "per_item.csv"
        )
        .set_index("item_id")
        .reindex(items)
    )

    new_items = (
        scores_by_item
        .set_index("item_id")
        .reindex(items)
    )

    require(
        not old_items.index.has_duplicates
        and not new_items.index.has_duplicates,
        "Duplicate item IDs.",
    )

    np.testing.assert_allclose(
        old_items["scale"],
        new_items["scale"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        old_items["actual_units"],
        truth.sum(axis=1),
        rtol=0,
        atol=1e-9,
    )

    require(
        int(score["rmsse_valid_items"])
        == int(baseline["valid_rmsse_items"]),
        "Different valid RMSSE population.",
    )

    np.testing.assert_allclose(
        truth.sum(),
        baseline["actual"],
        rtol=0,
        atol=0,
    )

    # --------------------------------------------------------
    # 4.9. Tinh metric C6.1
    # --------------------------------------------------------

    actual_sum = float(truth.sum())
    predicted_sum = float(yhat.sum())

    z = (
        (truth > 0)
        .astype(float)
        .T.reshape(-1)
    )

    p_new = comps[
        "probability_positive"
    ].to_numpy(dtype=float)

    p_raw = comps[
        "probability_raw"
    ].to_numpy(dtype=float)

    require(
        len(z) == len(p_new) == len(p_raw),
        "Incorrect probability count.",
    )

    record = {
        "origin": origin,
        "store_id": store,
        "n_items": len(items),

        "valid_rmsse_items":
            int(score["rmsse_valid_items"]),

        "actual": actual_sum,

        "predicted_C5":
            float(baseline["predicted_C5"]),

        "predicted_C6":
            float(baseline["predicted_C6"]),

        "predicted_C6_1": predicted_sum,

        "RMSSE_C5":
            float(baseline["RMSSE_C5"]),

        "RMSSE_C6":
            float(baseline["RMSSE_C6"]),

        "RMSSE_C6_1":
            float(score["RMSSE_mean_unweighted"]),

        "MAE_C5":
            float(baseline["MAE_C5"]),

        "MAE_C6":
            float(baseline["MAE_C6"]),

        "MAE_C6_1":
            float(score["MAE"]),

        "bias_C5_pct":
            float(baseline["bias_C5_pct"]),

        "bias_C6_pct":
            float(baseline["bias_C6_pct"]),

        "bias_C6_1_pct":
            float(
                100 * (
                    predicted_sum / actual_sum - 1
                )
            ),

        "mean_p_raw_C6_1_path":
            float(p_raw.mean()),

        "mean_p_weighted_C6_1_path":
            float(p_new.mean()),

        "positive_rate":
            float(z.mean()),

        "Brier_raw_C6_1_path":
            float(np.mean((p_raw - z) ** 2)),

        "Brier_weighted_C6_1_path":
            float(np.mean((p_new - z) ** 2)),

        "C6_1_better_RMSSE_than_C6": bool(
            score["RMSSE_mean_unweighted"]
            < baseline["RMSSE_C6"]
        ),
    }

    # --------------------------------------------------------
    # 4.10. Ket qua theo san pham
    # --------------------------------------------------------

    per_item = pd.DataFrame({
        "origin": origin,
        "store_id": store,
        "item_id": items,

        "RMSSE_C5":
            old_items["RMSSE_C5"].to_numpy(),

        "RMSSE_C6":
            old_items["RMSSE_C6"].to_numpy(),

        "RMSSE_C6_1":
            new_items["RMSSE"].to_numpy(),

        "actual_units":
            truth.sum(axis=1),

        "predicted_C5":
            old_items["predicted_C5"].to_numpy(),

        "predicted_C6":
            old_items["predicted_C6"].to_numpy(),

        "predicted_C6_1":
            yhat.sum(axis=1),
    })

    per_item["C6_1_better_than_C6"] = (
        per_item["RMSSE_C6_1"]
        < per_item["RMSSE_C6"]
    )

    record[
        "n_items_C6_1_better_RMSSE_than_C6"
    ] = int(
        per_item["C6_1_better_than_C6"].sum()
    )

    # --------------------------------------------------------
    # 4.11. Ket qua theo tuan
    # --------------------------------------------------------

    old_weeks = (
        pd.read_csv(
            stage_folder / "weekly.csv"
        )
        .set_index("week")
    )

    scale = new_items[
        "scale"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(scale)
        & (scale > 0)
    )

    weeks = []

    for start in range(0, H, 7):

        week = start // 7 + 1

        a = truth[:, start:start + 7]
        f = yhat[:, start:start + 7]

        actual_week = float(a.sum())
        forecast_week = float(f.sum())

        mse = np.mean(
            (a - f) ** 2,
            axis=1,
        )

        rmsse_week = float(
            np.sqrt(
                mse[valid] / scale[valid]
            ).mean()
        )

        weeks.append({
            "origin": origin,
            "store_id": store,
            "week": week,

            "actual": actual_week,

            "predicted_C5":
                float(
                    old_weeks.loc[
                        week,
                        "predicted_C5",
                    ]
                ),

            "predicted_C6":
                float(
                    old_weeks.loc[
                        week,
                        "predicted_C6",
                    ]
                ),

            "predicted_C6_1": forecast_week,

            "RMSSE_C5":
                float(
                    old_weeks.loc[
                        week,
                        "RMSSE_C5",
                    ]
                ),

            "RMSSE_C6":
                float(
                    old_weeks.loc[
                        week,
                        "RMSSE_C6",
                    ]
                ),

            "RMSSE_C6_1": rmsse_week,

            "MAE_C6_1":
                float(np.abs(a - f).mean()),

            "bias_C6_1_pct": (
                float(
                    100 * (
                        forecast_week / actual_week - 1
                    )
                )
                if actual_week > 0
                else np.nan
            ),
        })

    # --------------------------------------------------------
    # 4.12. Luu ket qua fold
    # --------------------------------------------------------

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        weighted_platt,
        folder / "platt_weighted.joblib",
    )

    per_item.to_csv(
        folder / "per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(weeks).to_csv(
        folder / "by_week.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred.to_csv(
        folder / "forecast_C6_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Completed duoc ghi sau khi luu du cac file
    write_json(
        done_file,
        {
            "version": VERSION,
            "origin": origin,
            "store_id": store,
            "record": record,
            "calibration": cal_diagnostics,
        },
    )

    print(
        f"{store} d_{origin}"
        f" | C6={record['RMSSE_C6']:.6f}"
        f" -> C6.1={record['RMSSE_C6_1']:.6f}"
        f" | bias={record['bias_C6_1_pct']:+.2f}%",
        flush=True,
    )

    del (
        wide,
        history,
        main,
        pred,
        comps,
        holdout,
        truth,
        yhat,
    )

    gc.collect()

    return (
        record,
        pd.DataFrame(weeks),
        per_item,
        cal_diagnostics,
    )


# ============================================================
# 5. GHEP KET QUA C7 (0.75) NEU DA CO
# ============================================================

def attach_c7_075(
    results,
    stores,
    origins,
    by_store,
):

    path = (
        results
        / "analysis"
        / "stage_C7_lambda_075"
    )

    table_path = (
        path / "02_by_store_origin.csv"
    )

    manifest_path = (
        path / "manifest.json"
    )

    if (
        not table_path.is_file()
        or not manifest_path.is_file()
    ):
        print(
            "NOTE: C7(0.75) not found.",
            flush=True,
        )
        return by_store

    manifest = read_json(manifest_path)

    require(
        np.isclose(
            float(manifest.get("lambda", -1)),
            0.75,
        ),
        "C7 folder has different lambda.",
    )

    df = pd.read_csv(table_path)

    require(
        not df[
            ["origin", "store_id"]
        ].duplicated().any(),
        "Duplicate C7 fold.",
    )

    selected = df.loc[
        df.origin.isin(origins)
        & df.store_id.isin(stores),
        [
            "origin",
            "store_id",
            "RMSSE_C7",
            "MAE_C7",
            "bias_C7_pct",
            "predicted_C7",
        ],
    ].copy()

    selected = selected.rename(columns={
        "RMSSE_C7": "RMSSE_C7_075",
        "MAE_C7": "MAE_C7_075",
        "bias_C7_pct": "bias_C7_075_pct",
        "predicted_C7": "predicted_C7_075",
    })

    require(
        len(selected) == len(by_store),
        "C7 must cover all selected folds.",
    )

    return by_store.merge(
        selected,
        on=["origin", "store_id"],
        validate="one_to_one",
    )


# ============================================================
# 6. CHUONG TRINH CHINH
# ============================================================

def main(args):

    require(
        args.stores is None
        or len(set(args.stores)) == len(args.stores),
        "Duplicate stores.",
    )

    stores = sorted(
        args.stores or STORE_CODES
    )

    origins = sorted(
        set(args.origins)
    )

    require(
        stores
        and set(stores).issubset(STORE_CODES),
        "Invalid stores.",
    )

    require(
        origins
        and set(origins).issubset(FOLDS),
        "Use d_1829 and d_1857 only.",
    )

    # --------------------------------------------------------
    # 6.1. Cau hinh thu muc
    # --------------------------------------------------------

    config = yaml.safe_load(
        (
            ROOT / "configs" / "config.yaml"
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
        manifest_b.get("status") == "complete"
        and set(
            manifest_b.get("origins_requested", [])
        ) == set(FOLDS)
        and set(
            manifest_b.get("stores", [])
        ) == set(STORE_CODES),
        "Full Stage B must be complete.",
    )

    c6_tag = manifest_b["C6_batch"]

    output = (
        results
        / "analysis"
        / OUTPUT_DIR
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_files(
        data_dir,
        use_prices=True,
    )

    calendar = load_calendar(data_dir)

    rows = []
    week_parts = []
    item_parts = []
    calibration = []

    # --------------------------------------------------------
    # 6.2. Chay tung store / origin
    # --------------------------------------------------------

    for store in stores:

        for origin in origins:

            row, w, i, cal = evaluate_one(
                store,
                origin,
                data_dir,
                config["chunk_size"],
                calendar,
                results,
                stage_b,
                c6_tag,
                output,
            )

            rows.append(row)
            week_parts.append(w)
            item_parts.append(i)

            calibration.append({
                "origin": origin,
                "store_id": store,
                **cal,
            })

    # --------------------------------------------------------
    # 6.3. Tong hop theo store-origin
    # --------------------------------------------------------

    by_store = (
        pd.DataFrame(rows)
        .sort_values(["origin", "store_id"])
        .reset_index(drop=True)
    )

    by_store = attach_c7_075(
        results,
        stores,
        origins,
        by_store,
    )

    methods = [
        "C5",
        "C6",
        "C6_1",
    ]

    if "RMSSE_C7_075" in by_store.columns:
        methods.append("C7_075")

    # --------------------------------------------------------
    # 6.4. Tong hop theo origin
    # --------------------------------------------------------

    origin_rows = []

    for origin, df in by_store.groupby(
        "origin",
        sort=True,
    ):

        total = float(
            df["actual"].sum()
        )

        weights = df[
            "valid_rmsse_items"
        ].to_numpy(dtype=float)

        entry = {
            "origin": int(origin),
            "n_stores": len(df),
            "actual": total,

            "n_items": int(
                df["n_items"].sum()
            ),

            "valid_rmsse_items": int(
                weights.sum()
            ),
        }

        for m in methods:

            predicted = float(
                df[f"predicted_{m}"].sum()
            )

            entry[f"predicted_{m}"] = predicted

            entry[f"bias_{m}_pct"] = float(
                100 * (
                    predicted / total - 1
                )
            )

            entry[f"RMSSE_{m}"] = float(
                np.average(
                    df[f"RMSSE_{m}"],
                    weights=weights,
                )
            )

            entry[f"MAE_{m}"] = float(
                np.average(
                    df[f"MAE_{m}"],
                    weights=df["n_items"],
                )
            )

        origin_rows.append(entry)

    by_origin = (
        pd.DataFrame(origin_rows)
        .sort_values("origin")
    )

    # --------------------------------------------------------
    # 6.5. Tong hop trung binh
    # --------------------------------------------------------

    summary = {
        "origins": len(origins),
        "stores": len(stores),

        "first_14_day_weight":
            FIRST_HALF_WEIGHT,

        "last_14_day_weight":
            SECOND_HALF_WEIGHT,
    }

    for m in methods:

        summary[f"mean_RMSSE_{m}"] = float(
            by_origin[
                f"RMSSE_{m}"
            ].mean()
        )

        summary[f"mean_MAE_{m}"] = float(
            by_origin[
                f"MAE_{m}"
            ].mean()
        )

        summary[
            f"mean_abs_store_bias_{m}_pct"
        ] = float(
            by_store[
                f"bias_{m}_pct"
            ].abs().mean()
        )

    summary[
        "C6_1_better_than_C6_origin_count"
    ] = int(
        (
            by_origin["RMSSE_C6_1"]
            < by_origin["RMSSE_C6"]
        ).sum()
    )

    summary[
        "C6_1_better_than_C6_store_origin_count"
    ] = int(
        (
            by_store["RMSSE_C6_1"]
            < by_store["RMSSE_C6"]
        ).sum()
    )

    # --------------------------------------------------------
    # 6.6. Xuat bao cao
    # --------------------------------------------------------

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

    pd.DataFrame([summary]).to_csv(
        output / "03_weighted_platt_summary.csv",
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

    pd.DataFrame(
        calibration
    ).to_csv(
        output / "06_calibration_by_store_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6.7. Luu manifest
    # --------------------------------------------------------

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
            "version": VERSION,

            "origins": origins,
            "stores": stores,

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "stage_b_source":
                "stage_B_C5_C6_full",

            "C6_batch": c6_tag,

            "lightgbm_retrained": False,
            "platt_retrained": True,

            "platt_calibration_window_days": 28,

            "raw_weight_first_14_days":
                FIRST_HALF_WEIGHT,

            "raw_weight_last_14_days":
                SECOND_HALF_WEIGHT,

            "normalization":
                "sample weights divided by their average",

            "regularization_C": 1.0,

            "used_d1885_for_selection": False,
            "test_performed": False,
        },
    )

    # --------------------------------------------------------
    # 6.8. Bao cao nhanh tren terminal
    # --------------------------------------------------------

    print(
        "\n=== C6.1: Weighted Platt 1x/2x ==="
    )

    print(
        by_origin.to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )

    print(
        "\n=== MEAN METRICS ==="
    )

    print(
        pd.DataFrame([summary]).T.to_string(
            header=False
        )
    )

    print(
        f"\nOutput: {output}"
    )


# ============================================================
# 7. CHAY
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

    main(
        parser.parse_args()
    )
