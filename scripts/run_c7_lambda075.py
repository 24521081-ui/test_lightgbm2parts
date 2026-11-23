
"""C7: Partial Platt Calibration, lambda = 0.75.

Su dung lai model va Platt Scaling cua Stage B.
Danh gia tren hai goc d_1829 va d_1857.
KHONG train lai LightGBM.
KHONG train lai Platt Scaling.
KHONG su dung Validation d_1885 hoac Test de chon lambda.
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
from src.training import verify_saved_models

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


# ======================================================
# 1. CAU HINH C7
# ======================================================

LAM = 0.75

B_VERSION = "B1_20261121"
C7_VERSION = "C7_075_v1"


# ======================================================
# 2. NAP MO HINH DA TRAIN O STAGE B
# ======================================================


def load_model(
    results,
    stage_b,
    c6_tag,
    store,
    origin,
    items,
):

    cached = (
        stage_b
        / "models"
        / store
        / f"d_{origin}"
        / "model.joblib"
    )

    reused = (
        results
        / "models"
        / f"{c6_tag}_{store}"
        / "aux_model.joblib"
    )

    if cached.is_file():
        path = cached

    elif origin == 1857 and reused.is_file():
        path = reused

    else:
        raise FileNotFoundError(
            f"Thieu model {store} d_{origin}. "
            "Can chay Stage B Full truoc."
        )

    model = verify_saved_models(path)

    require(
        model["origin_day"] == origin
        and model["categories"] == items,
        f"Model khong dung goc/san pham: {path}",
    )

    require(
        all(
            x in model["features"]
            for x in ["sell_price", *EXTRAS]
        ),
        f"Model thieu dac trung gia: {path}",
    )

    return model


# ======================================================
# 3. DU BAO DE QUY C7 VOI LAMBDA = 0.75
# ======================================================

def predict_c7(bundle, platt, matrix, items, store, state, calendar, origin, prices):
    """Forecast recursively using this entrypoint's frozen lambda."""
    return predict_blended(
        bundle, platt, matrix, items, store, state, calendar, origin, prices,
        LAM,
    )


# ======================================================
# 4. CHAY VA DANH GIA MOT CUA HANG
# ======================================================

def run_store(
    store,
    origin,
    data_dir,
    config,
    calendar,
    results,
    stage_b,
    c6_tag,
    out,
):

    source = (
        stage_b
        / "folds"
        / f"d_{origin}"
        / store
    )

    old = read_json(
        source / "completed.json"
    )

    require(
        old["version"] == B_VERSION
        and old["origin"] == origin,
        f"Stage B fold khong hop le: {store} d_{origin}",
    )

    old_row = old["record"]

    folder = (
        out
        / "folds"
        / f"d_{origin}"
        / store
    )

    done = folder / "completed.json"

    # Tai su dung ket qua C7 neu da hoan thanh
    if done.is_file():

        saved = read_json(done)

        require(
            saved["version"] == C7_VERSION
            and saved["lambda"] == LAM,
            "C7 cache khong tuong thich.",
        )

        require(
            all(
                (folder / f).is_file()
                for f in [
                    "per_item.csv",
                    "by_week.csv",
                ]
            ),
            "C7 cache thieu du lieu.",
        )

        print_validation_summary(
            saved["record"], f"C7 lambda={LAM:.2f}", origin, store, folder, status="REUSE"
        )

        return (
            saved["record"],
            pd.read_csv(folder / "by_week.csv"),
            pd.read_csv(folder / "per_item.csv"),
        )

    # --------------------------------------------------
    # 4.1. DOC DU LIEU LICH SU
    # --------------------------------------------------

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
        f"{store}: so luong series khong dung.",
    )

    state = str(
        wide.state_id.iloc[0]
    )

    matrix = history_matrix(
        wide,
        origin,
    )

    # --------------------------------------------------
    # 4.2. NAP LIGHTGBM VA PLATT DA CO
    # --------------------------------------------------

    bundle = load_model(
        results,
        stage_b,
        c6_tag,
        store,
        origin,
        items,
    )

    path_cal = source / "platt.joblib"

    require(
        path_cal.is_file(),
        f"Thieu Platt model: {path_cal}",
    )

    platt = joblib.load(path_cal)

    # --------------------------------------------------
    # 4.3. DAC TRUNG GIA TAI GOC DU BAO
    # --------------------------------------------------

    weekly_prices = build_weekly_features(
        load_prices(data_dir, store),
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

    # --------------------------------------------------
    # 4.4. DU BAO C7
    # --------------------------------------------------

    forecast, comps = predict_c7(
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

    # Chi doc nhan thuc te sau khi du bao xong
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
        "Holdout item order khong khop.",
    )

    # --------------------------------------------------
    # 4.5. TINH RMSSE VA MAE
    # --------------------------------------------------

    score, item_scores, truth, yhat = (
        evaluate_forecast(
            forecast,
            actual,
            matrix,
            items,
        )
    )

    # Lay RMSSE C5/C6 da chay tu Stage B
    base = pd.read_csv(
        source / "per_item.csv"
    ).set_index(
        "item_id"
    ).reindex(items)

    current = item_scores.set_index(
        "item_id"
    ).reindex(items)

    np.testing.assert_allclose(
        base["scale"],
        current["scale"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        base["actual_units"],
        truth.sum(axis=1),
        rtol=0,
        atol=1e-9,
    )

    require(
        int(score["rmsse_valid_items"])
        == int(old_row["valid_rmsse_items"]),
        "So chuoi RMSSE hop le khac Stage B.",
    )

    # --------------------------------------------------
    # 4.6. KET QUA THEO SAN PHAM
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 4.7. KET QUA THEO CUA HANG
    # --------------------------------------------------

    total = float(truth.sum())
    pred_total = float(yhat.sum())

    z = (
        (truth > 0)
        .astype(float)
        .T.reshape(-1)
    )

    p = comps["p_C7"].to_numpy()

    row = {
        "origin": origin,
        "store_id": store,
        "lambda": LAM,
        "n_items": len(items),

        "valid_rmsse_items":
            int(score["rmsse_valid_items"]),

        "actual": total,

        **{
            f"predicted_{m}":
                float(old_row[f"predicted_{m}"])

            for m in ["C5", "C6"]
        },

        "predicted_C7": pred_total,

        **{
            f"RMSSE_{m}":
                float(old_row[f"RMSSE_{m}"])

            for m in ["C5", "C6"]
        },

        "RMSSE_C7":
            float(score["RMSSE_mean_unweighted"]),

        **{
            f"MAE_{m}":
                float(old_row[f"MAE_{m}"])

            for m in ["C5", "C6"]
        },

        "MAE_C7":
            float(score["MAE"]),

        **{
            f"bias_{m}_pct":
                float(old_row[f"bias_{m}_pct"])

            for m in ["C5", "C6"]
        },

        "bias_C7_pct":
            100 * (pred_total / total - 1),

        "Brier_C7_path":
            float(np.mean((p - z) ** 2)),

        "mean_probability_C7":
            float(p.mean()),

        "positive_rate":
            float(z.mean()),

        "n_series_C7_better_C6":
            int(item_out["C7_better_C6"].sum()),
    }

    np.testing.assert_allclose(
        total,
        old_row["actual"],
        rtol=0,
        atol=0,
    )

    # --------------------------------------------------
    # 4.8. KET QUA THEO TUNG TUAN
    # --------------------------------------------------

    ref_week = pd.read_csv(
        source / "weekly.csv"
    ).set_index("week")

    scale = current["scale"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(scale)
        & (scale > 0)
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

        actual_week = float(a.sum())
        forecast_week = float(f.sum())

        week_rows.append({
            "origin": origin,
            "store_id": store,
            "week": week,
            "actual": actual_week,

            **{
                f"predicted_{m}":
                    float(
                        ref_week.loc[
                            week,
                            f"predicted_{m}",
                        ]
                    )

                for m in ["C5", "C6"]
            },

            "predicted_C7": forecast_week,

            **{
                f"RMSSE_{m}":
                    float(
                        ref_week.loc[
                            week,
                            f"RMSSE_{m}",
                        ]
                    )

                for m in ["C5", "C6"]
            },

            "RMSSE_C7": float(
                np.sqrt(
                    np.mean(
                        (a - f) ** 2,
                        axis=1,
                    )[valid]
                    / scale[valid]
                ).mean()
            ),

            "MAE_C7":
                float(np.abs(a - f).mean()),

            "bias_C7_pct": (
                100 * (
                    forecast_week
                    / actual_week
                    - 1
                )
                if actual_week
                else np.nan
            ),
        })

    # --------------------------------------------------
    # 4.9. LUU KET QUA
    # --------------------------------------------------

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    item_out.to_csv(
        folder / "per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(week_rows).to_csv(
        folder / "by_week.csv",
        index=False,
        encoding="utf-8-sig",
    )

    forecast.to_csv(
        folder / "forecast_C7.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        folder / "completed.json",
        {
            "version": C7_VERSION,
            "lambda": LAM,
            "origin": origin,
            "store_id": store,
            "record": row,
        },
    )

    print_validation_summary(row, f"C7 lambda={LAM:.2f}", origin, store, folder)

    return (
        row,
        pd.DataFrame(week_rows),
        item_out,
    )


# ======================================================
# 5. TONG HOP TOAN BO 10 CUA HANG
# ======================================================

def main(args):

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
        manifest_b["status"] == "complete"
        and set(
            manifest_b["origins_requested"]
        ) == set(FOLDS),
        "Can chay day du Stage B truoc.",
    )

    c6_tag = manifest_b["C6_batch"]

    stores = sorted(
        args.stores or STORE_CODES
    )

    origins = sorted(
        set(args.origins)
    )

    require(
        stores
        and set(stores).issubset(STORE_CODES),
        "Store khong hop le.",
    )

    require(
        origins
        and set(origins).issubset(FOLDS),
        "Chi dung d_1829 va d_1857.",
    )

    check_files(
        data_dir,
        use_prices=True,
    )

    calendar = load_calendar(data_dir)

    out = (
        results
        / "analysis"
        / "stage_C7_lambda_075"
    )

    records = []
    weeks = []
    items = []

    # --------------------------------------------------
    # CHAY TUNG CUA HANG VA GOC DU BAO
    # --------------------------------------------------

    for store in stores:

        for origin in origins:

            r, w, i = run_store(
                store,
                origin,
                data_dir,
                config,
                calendar,
                results,
                stage_b,
                c6_tag,
                out,
            )

            records.append(r)
            weeks.append(w)
            items.append(i)

    by_store = (
        pd.DataFrame(records)
        .sort_values(
            ["origin", "store_id"]
        )
    )

    # --------------------------------------------------
    # TONG HOP THEO GOC
    # --------------------------------------------------

    by_origin_rows = []

    for origin, df in by_store.groupby(
        "origin"
    ):

        total = df["actual"].sum()

        weighted = df[
            "valid_rmsse_items"
        ].to_numpy()

        entry = {
            "origin": origin,
            "actual": total,
            "n_stores": len(df),
        }

        for m in ["C5", "C6", "C7"]:

            predicted = df[
                f"predicted_{m}"
            ].sum()

            entry[f"RMSSE_{m}"] = np.average(
                df[f"RMSSE_{m}"],
                weights=weighted,
            )

            entry[f"MAE_{m}"] = np.average(
                df[f"MAE_{m}"],
                weights=df["n_items"],
            )

            entry[f"predicted_{m}"] = predicted

            entry[f"bias_{m}_pct"] = (
                100 * (
                    predicted / total - 1
                )
            )

        by_origin_rows.append(entry)

    by_origin = (
        pd.DataFrame(by_origin_rows)
        .sort_values("origin")
    )

    # --------------------------------------------------
    # BANG TONG HOP LAMBDA 0.75
    # --------------------------------------------------

    summary = pd.DataFrame([
        {
            "lambda": LAM,
            "origins": len(by_origin),
            "stores": len(stores),

            **{
                f"mean_RMSSE_{m}":
                    by_origin[
                        f"RMSSE_{m}"
                    ].mean()

                for m in ["C5", "C6", "C7"]
            },

            **{
                f"mean_MAE_{m}":
                    by_origin[
                        f"MAE_{m}"
                    ].mean()

                for m in ["C5", "C6", "C7"]
            },

            **{
                f"mean_abs_store_bias_{m}_pct":
                    by_store[
                        f"bias_{m}_pct"
                    ].abs().mean()

                for m in ["C5", "C6", "C7"]
            },

            "C7_beats_C6_origin_count": int(
                (
                    by_origin["RMSSE_C7"]
                    < by_origin["RMSSE_C6"]
                ).sum()
            ),

            "C7_beats_C6_store_origin_count": int(
                (
                    by_store["RMSSE_C7"]
                    < by_store["RMSSE_C6"]
                ).sum()
            ),
        }
    ])

    # --------------------------------------------------
    # XUAT CAC FILE BAO CAO
    # --------------------------------------------------

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_origin.to_csv(
        out / "01_by_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_store.to_csv(
        out / "02_by_store_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        out / "03_lambda_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        weeks,
        ignore_index=True,
    ).to_csv(
        out / "04_by_week_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        items,
        ignore_index=True,
    ).to_csv(
        out / "05_per_item.csv",
        index=False,
        encoding="utf-8-sig",
    )

    status = (
        "complete"
        if (
            set(stores) == set(STORE_CODES)
            and set(origins) == set(FOLDS)
        )
        else "partial"
    )

    write_json(
        out / "manifest.json",
        {
            "status": status,
            "lambda": LAM,
            "version": C7_VERSION,
            "origins": origins,
            "stores": stores,
            "C6_batch": c6_tag,

            "lightgbm_retrained": False,
            "platt_retrained": False,

            "d_1885_not_used_for_lambda_selection": True,
            "test_performed": False,
        },
    )

    # --------------------------------------------------
    # IN TONG QUAN
    # --------------------------------------------------

    print(
        "\n=== C5 vs C6 vs C7 (lambda = 0.75) ==="
    )

    print(
        by_origin[
            [
                "origin",
                "RMSSE_C5",
                "RMSSE_C6",
                "RMSSE_C7",
                "bias_C5_pct",
                "bias_C6_pct",
                "bias_C7_pct",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nKet qua: {out}"
    )


# ======================================================
# 6. CHAY CHUONG TRINH
# ======================================================

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
