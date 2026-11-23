
"""C5: C4 + hai dac trung bien dong gia qua khu.

- Classifier: 48 la, 300 cay.
- Regressor: 31 la, 200 cay.
- Giu nguyen cac tham so C4.
- Them price_change_1w va price_ratio_4w.
- Train lai ca hai thanh phan.
- Validation d_1886 den d_1913.
- Khong dung gia hoac luong ban tuong lai.
"""

import copy
import gc
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data_loader import (
    check_files,
    load_calendar,
    load_prices,
    load_store,
)

from src.feature_engineering import (
    next_day_features,
    prepare_training_data,
)

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

from src.training import train_model

from run_c4_with_prices_lagged import (
    C0_TAG,
    append_csv,
    check_join,
    diagnostics,
    previous_completed_week_prices,
    read_json,
    require,
)


LOG = logging.getLogger("m5")

C4_TAG = (
    "validation_C4_full_10_stores_"
    "20261121T133611960516Z"
)

EXTRAS = [
    "price_change_1w",
    "price_ratio_4w",
]

KEYS = [
    "store_id",
    "item_id",
    "wm_yr_wk",
]

FILES = [
    "lightgbm_two_part.csv",
    "seasonal_naive.csv",
    "actual_vs_predictions.csv",
]


# ==========================================================
# 1. TAO DAC TRUNG GIA THEO TUAN
# ==========================================================

def build_weekly_features(prices, calendar):

    weeks = (
        calendar.groupby("wm_yr_wk")["date"]
        .min()
        .sort_values()
        .index
        .to_list()
    )

    ordinal = {
        week: index
        for index, week in enumerate(weeks)
    }

    prices = (
        prices.sort_values(KEYS)
        .reset_index(drop=True)
        .copy()
    )

    prices["week_pos"] = (
        prices["wm_yr_wk"].map(ordinal)
    )

    require(
        prices["week_pos"].notna().all(),
        "Gia chua tuan khong co trong calendar.",
    )

    group = prices.groupby(
        ["store_id", "item_id"],
        sort=False,
    )

    p = prices["sell_price"]

    p1 = group["sell_price"].shift(1)
    p2 = group["sell_price"].shift(2)
    p3 = group["sell_price"].shift(3)
    p4 = group["sell_price"].shift(4)

    contiguous_1 = (
        prices["week_pos"]
        .sub(group["week_pos"].shift(1))
        .eq(1)
    )

    contiguous_4 = (
        prices["week_pos"]
        .sub(group["week_pos"].shift(4))
        .eq(4)
    )

    # Bien 1: thay doi gia so voi tuan truoc.
    prices["price_change_1w"] = (
        (p / p1 - 1)
        .where(contiguous_1)
        .astype("float32")
    )

    # Bien 2: gia so voi trung binh 4 tuan truoc.
    prices["price_ratio_4w"] = (
        (
            p / ((p1 + p2 + p3 + p4) / 4)
        )
        .where(contiguous_4)
        .astype("float32")
    )

    require(
        not np.isinf(
            prices[EXTRAS].to_numpy()
        ).any(),
        "Bien gia chua gia tri vo han.",
    )

    # Dich thong tin cua tuan W sang tuan W+1.
    result = previous_completed_week_prices(
        prices[
            KEYS + ["sell_price"] + EXTRAS
        ],
        calendar,
    )

    return result


# ==========================================================
# 2. GHEP CAC DAC TRUNG GIA VAO TRAIN
# ==========================================================

def attach_price_features(
    history,
    week_features,
    calendar,
):

    day_to_week = (
        calendar.assign(
            day=calendar["d"]
            .str[2:]
            .astype(int)
        )
        .set_index("day")["wm_yr_wk"]
    )

    n = len(history)

    item_categories = (
        history["item_id"].cat.categories
    )

    store_categories = (
        history["store_id"].cat.categories
    )

    history["wm_yr_wk"] = (
        history["day"]
        .map(day_to_week)
        .astype("int32")
    )

    history = (
        history.merge(
            week_features[KEYS + EXTRAS],
            on=KEYS,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        .drop(columns=["wm_yr_wk"])
    )

    require(
        len(history) == n,
        "Merge bien gia lam thay doi so dong.",
    )

    # Khoi phuc categorical sau khi merge.
    history["item_id"] = pd.Categorical(
        history["item_id"],
        categories=item_categories,
    )

    history["store_id"] = pd.Categorical(
        history["store_id"],
        categories=store_categories,
    )

    return history


def make_training_data(history):

    train = prepare_training_data(history)

    base_index = train["X"].index

    train["X"] = train["X"].copy()

    for feature in EXTRAS:

        train["X"][feature] = (
            history.loc[
                base_index,
                feature,
            ]
            .to_numpy(dtype="float32")
        )

        train["features"].append(feature)

    require(
        train["X"].columns.to_list()
        == train["features"],
        "Thu tu feature Train bi sai.",
    )

    require(
        not np.isinf(
            train["X"][EXTRAS].to_numpy()
        ).any(),
        "Bien gia Train khong hop le.",
    )

    return train


# ==========================================================
# 3. DU BAO DE QUY CO THEM HAI BIEN GIA
# ==========================================================

def forecast_c5(
    bundle,
    history,
    items,
    store,
    state,
    calendar,
    origin,
    price_arrays,
):

    model = bundle["model"]

    features = bundle["features"]
    categories = bundle["categories"]

    cal = calendar.set_index(
        "d",
        verify_integrity=True,
    )

    temporary = np.asarray(
        history[:, -28:],
        dtype="float32",
    ).copy()

    predictions = []
    components = []

    require(
        temporary.shape == (len(items), 28),
        "Lich su forecast sai kich thuoc.",
    )

    for step in range(1, 29):

        day = f"d_{origin + step}"

        X = next_day_features(
            temporary,
            items,
            cal.loc[day],
            state,
            categories,
            last_prices=price_arrays["sell_price"],
        )

        # Giu cac dac trung gia da biet
        # tai origin trong 28 ngay.
        for feature in EXTRAS:
            X[feature] = price_arrays[feature]

        require(
            X[features].columns.to_list()
            == features,
            "Thieu feature khi du bao.",
        )

        p, mu = model.predict_components(
            X[features]
        )

        yhat = np.maximum(
            p * mu,
            0,
        )

        require(
            np.isfinite(yhat).all(),
            "Du bao C5 khong huu han.",
        )

        predictions.append(
            pd.DataFrame({
                "model": "lightgbm_two_part",
                "split": "validation",
                "origin_day": f"d_{origin}",
                "item_id": items,
                "store_id": store,
                "target_day": day,
                "horizon": step,
                "y_pred": yhat,
            })
        )

        components.append(
            pd.DataFrame({
                "item_id": items,
                "store_id": store,
                "target_day": day,
                "probability_positive": p,
                "conditional_units": mu,
            })
        )

        # Du bao ngay truoc tro thanh lich su
        # de tao lag/rolling cho ngay ke tiep.
        temporary = np.column_stack(
            (
                temporary[:, 1:],
                yhat,
            )
        )

        LOG.debug(
            "C5 %s | forecast %d/28",
            store,
            step,
        )

    predicted = pd.concat(
        predictions,
        ignore_index=True,
    )[FORECAST_COLUMNS]

    validate_forecasts(predicted)

    return (
        predicted,
        pd.concat(
            components,
            ignore_index=True,
        ),
    )


# ==========================================================
# 4. CHUONG TRINH CHINH
# ==========================================================

def main():

    original = yaml.safe_load(
        (
            ROOT / "configs/config.yaml"
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    out = ROOT / original["results_dir"]

    c0 = read_json(
        out
        / "metrics"
        / C0_TAG
        / "validation_manifest.json"
    )

    c4_dir = out / "metrics" / C4_TAG

    c4 = read_json(
        c4_dir / "manifest.json"
    )

    require(
        c0["status"] == "completed"
        and c4["status"] == "completed",
        "C0/C4 chua completed.",
    )

    require(
        original == c0["config"]
        and not original["use_prices"],
        "config.yaml phai giu C0.",
    )

    expected_c4 = copy.deepcopy(
        original
    )

    expected_c4["classifier"].update(
        num_leaves=48,
        n_estimators=300,
    )

    expected_c4["use_prices"] = True

    require(
        c4["config"] == expected_c4
        and c4["price_policy"]
        == "previous_completed_week",
        "C4 khong dung thiet ke.",
    )

    require(
        c4["n_items"] == 30490
        and c4["rows_per_configuration"]
        == 853720,
        "C4 chua du chuoi.",
    )

    for key in (
        "same_keys_as_C3",
        "same_actuals_as_C3",
        "same_RMSSE_scales",
    ):
        require(
            c4[key],
            f"C4 chua xac nhan {key}",
        )

    c0_sources = {
        r["store_id"]: r
        for r in c0["sources"]
    }

    c4_sources = {
        r["store_id"]: r
        for r in c4["sources"]
    }

    require(
        len(c0_sources) == 10
        and len(c4_sources) == 10
        and set(c0_sources)
        == set(c4_sources),
        "Thieu cua hang C0/C4.",
    )

    # ------------------------------------------------------
    # 5. DOC DU LIEU
    # ------------------------------------------------------

    data_dir = (
        ROOT / original["data_dir"]
    )

    check_files(
        data_dir,
        use_prices=True,
    )

    calendar = load_calendar(data_dir)

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    tag = (
        f"validation_C5_full_10_stores_{run_id}"
    )

    metrics_root = (
        out / "metrics" / tag
    )

    aggregate = (
        out / "validation" / tag
    )

    metrics_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    aggregate.mkdir(
        parents=True,
        exist_ok=False,
    )

    save_json(
        {
            "status": "running",
            "configuration": "C5",
            "config": expected_c4,
            "c4_batch": C4_TAG,
            "features_added": EXTRAS,
            "price_policy": "previous_completed_week",
        },
        metrics_root / "manifest.json",
    )

    store_rows = []
    score_rows = []
    sources = []

    # ------------------------------------------------------
    # 6. TRAIN VA VALIDATION TUNG CUA HANG
    # ------------------------------------------------------

    for store in c0_sources:

        old = c4_sources[store]

        source_path = Path(
            c0_sources[store][
                "source_model_path"
            ]
        )

        item_order = read_json(
            source_path.parent
            / "metadata.json"
        )["selected_items"]

        LOG.info(
            "C5 START | %s",
            store,
        )

        wide = load_store(
            data_dir,
            store,
            1885,
            chunk_size=original["chunk_size"],
            item_ids=item_order,
        )

        items = wide.item_id.to_list()

        require(
            items == item_order
            and len(items) == 3049,
            "Danh sach item khong khop.",
        )

        state = wide.state_id.iloc[0]

        matrix = history_matrix(
            wide,
            1885,
        )

        _, selected, component = (
            training_patterns(
                matrix,
                items,
            )
        )

        # --------------------------------------------------
        # 7. TAO FEATURE GIA
        # --------------------------------------------------

        prices = load_prices(
            data_dir,
            store,
        )

        week_features = (
            build_weekly_features(
                prices,
                calendar,
            )
        )

        history = prepare_history(
            wide,
            calendar,
            1885,
            week_features[
                KEYS + ["sell_price"]
            ],
        )

        history = attach_price_features(
            history,
            week_features,
            calendar,
        )

        origin_frame = (
            history.loc[
                history.day == 1885
            ]
            .set_index("item_id")
            .reindex(items)
        )

        origin_arrays = {
            name: origin_frame[name]
            .to_numpy(dtype="float32")

            for name in (
                ["sell_price"] + EXTRAS
            )
        }

        # --------------------------------------------------
        # 8. KIEM TRA GIA TAI ORIGIN
        # --------------------------------------------------

        origin_week = int(
            calendar.loc[
                calendar.d.eq("d_1885"),
                "wm_yr_wk",
            ].iloc[0]
        )

        prev_week = int(
            calendar.loc[
                calendar.d.eq("d_1883"),
                "wm_yr_wk",
            ].iloc[0]
        )

        require(
            origin_week == 11609
            and prev_week == 11608,
            "Moc tuan Validation sai.",
        )

        check = (
            week_features.loc[
                week_features.wm_yr_wk.eq(
                    origin_week
                )
            ]
            .set_index("item_id")
            .reindex(items)
        )

        for name, array in (
            origin_arrays.items()
        ):

            np.testing.assert_allclose(
                array,
                check[name].to_numpy(
                    dtype="float32"
                ),
                rtol=0,
                atol=0,
                equal_nan=True,
            )

        require(
            np.isfinite(
                origin_arrays["sell_price"]
            ).all(),
            "Gia origin bi thieu.",
        )

        LOG.info(
            "C5 PRICE CHECK | %s "
            "| week=%d | missing change=%d "
            "| missing ratio=%d",
            store,
            prev_week,
            np.isnan(
                origin_arrays[EXTRAS[0]]
            ).sum(),
            np.isnan(
                origin_arrays[EXTRAS[1]]
            ).sum(),
        )

        del prices, week_features, wide

        # --------------------------------------------------
        # 9. HUAN LUYEN CLASSIFIER VA REGRESSOR
        # --------------------------------------------------

        train = make_training_data(
            history
        )

        require(
            train["fit_end_day"] == 1885
            and len(train["X"]) == 5661993,
            "So dong Train sai.",
        )

        del history

        gc.collect()

        path_tag = (
            f"validation_C5_full_{store}_{run_id}"
        )

        paths = {
            name: out / name / path_tag
            for name in (
                "models",
                "metrics",
                "validation",
            )
        }

        params = {
            name: {
                **expected_c4[name],
                "importance_type": "gain",
            }
            for name in (
                "classifier",
                "regressor",
            )
        }

        bundle, info = train_model(
            train,
            params,
            paths["models"]
            / "model.joblib",
        )

        require(
            bundle["features"]
            == train["features"],
            "Feature model khong khop.",
        )

        del train

        gc.collect()

        paths["validation"].mkdir(
            parents=True,
            exist_ok=False,
        )

        paths["metrics"].mkdir(
            parents=True,
            exist_ok=False,
        )

        # --------------------------------------------------
        # 10. DU BAO DE QUY 28 NGAY
        # --------------------------------------------------

        prediction, components = (
            forecast_c5(
                bundle,
                matrix,
                items,
                store,
                state,
                calendar,
                1885,
                origin_arrays,
            )
        )

        export_forecasts(
            prediction,
            paths["validation"]
            / "lightgbm_two_part.csv",
        )

        # Chi doc Validation thuc te
        # SAU KHI du bao da hoan tat.
        actual = load_store(
            data_dir,
            store,
            1913,
            start_day=1886,
            item_ids=items,
            chunk_size=original["chunk_size"],
        )

        baseline = pd.read_csv(
            Path(
                old["actual_vs_predictions"]
            ).parent
            / "seasonal_naive.csv",
            float_precision="round_trip",
        )

        validate_forecasts(baseline)

        export_forecasts(
            baseline,
            paths["validation"]
            / "seasonal_naive.csv",
        )

        context = dict(
            config=expected_c4,
            configuration_name="C5",
            store_id=store,
            scope="full",
            split="validation",
            origin_day=1885,
            run_id=run_id,
            batch_tag=tag,
            illustration_items=selected,
            component_item=component,
            save_figures=False,
            mode="validation",
            status="completed",
            regressor_retrained=True,
            forecast_performed=True,
            evaluation_performed=True,
            price_policy="previous_completed_week",
            features_added=EXTRAS,
        )

        scores = evaluate_and_export(
            prediction,
            baseline,
            actual,
            matrix,
            items,
            bundle,
            info,
            paths,
            context,
            components=components,
        )

        current = pd.read_csv(
            paths["validation"]
            / "actual_vs_predictions.csv",
            float_precision="round_trip",
        )

        reference = pd.read_csv(
            old["actual_vs_predictions"],
            float_precision="round_trip",
        )

        # --------------------------------------------------
        # 11. KIEM TRA C4 VA C5 CUNG DU LIEU
        # --------------------------------------------------

        check_join(
            reference,
            current,
            store,
        )

        scales4 = pd.read_csv(
            Path(
                old["summary"]
            ).parent
            / "lightgbm_two_part_per_item.csv",
            float_precision="round_trip",
        ).set_index("item_id")

        scales5 = pd.read_csv(
            paths["metrics"]
            / "lightgbm_two_part_per_item.csv",
            float_precision="round_trip",
        ).set_index("item_id")

        np.testing.assert_allclose(
            scales4.loc[items, "scale"],
            scales5.loc[items, "scale"],
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        row = {
            "store_id": store
        }

        for name, detail in (
            ("C4", reference),
            ("C5", current),
        ):
            row.update({
                f"{key}_{name}": value
                for key, value in diagnostics(
                    detail
                ).items()
            })

        store_rows.append(row)

        score = scores.loc[
            scores.model.eq(
                "lightgbm_two_part"
            )
        ]

        require(
            len(score) == 1,
            "Khong co RMSSE C5.",
        )

        score_rows.append(
            score.iloc[0].to_dict()
        )

        # Gop ket qua tung cua hang.
        for filename in FILES:
            append_csv(
                paths["validation"] / filename,
                aggregate / filename,
            )

        sources.append({
            "store_id": store,
            "run_tag": path_tag,

            "model_path": str(
                (
                    paths["models"]
                    / "model.joblib"
                ).resolve()
            ),

            "actual_vs_predictions": str(
                (
                    paths["validation"]
                    / "actual_vs_predictions.csv"
                ).resolve()
            ),

            "summary": str(
                (
                    paths["metrics"]
                    / "summary.csv"
                ).resolve()
            ),
        })

        LOG.info(
            "C5 COMPLETE | %s | RMSSE %.6f",
            store,
            score.iloc[0][
                "RMSSE_mean_unweighted"
            ],
        )

        del (
            matrix,
            bundle,
            prediction,
            components,
            actual,
            baseline,
            current,
            reference,
            scales4,
            scales5,
        )

        gc.collect()

    # ------------------------------------------------------
    # 12. TONG HOP KET QUA
    # ------------------------------------------------------

    require(
        len(sources) == 10,
        "Chua du 10 cua hang.",
    )

    stores = pd.DataFrame(
        store_rows
    )

    stores.to_csv(
        metrics_root
        / "comparison_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics5 = pd.DataFrame(
        score_rows
    )

    metrics5.to_csv(
        metrics_root
        / "C5_metrics_by_store.csv",
        index=False,
    )

    metrics4 = pd.read_csv(
        c4_dir / "C4_metrics_by_store.csv",
        float_precision="round_trip",
    )

    overviews = []
    parts = []

    for name, frame in (
        ("C4", metrics4),
        ("C5", metrics5),
    ):

        subset = frame.loc[
            frame.model.eq(
                "lightgbm_two_part"
            )
        ].copy()

        require(
            len(subset) == 10
            and subset.store_id.nunique() == 10,
            f"Metrics {name} khong du.",
        )

        valid = subset.rmsse_valid_items

        actual_sum = float(
            stores[
                f"actual_{name}"
            ].sum()
        )

        pred_sum = float(
            stores[
                f"prediction_{name}"
            ].sum()
        )

        overviews.append({
            "configuration": name,
            "actual": actual_sum,
            "prediction": pred_sum,

            "total_difference": (
                pred_sum - actual_sum
            ),

            "bias_pct": (
                100 * (
                    pred_sum / actual_sum - 1
                )
            ),

            "MAE": float(
                np.average(
                    subset.MAE,
                    weights=subset.n_items,
                )
            ),

            "RMSSE_mean_unweighted": float(
                (
                    subset.RMSSE_mean_unweighted
                    * valid
                ).sum() / valid.sum()
            ),
        })

        values = subset.set_index(
            "store_id"
        )[
            [
                "RMSSE_mean_unweighted",
                "MAE",
                "RMSE",
            ]
        ]

        values.columns = [
            f"{key}_{name}"
            for key in values.columns
        ]

        parts.append(values)

    rmsse_table = parts[0].join(
        parts[1],
        how="inner",
        validate="one_to_one",
    )

    require(
        len(rmsse_table) == 10,
        "Bang so sanh thieu cua hang.",
    )

    rmsse_table[
        "delta_RMSSE_C5_vs_C4"
    ] = (
        rmsse_table.RMSSE_mean_unweighted_C5
        - rmsse_table.RMSSE_mean_unweighted_C4
    )

    rmsse_table.reset_index().to_csv(
        metrics_root
        / "comparison_RMSSE_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        overviews
    ).to_csv(
        metrics_root
        / "comparison_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------
    # 13. DANH DAU HOAN TAT
    # ------------------------------------------------------

    save_json(
        dict(
            status="completed",
            configuration="C5",
            config=expected_c4,
            c4_batch=C4_TAG,
            features_added=EXTRAS,

            rows_per_configuration=853720,
            n_items=30490,
            stores=10,

            origin_day=1885,
            target_days=[1886, 1913],

            same_keys_as_C4=True,
            same_actuals_as_C4=True,
            same_RMSSE_scales=True,

            regressors_retrained=True,
            test_performed=False,

            price_policy="previous_completed_week",

            sources=sources,

            aggregate_predictions=str(
                aggregate.resolve()
            ),
        ),
        metrics_root / "manifest.json",
    )

    LOG.info(
        "C5 ALL COMPLETE | %s",
        metrics_root,
    )

    print(
        pd.DataFrame(
            overviews
        ).to_string(index=False)
    )

    print(
        rmsse_table.to_string()
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    main()
