
"""C4 - LightGBM Two-Part có giá bán lịch sử.

Classifier: 48 lá, 300 cây.
Regressor: 31 lá, 200 cây.

- Huấn luyện lại cả Classifier và Regressor.
- Giá dùng cho ngày thuộc tuần W là giá của tuần W-1.
- Validation 28 ngày, giữ giá đã biết tại origin.
- Không dùng lượng bán hoặc giá tương lai.
- Không ghi đè kết quả C0-C3.
"""

import copy
import gc
import json
import logging
import shutil
import sys

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from experiment_common import require

from src.data_loader import (
    check_files,
    load_calendar,
    load_prices,
    load_store,
)
from src.feature_engineering import prepare_training_data
from src.forecasting import (
    export_forecasts,
    recursive_forecast,
    validate_forecasts,
)
from src.preprocessing import history_matrix, prepare_history
from src.reporting import (
    evaluate_and_export,
    save_json,
    training_patterns,
)
from src.training import train_model


LOG = logging.getLogger("m5")

C0_TAG = "validation_C0_full_10_stores_20261119T103613400752Z"

C3_TAG = "validation_C3_full_10_stores_20261121T104421607533Z"

KEYS = [
    "split",
    "origin_day",
    "store_id",
    "item_id",
    "target_day",
    "horizon",
]

FORECAST_FILES = (
    "lightgbm_two_part.csv",
    "seasonal_naive.csv",
    "actual_vs_predictions.csv",
)


# ============================================================
# 1. CÁC HÀM HỖ TRỢ
# ============================================================


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Thiếu file: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def diagnostics(frame):
    positive = frame["y_actual"] > 0

    actual = float(
        frame["y_actual"].sum()
    )

    predicted = float(
        frame["y_pred_two_part"].sum()
    )

    actual_pos = float(
        frame.loc[positive, "y_actual"].sum()
    )

    conditional_pos = float(
        frame.loc[positive, "conditional_units"].sum()
    )

    return {
        "actual": actual,
        "prediction": predicted,

        "bias_pct": (
            100 * (predicted / actual - 1)
            if actual else np.nan
        ),

        "p_gap_pp": 100 * (
            frame["probability_positive"].mean()
            - positive.mean()
        ),

        "mu_bias_pct": (
            100 * (conditional_pos / actual_pos - 1)
            if actual_pos else np.nan
        ),
    }


def check_join(reference, current, store):
    matched = reference[
        KEYS + ["y_actual"]
    ].merge(
        current[KEYS + ["y_actual"]],
        on=KEYS,
        how="outer",
        suffixes=("_reference", "_C4"),
        indicator=True,
        validate="one_to_one",
    )

    require(
        len(matched) == 3049 * 28,
        f"Số dòng lệch: {store}",
    )

    require(
        matched["_merge"].eq("both").all(),
        f"Khóa không khớp: {store}",
    )

    np.testing.assert_array_equal(
        matched["y_actual_reference"],
        matched["y_actual_C4"],
    )


def append_csv(source, target):
    with (
        source.open("rb") as reader,
        target.open("ab") as writer,
    ):
        if target.stat().st_size:
            reader.readline()

        shutil.copyfileobj(reader, writer)


# ============================================================
# 2. XỬ LÝ GIÁ CỦA TUẦN ĐÃ HOÀN TẤT
# ============================================================

def previous_completed_week_prices(prices, calendar):
    """
    Chuyển giá từ tuần W sang đặc trưng của tuần W+1.

    Ví dụ:
    - Giá tuần 11325 dùng cho các ngày thuộc tuần 11326.
    - Giá tuần 11608 dùng cho các ngày thuộc tuần 11609.

    Không dùng giá tuần tương lai.
    Không điền giá ngược về quá khứ.
    """

    ordered_weeks = (
        calendar.groupby("wm_yr_wk")["date"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    if (
        len(ordered_weeks) < 2
        or len(ordered_weeks) != len(set(ordered_weeks))
    ):
        raise ValueError(
            "Calendar không có thứ tự tuần hợp lệ."
        )

    next_week_by_source = dict(
        zip(
            ordered_weeks[:-1],
            ordered_weeks[1:],
        )
    )

    result = prices.copy()

    result["wm_yr_wk"] = (
        result["wm_yr_wk"]
        .map(next_week_by_source)
    )

    result = result.dropna(
        subset=["wm_yr_wk"]
    ).copy()

    result["wm_yr_wk"] = (
        result["wm_yr_wk"].astype("int32")
    )

    require(
        not result.duplicated(
            ["store_id", "item_id", "wm_yr_wk"]
        ).any(),
        "Giá sau khi dịch một tuần bị trùng khóa.",
    )

    return result


# ============================================================
# 3. CHƯƠNG TRÌNH C4
# ============================================================

def main():

    # --------------------------------------------------------
    # BƯỚC 1: ĐỌC CẤU HÌNH GỐC VÀ KIỂM TRA C0/C3
    # --------------------------------------------------------

    original = yaml.safe_load(
        (ROOT / "configs/config.yaml").read_text(
            encoding="utf-8-sig"
        )
    )

    output = ROOT / original["results_dir"]

    c0_dir = output / "metrics" / C0_TAG
    c3_dir = output / "metrics" / C3_TAG

    c0 = read_json(
        c0_dir / "validation_manifest.json"
    )

    c3 = read_json(
        c3_dir / "manifest.json"
    )

    require(
        c0["status"] == "completed"
        and c3["status"] == "completed",
        "C0 hoặc C3 chưa hoàn tất.",
    )

    require(
        original == c0["config"]
        and not original["use_prices"],
        "config.yaml phải giữ nguyên cấu hình C0.",
    )

    expected_c3 = copy.deepcopy(original)

    expected_c3["classifier"].update(
        num_leaves=48,
        n_estimators=300,
    )

    require(
        c3["config"] == expected_c3,
        "C3 không đúng cấu hình 48 lá / 300 cây.",
    )

    require(
        c3["c0_batch"] == C0_TAG,
        "C3 không liên kết với C0 dự kiến.",
    )

    require(
        c3["n_items"] == 30490
        and c3["rows_per_configuration"] == 853720,
        "C3 chưa đủ phạm vi dữ liệu.",
    )

    require(
        len(c0["sources"]) == 10
        and len(c3["sources"]) == 10,
        "Thiếu cửa hàng.",
    )

    for flag in (
        "same_keys_as_C0",
        "same_actuals_as_C0",
        "same_RMSSE_scales",
    ):
        require(
            c3[flag],
            f"C3 chưa xác nhận {flag}",
        )

    # --------------------------------------------------------
    # BƯỚC 2: TẠO CẤU HÌNH C4
    # --------------------------------------------------------

    c4 = copy.deepcopy(expected_c3)

    # Chỉ bật biến giá bán.
    # Các siêu tham số giữ nguyên C3.
    c4["use_prices"] = True

    require(
        c4["price_policy"] == "last_known",
        "config.yaml gốc phải giữ price_policy=last_known.",
    )

    # Chính sách giá thực tế của C4:
    # previous_completed_week
    #
    # Không sửa config.yaml gốc.

    data_dir = ROOT / c4["data_dir"]

    check_files(
        data_dir,
        use_prices=True,
    )

    calendar = load_calendar(data_dir)

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    batch = (
        f"validation_C4_full_10_stores_{run_id}"
    )

    metrics_root = (
        output / "metrics" / batch
    )

    aggregate = (
        output / "validation" / batch
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
            "configuration": "C4",
            "config": c4,
            "price_feature_policy": "previous_completed_week",
            "c0_batch": C0_TAG,
            "c3_batch": C3_TAG,
        },
        metrics_root / "manifest.json",
    )

    c0_sources = {
        entry["store_id"]: entry
        for entry in c0["sources"]
    }

    c3_sources = {
        entry["store_id"]: entry
        for entry in c3["sources"]
    }

    require(
        set(c0_sources) == set(c3_sources),
        "C0 và C3 không có cùng danh sách cửa hàng.",
    )

    stats_by_store = []
    score_rows = []
    sources = []

    seen_pairs = set()

    # --------------------------------------------------------
    # BƯỚC 3: LẶP QUA 10 CỬA HÀNG
    # --------------------------------------------------------

    for store in c0_sources:

        baseline_source = c0_sources[store]
        previous_source = c3_sources[store]

        c0_model_metadata = read_json(
            Path(
                baseline_source["source_model_path"]
            ).parent / "metadata.json"
        )

        require(
            c0_model_metadata["config"] == original,
            f"Metadata C0 không khớp: {store}",
        )

        items_expected = (
            c0_model_metadata["selected_items"]
        )

        require(
            len(items_expected) == 3049,
            f"Thiếu sản phẩm tại {store}",
        )

        LOG.info(
            "C4 START | %s | classifier 48 la/300 cay "
            "| regressor 31 la/200 cay | USE PRICE",
            store,
        )

        # ----------------------------------------------------
        # BƯỚC 4: ĐỌC LƯỢNG BÁN TRAIN
        # ----------------------------------------------------

        wide = load_store(
            data_dir,
            store,
            1885,
            chunk_size=c4["chunk_size"],
            item_ids=items_expected,
        )

        items = wide.item_id.tolist()

        require(
            items == items_expected,
            f"Thứ tự sản phẩm thay đổi: {store}",
        )

        state_values = list(
            wide.state_id.unique()
        )

        require(
            len(state_values) == 1,
            f"Bang không hợp lệ: {store}",
        )

        state = state_values[0]

        # ----------------------------------------------------
        # BƯỚC 5: ĐỌC VÀ XỬ LÝ GIÁ
        # ----------------------------------------------------

        prices = load_prices(
            data_dir,
            store,
        )

        past_prices = previous_completed_week_prices(
            prices,
            calendar,
        )

        # Ghép Sales + Calendar + giá tuần trước.
        history = prepare_history(
            wide,
            calendar,
            1885,
            past_prices,
        )

        matrix = history_matrix(
            wide,
            1885,
        )

        _, selected, component = training_patterns(
            matrix,
            items,
        )

        # Lấy giá gần nhất được phép sử dụng tại origin.
        last_prices = (
            history.loc[history.day == 1885]
            .set_index("item_id")
            .loc[items, "sell_price"]
            .to_numpy(dtype=np.float32)
        )

        # Kiểm tra gốc dự báo d_1885.
        origin_week = int(
            calendar.loc[
                calendar["d"].eq("d_1885"),
                "wm_yr_wk",
            ].iloc[0]
        )

        week_end = (
            calendar.assign(
                day_num=calendar["d"]
                .str[2:]
                .astype(int)
            )
            .groupby("wm_yr_wk")["day_num"]
            .max()
        )

        prior_weeks = week_end[
            week_end.lt(1885)
        ]

        require(
            not prior_weeks.empty,
            "Không có tuần hoàn tất trước origin.",
        )

        prior_week = int(
            prior_weeks.index[-1]
        )

        require(
            prior_week != origin_week,
            "Đang sử dụng giá của tuần hiện tại.",
        )

        expected_prices = (
            prices.loc[
                prices["wm_yr_wk"].eq(prior_week)
            ]
            .set_index("item_id")["sell_price"]
            .reindex(items)
            .to_numpy(dtype=np.float32)
        )

        np.testing.assert_allclose(
            last_prices,
            expected_prices,
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        LOG.info(
            "C4 PRICE CHECK | %s | origin=%s "
            "| gia tuan=%s | %d items",
            store,
            origin_week,
            prior_week,
            len(items),
        )

        missing_origin = int(
            np.isnan(last_prices).sum()
        )

        require(
            missing_origin < len(items),
            f"Không có giá hợp lệ tại origin: {store}",
        )

        LOG.info(
            "C4 PRICE | %s | thiếu giá tại origin: %d/%d",
            store,
            missing_origin,
            len(items),
        )

        del wide, prices, past_prices

        # ----------------------------------------------------
        # BƯỚC 6: TẠO X VÀ Y CÓ GIÁ BÁN
        # ----------------------------------------------------

        train = prepare_training_data(
            history
        )

        del history

        gc.collect()

        require(
            "sell_price" in train["features"],
            f"C4 thiếu đặc trưng sell_price: {store}",
        )

        require(
            train["fit_end_day"] == 1885,
            f"Train vượt quá origin: {store}",
        )

        LOG.info(
            "C4 FEATURES | %s | %s",
            store,
            train["features"],
        )

        # ----------------------------------------------------
        # BƯỚC 7: HUẤN LUYỆN MỚI HAI THÀNH PHẦN
        # ----------------------------------------------------

        tag = (
            f"validation_C4_full_{store}_{run_id}"
        )

        locations = {
            name: output / name / tag
            for name in (
                "validation",
                "models",
                "metrics",
            )
        }

        parameters = {
            part: {
                **c4[part],
                "importance_type": "gain",
            }
            for part in (
                "classifier",
                "regressor",
            )
        }

        # Train mới Classifier + Regressor.
        # KHÔNG tái sử dụng Regressor C0.
        bundle, info = train_model(
            train,
            parameters,
            locations["models"] / "model.joblib",
        )

        require(
            info["n_train_rows"] == 5661993,
            f"Không đủ dữ liệu Train: {store}",
        )

        require(
            "sell_price" in bundle["features"],
            f"Model không sử dụng sell_price: {store}",
        )

        del train

        gc.collect()

        locations["validation"].mkdir(
            parents=True,
            exist_ok=False,
        )

        locations["metrics"].mkdir(
            parents=True,
            exist_ok=False,
        )

        # ----------------------------------------------------
        # BƯỚC 8: DỰ BÁO ĐỆ QUY 28 NGÀY
        # ----------------------------------------------------

        prediction, components = recursive_forecast(
            bundle["model"],
            history=matrix,
            items=items,
            store=store,
            state=state,
            calendar=calendar,
            origin=1885,
            features=bundle["features"],
            categories=bundle["categories"],
            split="validation",
            horizon=28,
            last_prices=last_prices,
            return_components=True,
        )

        export_forecasts(
            prediction,
            locations["validation"]
            / "lightgbm_two_part.csv",
        )

        # ----------------------------------------------------
        # BƯỚC 9: ĐỌC THỰC TẾ VALIDATION ĐỂ CHẤM
        # ----------------------------------------------------

        # Chỉ đọc thực tế SAU khi dự báo đã hoàn tất.
        actual = load_store(
            data_dir,
            store,
            1913,
            start_day=1886,
            item_ids=items,
            chunk_size=c4["chunk_size"],
        )

        naive_path = (
            Path(
                baseline_source["actual_vs_predictions"]
            ).parent
            / "seasonal_naive.csv"
        )

        naive = pd.read_csv(
            naive_path,
            float_precision="round_trip",
        )

        validate_forecasts(naive)

        export_forecasts(
            naive,
            locations["validation"]
            / "seasonal_naive.csv",
        )

        context = dict(
            config=c4,
            configuration_name="C4",
            store_id=store,
            scope="full",
            split="validation",
            origin_day=1885,
            run_id=run_id,
            batch_tag=batch,
            illustration_items=selected,
            component_item=component,
            save_figures=False,
            mode="validation",
            status="completed",
            regressor_retrained=True,
            forecast_performed=True,
            evaluation_performed=True,
            price_origin_missing_items=missing_origin,
            price_policy="previous_completed_week",
        )

        scores = evaluate_and_export(
            prediction,
            naive,
            actual,
            matrix,
            items,
            bundle,
            info,
            locations,
            context,
            components=components,
        )

        current = pd.read_csv(
            locations["validation"]
            / "actual_vs_predictions.csv",
            float_precision="round_trip",
        )

        previous = pd.read_csv(
            previous_source["actual_vs_predictions"],
            float_precision="round_trip",
        )

        # ----------------------------------------------------
        # BƯỚC 10: KIỂM TRA C3/C4 CÙNG TẬP VALIDATION
        # ----------------------------------------------------

        check_join(
            previous,
            current,
            store,
        )

        np.testing.assert_allclose(
            current["y_pred_two_part"],
            current["probability_positive"]
            * current["conditional_units"],
            rtol=1e-12,
            atol=1e-12,
        )

        scale3 = pd.read_csv(
            Path(
                previous_source["summary"]
            ).parent
            / "lightgbm_two_part_per_item.csv",
            float_precision="round_trip",
        ).set_index("item_id")

        scale4 = pd.read_csv(
            locations["metrics"]
            / "lightgbm_two_part_per_item.csv",
            float_precision="round_trip",
        ).set_index("item_id")

        np.testing.assert_allclose(
            scale3.loc[items, "scale"],
            scale4.loc[items, "scale"],
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        row = {
            "store_id": store,
            "missing_price_origin": missing_origin,
        }

        for name, frame in (
            ("C3", previous),
            ("C4", current),
        ):
            row.update({
                f"{field}_{name}": value
                for field, value in diagnostics(
                    frame
                ).items()
            })

        stats_by_store.append(row)

        point = scores.loc[
            scores.model == "lightgbm_two_part"
        ]

        require(
            len(point) == 1,
            f"Thiếu kết quả RMSSE C4: {store}",
        )

        score_rows.append(
            point.iloc[0].to_dict()
        )

        # Kiểm tra chuỗi không trùng.
        for item in items:
            require(
                (store, item) not in seen_pairs,
                f"Chuỗi trùng: {store}/{item}",
            )

            seen_pairs.add(
                (store, item)
            )

        # ----------------------------------------------------
        # BƯỚC 11: GỘP DỰ BÁO TỪNG CỬA HÀNG
        # ----------------------------------------------------

        for name in FORECAST_FILES:
            append_csv(
                locations["validation"] / name,
                aggregate / name,
            )

        sources.append({
            "store_id": store,
            "run_tag": tag,

            "model_path": str(
                (
                    locations["models"]
                    / "model.joblib"
                ).resolve()
            ),

            "actual_vs_predictions": str(
                (
                    locations["validation"]
                    / "actual_vs_predictions.csv"
                ).resolve()
            ),

            "summary": str(
                (
                    locations["metrics"]
                    / "summary.csv"
                ).resolve()
            ),

            "missing_price_origin": missing_origin,
        })

        LOG.info(
            "C4 COMPLETE | %s | train %.1fs | RMSSE %.6f",
            store,
            info["training_time_seconds"],
            point.iloc[0]["RMSSE_mean_unweighted"],
        )

        del (
            bundle,
            matrix,
            actual,
            prediction,
            components,
            naive,
            previous,
            current,
            scale3,
            scale4,
        )

        gc.collect()

    # --------------------------------------------------------
    # BƯỚC 12: TỔNG HỢP 10 CỬA HÀNG
    # --------------------------------------------------------

    require(
        len(sources) == 10
        and len(seen_pairs) == 30490,
        "C4 chưa đủ 10 cửa hàng hoặc 30490 chuỗi.",
    )

    comparison = pd.DataFrame(
        stats_by_store
    )

    comparison.to_csv(
        metrics_root / "comparison_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    c4_scores = pd.DataFrame(
        score_rows
    )

    c4_scores.to_csv(
        metrics_root / "C4_metrics_by_store.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BƯỚC 13: SO SÁNH C3 VỚI C4
    # --------------------------------------------------------

    c3_scores = pd.read_csv(
        c3_dir / "C3_metrics_by_store.csv",
        float_precision="round_trip",
    )

    overview = []
    wide_metrics = []

    for label, table in (
        ("C3", c3_scores),
        ("C4", c4_scores),
    ):

        selected_scores = table.loc[
            table.model == "lightgbm_two_part"
        ].copy()

        require(
            len(selected_scores) == 10
            and selected_scores.store_id.nunique() == 10,
            f"Metrics {label} thiếu cửa hàng.",
        )

        count = selected_scores[
            "rmsse_valid_items"
        ]

        actual_sum = float(
            comparison[
                f"actual_{label}"
            ].sum()
        )

        predicted_sum = float(
            comparison[
                f"prediction_{label}"
            ].sum()
        )

        overview.append({
            "configuration": label,
            "actual": actual_sum,
            "prediction": predicted_sum,

            "total_difference": (
                predicted_sum - actual_sum
            ),

            "bias_pct": (
                100 * (
                    predicted_sum / actual_sum - 1
                )
            ),

            "MAE": float(
                np.average(
                    selected_scores["MAE"],
                    weights=selected_scores["n_items"],
                )
            ),

            "RMSSE_mean_unweighted": float(
                (
                    selected_scores[
                        "RMSSE_mean_unweighted"
                    ] * count
                ).sum() / count.sum()
            ),
        })

        by_store = selected_scores.set_index(
            "store_id"
        )[
            [
                "RMSSE_mean_unweighted",
                "MAE",
                "RMSE",
            ]
        ]

        by_store.columns = [
            f"{column}_{label}"
            for column in by_store.columns
        ]

        wide_metrics.append(by_store)

    table = wide_metrics[0].join(
        wide_metrics[1],
        how="inner",
        validate="one_to_one",
    )

    require(
        len(table) == 10,
        "Không đủ 10 cửa hàng khi so sánh.",
    )

    table["delta_RMSSE_C4_vs_C3"] = (
        table["RMSSE_mean_unweighted_C4"]
        - table["RMSSE_mean_unweighted_C3"]
    )

    table.reset_index().to_csv(
        metrics_root
        / "comparison_RMSSE_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(overview).to_csv(
        metrics_root / "comparison_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # BƯỚC 14: GHI MANIFEST HOÀN THÀNH
    # --------------------------------------------------------

    save_json(
        {
            "status": "completed",
            "configuration": "C4",
            "config": c4,
            "c0_batch": C0_TAG,
            "c3_batch": C3_TAG,

            "rows_per_configuration": 853720,
            "n_items": 30490,
            "stores": 10,

            "origin_day": 1885,
            "target_days": [1886, 1913],

            "same_keys_as_C3": True,
            "same_actuals_as_C3": True,
            "same_RMSSE_scales": True,

            "regressors_retrained": True,
            "test_performed": False,

            "price_policy": "previous_completed_week",

            "sources": sources,

            "aggregate_predictions": str(
                aggregate.resolve()
            ),
        },
        metrics_root / "manifest.json",
    )

    LOG.info(
        "C4 ALL COMPLETE | %s",
        metrics_root,
    )

    print(
        pd.DataFrame(overview).to_string(
            index=False
        )
    )

    print(
        table.to_string()
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    main()
