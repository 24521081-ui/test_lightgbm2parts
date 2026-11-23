
"""C3: Classifier 48 lá, 300 cây. Tái sử dụng regressor C0. Chỉ Validation."""

import copy
import gc
import hashlib
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

from src.data_loader import check_files, load_calendar, load_store
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
from src.training import train_classifier_reuse_regressor

LOG = logging.getLogger("m5")

C0_TAG = "validation_C0_full_10_stores_20261119T103613400752Z"
C2_TAG = "validation_C2_full_10_stores_20261121T101840433405Z"


def diagnostics(frame):
    positive = frame["y_actual"] > 0
    actual = float(frame["y_actual"].sum())
    prediction = float(frame["y_pred_two_part"].sum())

    return {
        "actual": actual,
        "prediction": prediction,
        "bias_pct": 100 * (prediction / actual - 1),
        "p_gap_pp": 100 * (
            frame["probability_positive"].mean()
            - positive.mean()
        ),
        "mu_bias_pct": 100 * (
            frame.loc[positive, "conditional_units"].sum()
            / frame.loc[positive, "y_actual"].sum()
            - 1
        ),
    }


def main():

    # =====================================================
    # 1. ĐỌC VÀ KIỂM TRA C0, C2
    # =====================================================

    config = yaml.safe_load(
        (ROOT / "configs/config.yaml").read_text(
            encoding="utf-8-sig"
        )
    )

    results = ROOT / config["results_dir"]

    c0_metrics = results / "metrics" / C0_TAG
    c2_metrics = results / "metrics" / C2_TAG

    c0 = json.loads(
        (c0_metrics / "validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    c2 = json.loads(
        (c2_metrics / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if not (
        c0["status"] == "completed"
        and c2["status"] == "completed"
    ):
        raise ValueError("C0 hoặc C2 chưa hoàn tất.")

    if (
        config != c0["config"]
        or config["classifier"]["num_leaves"] != 31
    ):
        raise ValueError(
            "config.yaml phải giữ nguyên cấu hình C0."
        )

    if config["use_prices"]:
        raise ValueError(
            "C3 phải giữ use_prices: false."
        )

    if not (
        c2["c0_batch"] == C0_TAG
        and c2["config"]["classifier"]["num_leaves"] == 48
        and c2["config"]["classifier"]["n_estimators"] == 200
        and c2["config"]["regressor"] == config["regressor"]
        and c2["same_keys_as_C0"]
        and c2["same_actuals_as_C0"]
        and c2["same_RMSSE_scales"]
    ):
        raise ValueError(
            "C2 không khớp cấu hình tham chiếu."
        )

    if not (
        len(c0["sources"]) == 10
        and c0["rows_per_model"] == 853720
    ):
        raise ValueError(
            "C0 chưa đủ phạm vi Full 10 cửa hàng."
        )

    # =====================================================
    # 2. TẠO CẤU HÌNH C3
    # =====================================================

    c3 = copy.deepcopy(config)

    # C2: 48 lá, 200 cây
    # C3: 48 lá, 300 cây
    c3["classifier"]["num_leaves"] = 48
    c3["classifier"]["n_estimators"] = 300

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    batch = f"validation_C3_full_10_stores_{run_id}"

    metrics_root = results / "metrics" / batch
    aggregate = results / "validation" / batch

    metrics_root.mkdir(
        parents=True, exist_ok=False
    )

    aggregate.mkdir(
        parents=True, exist_ok=False
    )

    save_json(
        {
            "status": "running",
            "configuration": "C3",
            "config": c3,
            "c0_batch": C0_TAG,
            "c2_batch": C2_TAG,
        },
        metrics_root / "manifest.json",
    )

    # =====================================================
    # 3. ĐỌC DỮ LIỆU
    # =====================================================

    data_dir = ROOT / config["data_dir"]

    check_files(data_dir, False)
    calendar = load_calendar(data_dir)

    store_rows = []
    metric_rows = []
    sources = []
    all_pairs = set()

    # =====================================================
    # 4. XỬ LÝ 10 CỬA HÀNG
    # =====================================================

    for source in c0["sources"]:

        store = source["store_id"]
        src_model = Path(source["source_model_path"])

        src_info = json.loads(
            (
                src_model.parent / "metadata.json"
            ).read_text(encoding="utf-8")
        )

        if not (
            src_info["config"] == config
            and src_info["fit_end_day"] == 1885
            and src_info["n_items"] == 3049
        ):
            raise ValueError(
                f"Metadata C0 không khớp: {store}"
            )

        src_hash = hashlib.sha256(
            src_model.read_bytes()
        ).hexdigest()

        if src_hash != source["source_model_sha256"]:
            raise ValueError(
                f"Hash mô hình C0 không khớp: {store}"
            )

        LOG.info(
            "C3 START | %s | classifier 48 lá, 300 cây | regressor C0",
            store,
        )

        # =================================================
        # 5. CHUẨN BỊ TRAIN
        # =================================================

        wide = load_store(
            data_dir,
            store,
            1885,
            chunk_size=config["chunk_size"],
            item_ids=src_info["selected_items"],
        )

        items = wide.item_id.tolist()

        if (
            items != src_info["selected_items"]
            or len(items) != 3049
        ):
            raise ValueError(
                f"Danh sách sản phẩm không khớp: {store}"
            )

        history = prepare_history(
            wide, calendar, 1885
        )

        matrix = history_matrix(
            wide, 1885
        )

        _, selected, component = training_patterns(
            matrix, items
        )

        del wide

        train = prepare_training_data(history)

        del history
        gc.collect()

        # =================================================
        # 6. HUẤN LUYỆN CLASSIFIER C3
        # =================================================

        tag = f"validation_C3_full_{store}_{run_id}"

        locations = {
            name: results / name / tag
            for name in [
                "validation",
                "models",
                "metrics",
            ]
        }

        parameters = {
            **c3["classifier"],
            "importance_type": "gain",
        }

        bundle, info = train_classifier_reuse_regressor(
            train,
            parameters,
            src_model,
            locations["models"] / "model.joblib",
        )

        expected_classifier = dict(
            src_info["effective_model_params"]["classifier"]
        )

        expected_classifier["num_leaves"] = 48
        expected_classifier["n_estimators"] = 300

        if not (
            info["n_train_rows"] == 5661993
            and info["effective_model_params"]["classifier"]
            == expected_classifier
            and info["effective_model_params"]["regressor"]
            == src_info["effective_model_params"]["regressor"]
            and not info["regressor_retrained"]
        ):
            raise ValueError(
                f"Huấn luyện C3 không đúng tham số: {store}"
            )

        del train
        gc.collect()

        for name in ["validation", "metrics"]:
            locations[name].mkdir(
                parents=True, exist_ok=False
            )

        # =================================================
        # 7. DỰ BÁO ĐỆ QUY 28 NGÀY
        # =================================================

        common = dict(
            history=matrix,
            items=items,
            store=store,
            state=src_info["state_id"],
            calendar=calendar,
            origin=1885,
            features=bundle["features"],
            categories=bundle["categories"],
            split="validation",
            horizon=28,
        )

        prediction, components = recursive_forecast(
            bundle["model"],
            **common,
            return_components=True,
        )

        export_forecasts(
            prediction,
            locations["validation"]
            / "lightgbm_two_part.csv",
        )

        # Chỉ đọc thực tế Validation sau khi
        # hoàn tất dự báo 28 ngày.

        actual = load_store(
            data_dir,
            store,
            1913,
            start_day=1886,
            item_ids=items,
            chunk_size=config["chunk_size"],
        )

        old = pd.read_csv(
            source["actual_vs_predictions"],
            float_precision="round_trip",
        )

        baseline = pd.read_csv(
            Path(
                source["actual_vs_predictions"]
            ).parent / "seasonal_naive.csv",
            float_precision="round_trip",
        )

        validate_forecasts(baseline)

        export_forecasts(
            baseline,
            locations["validation"]
            / "seasonal_naive.csv",
        )

        # =================================================
        # 8. CHẤM ĐIỂM C3
        # =================================================

        context = dict(
            config=c3,
            configuration_name="C3",
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
            regressor_retrained=False,
            source_c0_model_path=str(src_model),
            source_c0_model_sha256=src_hash,
            forecast_performed=True,
            evaluation_performed=True,
        )

        scores = evaluate_and_export(
            prediction,
            baseline,
            actual,
            matrix,
            items,
            bundle,
            info,
            locations,
            context,
            components=components,
        )

        new = pd.read_csv(
            locations["validation"]
            / "actual_vs_predictions.csv",
            float_precision="round_trip",
        )

        # =================================================
        # 9. KIỂM TRA C0 VÀ C3 CÙNG DỮ LIỆU
        # =================================================

        keys = [
            "split",
            "origin_day",
            "item_id",
            "store_id",
            "target_day",
            "horizon",
        ]

        joined = old.merge(
            new,
            on=keys,
            how="outer",
            suffixes=("_C0", "_C3"),
            indicator=True,
            validate="one_to_one",
        )

        if (
            len(joined) != 85372
            or not joined["_merge"].eq("both").all()
        ):
            raise ValueError(
                f"Khóa C0/C3 không khớp: {store}"
            )

        np.testing.assert_array_equal(
            joined.y_actual_C0,
            joined.y_actual_C3,
        )

        np.testing.assert_allclose(
            new.y_pred_two_part,
            new.probability_positive
            * new.conditional_units,
            rtol=1e-12,
            atol=1e-12,
        )

        np.testing.assert_array_equal(
            new.signed_error_two_part,
            new.y_pred_two_part - new.y_actual,
        )

        scale0 = pd.read_csv(
            Path(source["summary"]).parent
            / "lightgbm_two_part_per_item.csv",
            float_precision="round_trip",
        ).set_index("item_id")

        scale3 = pd.read_csv(
            locations["metrics"]
            / "lightgbm_two_part_per_item.csv",
            float_precision="round_trip",
        ).set_index("item_id")

        np.testing.assert_allclose(
            scale0.loc[items, "scale"],
            scale3.loc[items, "scale"],
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        # =================================================
        # 10. GHI KẾT QUẢ THEO CỬA HÀNG
        # =================================================

        row = {"store_id": store}

        for label, detail in [
            ("C0", old),
            ("C3", new),
        ]:
            row.update({
                f"{key}_{label}": value
                for key, value in diagnostics(
                    detail
                ).items()
            })

        store_rows.append(row)

        metric_rows.append(
            scores[
                scores.model == "lightgbm_two_part"
            ].iloc[0].to_dict()
        )

        for item in items:
            if (store, item) in all_pairs:
                raise ValueError(
                    f"Chuỗi bị trùng: {(store, item)}"
                )

            all_pairs.add((store, item))

        # =================================================
        # 11. GỘP DỰ BÁO CỦA 10 CỬA HÀNG
        # =================================================

        for name in [
            "lightgbm_two_part.csv",
            "seasonal_naive.csv",
            "actual_vs_predictions.csv",
        ]:
            destination = aggregate / name

            with (
                (
                    locations["validation"] / name
                ).open("rb") as reader,
                destination.open("ab") as writer,
            ):
                if destination.stat().st_size:
                    reader.readline()

                shutil.copyfileobj(
                    reader, writer
                )

        sources.append(
            dict(
                store_id=store,
                run_tag=tag,
                model_path=str(
                    (
                        locations["models"]
                        / "model.joblib"
                    ).resolve()
                ),
                regressor_source=info[
                    "source_regressor_path"
                ],
                regressor_sha256=info[
                    "source_regressor_sha256"
                ],
                actual_vs_predictions=str(
                    (
                        locations["validation"]
                        / "actual_vs_predictions.csv"
                    ).resolve()
                ),
                summary=str(
                    (
                        locations["metrics"]
                        / "summary.csv"
                    ).resolve()
                ),
            )
        )

        LOG.info(
            "C3 COMPLETE | %s | rows=%d | fit %.2fs | trees=%s",
            store,
            len(new),
            info["training_time_seconds"],
            info["actual_trees"],
        )

        del (
            bundle,
            common,
            matrix,
            prediction,
            components,
            actual,
            old,
            new,
            joined,
            baseline,
            scale0,
            scale3,
        )

        gc.collect()

    # =====================================================
    # 12. TỔNG HỢP KẾT QUẢ 10 CỬA HÀNG
    # =====================================================

    if len(all_pairs) != 30490 or len(sources) != 10:
        raise ValueError(
            "Chưa đủ 30490 chuỗi và 10 cửa hàng."
        )

    stores = pd.DataFrame(
        store_rows
    ).set_index("store_id")

    # Đọc kết quả C2 để so sánh với C0 và C3.
    c2_stores = pd.read_csv(
        c2_metrics / "comparison_by_store.csv",
        encoding="utf-8-sig",
    ).set_index("store_id")

    if set(stores.index) != set(c2_stores.index):
        raise ValueError(
            "Danh sách cửa hàng C2/C3 không khớp."
        )

    c2_stores = c2_stores.reindex(
        stores.index
    )

    for key in [
        "actual",
        "prediction",
        "bias_pct",
        "p_gap_pp",
        "mu_bias_pct",
    ]:
        np.testing.assert_allclose(
            stores[f"{key}_C0"],
            c2_stores[f"{key}_C0"],
            rtol=1e-10,
            atol=1e-10,
        )

        stores[f"{key}_C2"] = (
            c2_stores[f"{key}_C2"]
        )

    stores = stores.reset_index()

    stores.to_csv(
        metrics_root / "comparison_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    current = pd.DataFrame(
        metric_rows
    )

    current.to_csv(
        metrics_root / "C3_metrics_by_store.csv",
        index=False,
    )

    # =====================================================
    # 13. SO SÁNH RMSSE C0 - C2 - C3
    # =====================================================

    c0_scores = pd.read_csv(
        c0_metrics / "Metrics_by_store.csv",
        float_precision="round_trip",
    )

    c2_scores = pd.read_csv(
        c2_metrics / "C2_metrics_by_store.csv",
        float_precision="round_trip",
    )

    score_parts = []
    overview = []

    for label, metrics in [
        ("C0", c0_scores),
        ("C2", c2_scores),
        ("C3", current),
    ]:

        metrics = metrics[
            metrics["model"] == "lightgbm_two_part"
        ].copy()

        if (
            len(metrics) != 10
            or metrics["store_id"].nunique() != 10
        ):
            raise ValueError(
                f"Metrics {label} chưa đủ 10 cửa hàng."
            )

        valid = metrics["rmsse_valid_items"]

        total_actual = float(
            stores[f"actual_{label}"].sum()
        )

        total_prediction = float(
            stores[f"prediction_{label}"].sum()
        )

        overview.append(
            dict(
                configuration=label,
                actual=total_actual,
                prediction=total_prediction,
                total_difference=(
                    total_prediction - total_actual
                ),
                bias_pct=100 * (
                    total_prediction
                    / total_actual - 1
                ),
                MAE=float(
                    np.average(
                        metrics["MAE"],
                        weights=metrics["n_items"],
                    )
                ),
                RMSSE_mean_unweighted=float(
                    (
                        metrics[
                            "RMSSE_mean_unweighted"
                        ].fillna(0) * valid
                    ).sum() / valid.sum()
                ),
            )
        )

        by_store = metrics.set_index(
            "store_id"
        )[
            [
                "RMSSE_mean_unweighted",
                "MAE",
                "RMSE",
            ]
        ]

        by_store.columns = [
            f"{col}_{label}"
            for col in by_store.columns
        ]

        score_parts.append(by_store)

    # =====================================================
    # 14. KIỂM TRA KẾT QUẢ C0/C2 KHÔNG THAY ĐỔI
    # =====================================================

    prior = pd.read_csv(
        c2_metrics / "comparison_overall.csv",
        encoding="utf-8-sig",
    ).set_index("configuration")

    for entry in overview[:2]:
        label = entry["configuration"]

        for key in [
            "actual",
            "prediction",
            "MAE",
            "RMSSE_mean_unweighted",
        ]:
            if not np.isclose(
                entry[key],
                prior.loc[label, key],
                rtol=1e-10,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Kết quả C0/C2 thay đổi: {label}, {key}"
                )

    # =====================================================
    # 15. XUẤT BẢNG RMSSE TỪNG CỬA HÀNG
    # =====================================================

    table = pd.concat(
        score_parts,
        axis=1,
    )

    table["delta_RMSSE_C3_vs_C2"] = (
        table["RMSSE_mean_unweighted_C3"]
        - table["RMSSE_mean_unweighted_C2"]
    )

    table["delta_RMSSE_C3_vs_C0"] = (
        table["RMSSE_mean_unweighted_C3"]
        - table["RMSSE_mean_unweighted_C0"]
    )

    table.reset_index().to_csv(
        metrics_root / "comparison_RMSSE_by_store.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(overview).to_csv(
        metrics_root / "comparison_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # 16. ĐÁNH DẤU C3 HOÀN TẤT
    # =====================================================

    save_json(
        dict(
            status="completed",
            configuration="C3",
            config=c3,
            c0_batch=C0_TAG,
            c2_batch=C2_TAG,
            rows_per_configuration=853720,
            n_items=30490,
            stores=10,
            origin_day=1885,
            target_days=[1886, 1913],
            same_keys_as_C0=True,
            same_actuals_as_C0=True,
            same_RMSSE_scales=True,
            regressors_retrained=False,
            test_performed=False,
            figures_generated=False,
            sources=sources,
            aggregate_predictions=str(
                aggregate.resolve()
            ),
        ),
        metrics_root / "manifest.json",
    )

    LOG.info(
        "C3 ALL COMPLETE | %s",
        metrics_root,
    )

    print(
        pd.DataFrame(overview).to_string(
            index=False
        )
    )

    print(table.to_string())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
