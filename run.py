"""Script chạy toàn bộ sáu giai đoạn, từ dữ liệu M5 đến kết quả đánh giá."""

import argparse
import gc
import logging
import json
import platform
import hashlib
import shutil
from importlib.metadata import version
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data_loader import (
    check_files,
    list_stores,
    load_calendar,
    load_prices,
    load_store,
)
from src.feature_engineering import prepare_training_data
from src.forecasting import SeasonalNaive, export_forecasts, recursive_forecast
from src.preprocessing import history_matrix, prepare_history
from src.reporting import evaluate_and_export, training_patterns, save_json, export_validation_workbook
from src.training import train_model, verify_saved_models

LOGGER = logging.getLogger("m5")


def load_config(path):
    """Đọc YAML và ngăn cấu hình làm sai thiết kế thí nghiệm."""
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (config["validation_origin"], config["test_origin"], config["horizon"]) != (
        1885,
        1913,
        28,
    ):
        raise ValueError("Thiết kế yêu cầu origin 1885/1913 và horizon 28.")
    for name, objective in [("classifier", "binary"), ("regressor", "regression")]:
        model = config[name]
        if model["objective"] != objective:
            raise ValueError(f"{name} cần objective={objective}.")
        if model["device_type"] != "cpu" or model["n_jobs"] < 1:
            raise ValueError("Cần CPU và n_jobs là số luồng dương.")
    if config["price_policy"] != "last_known":
        raise ValueError("Phiên bản này chỉ hỗ trợ price_policy=last_known.")
    if config["chunk_size"] < 1 or config["smoke_items"] < 1:
        raise ValueError("chunk_size và smoke_items phải dương.")
    return config


def run_store(config, args, data_dir, output_root, store):
    """Chạy từ đọc dữ liệu đến xuất kết quả, xử lý riêng từng cửa hàng."""
    # Thiết lập phạm vi trước khi đọc dữ liệu, không chọn theo sai số holdout.
    started = perf_counter()
    train_only = args.mode == "train"
    split = "test" if args.mode == "test" else "validation"
    origin = config[f"{split}_origin"]
    scope = "smoke" if args.mode == "smoke" else args.scope
    sample = config["smoke_items"] if scope == "smoke" else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tag = f"{'train' if train_only else split}_{scope}_{store}_{run_id}"

    locations = {
        name: output_root / name / tag
        for name in (["models"] if train_only else [split, "models", "figures", "metrics"])
    }

    for directory in locations.values():
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu
    # Chỉ lấy doanh số đến origin; chưa đọc nhãn của 28 ngày cần dự báo.
    LOGGER.info("1/6. Đọc dữ liệu | %s", store)
    calendar = load_calendar(data_dir)
    wide = load_store(
        data_dir,
        store,
        origin,
        chunk_size=config["chunk_size"],
        sample_items=sample,
        seed=config["seed"],
    )
    prices = load_prices(data_dir, store) if config["use_prices"] else None

    # 2. Chuẩn bị dữ liệu
    # Tạo bảng dài cho huấn luyện và ma trận lịch sử cho dự báo.
    LOGGER.info("2/6. Chuẩn bị dữ liệu")

    if scope == "full" and len(wide) != 3049:
        raise ValueError(
            f"Full M5 cần 3049 sản phẩm/cửa hàng; tìm thấy {len(wide)}. "
            "Không tự cắt mẫu."
        )

    states = wide.state_id.unique()

    if len(states) != 1 or states[0] not in ["CA", "TX", "WI"]:
        raise ValueError("Một cửa hàng phải thuộc đúng một bang CA/TX/WI.")

    state = states[0]
    items = wide.item_id.tolist()
    history = prepare_history(wide, calendar, origin, prices)
    matrix = history_matrix(wide, origin)
    _, illustration_items, component_item = training_patterns(matrix, items)
    last_prices = None

    if config["use_prices"]:
        # Giữ giá đã biết tại origin cố định cho toàn bộ 28 ngày dự báo.
        last_prices = (
            history.loc[history.day == origin]
            .set_index("item_id")
            .loc[items, "sell_price"]
            .to_numpy()
        )

    del wide, prices
    gc.collect()

    # 3. Tạo đặc trưng
    # Lag và rolling chỉ dùng quá khứ; bỏ hàng chưa đủ 28 ngày lịch sử.
    LOGGER.info("3/6. Tạo đặc trưng")
    training_data = prepare_training_data(history)

    del history
    gc.collect()

    # 4. Huấn luyện
    # Fit một model chung cho các sản phẩm đã chọn của cửa hàng.
    LOGGER.info("4/6. Huấn luyện LightGBM Two-Part")
    parameters = {name: {**config[name], "importance_type": "gain"}
                  for name in ["classifier", "regressor"]}
    bundle, info = train_model(
        training_data,
        parameters,
        locations["models"] / "model.joblib",
    )

    del training_data
    gc.collect()

    if train_only:
        metadata = {
            **info, "status": "completed", "mode": "train", "configuration_name": "C0",
            "config": config, "store_id": store, "state_id": state, "scope": scope,
            "run_id": run_id, "origin_day": origin, "train_start_day": 1,
            "selected_items": items, "categories": bundle["categories"],
            "intended_validation_days": [origin + 1, origin + config["horizon"]],
            "forecast_performed": False, "evaluation_performed": False,
            "store_wall_time_seconds": perf_counter() - started,
            "python_version": platform.python_version(),
            "libraries": {name: version(name) for name in ["numpy", "pandas", "lightgbm", "scikit-learn", "joblib"]},
            "model_path": str((locations["models"] / "model.joblib").resolve()),
        }
        save_json(metadata, locations["models"] / "metadata.json")
        LOGGER.info("TRAIN COMPLETE | %s | %d items | trees=%s | %s", store, len(items),
                    info["actual_trees"], locations["models"])
        return metadata

    # 5. Dự báo
    # Mỗi mô hình cập nhật lịch sử bằng dự báo của chính nó sau mỗi bước.
    LOGGER.info("5/6. Dự báo 28 ngày")
    common = {
        "history": matrix,
        "items": items,
        "store": store,
        "state": state,
        "calendar": calendar,
        "origin": origin,
        "features": bundle["features"],
        "categories": bundle["categories"],
        "split": split,
        "horizon": config["horizon"],
        "last_prices": last_prices,
    }
    prediction, components = recursive_forecast(bundle["model"], **common, return_components=True)
    baseline = recursive_forecast(
        SeasonalNaive(),
        **common,
        model_name="seasonal_naive",
    )

    export_forecasts(prediction, locations[split] / "lightgbm_two_part.csv")
    export_forecasts(baseline, locations[split] / "seasonal_naive.csv")

    # 6. Đánh giá và xuất kết quả
    # Chỉ đọc doanh số thật của holdout sau khi cả hai mô hình dự báo xong.
    LOGGER.info("6/6. Đánh giá và xuất kết quả")
    actual_wide = load_store(
        data_dir,
        store,
        origin + config["horizon"],
        chunk_size=config["chunk_size"],
        start_day=origin + 1,
        item_ids=items,
    )
    context = {
        "config": config,
        "store_id": store,
        "scope": scope,
        "split": split,
        "origin_day": origin,
        "run_id": run_id,
        "illustration_items": illustration_items,
        "component_item": component_item,
    }

    return evaluate_and_export(
        prediction,
        baseline,
        actual_wide,
        matrix,
        items,
        bundle,
        info,
        locations,
        context,
        components=components,
    )


def run_saved_validation(config, data_dir, output_root, stores, training_summary):
    """Validation từ mô hình đã lưu; không gọi train_model hoặc đọc nhãn Test."""
    import pandas as pd
    inventory = pd.read_csv(training_summary)
    if inventory.store_id.duplicated().any() or not set(stores).issubset(set(inventory.store_id)):
        raise ValueError("Bảng Train thiếu/trùng cửa hàng.")
    if config['use_prices'] or any(
        (config[name]['n_estimators'], config[name]['learning_rate'], config[name]['num_leaves'])
        != (200, .05, 31) for name in ['classifier', 'regressor']
    ):
        raise ValueError("Validation C0 yêu cầu tắt giá và 200 cây/0.05/31 lá.")
    prepared = {}
    for store in stores:
        model_path = Path(inventory.set_index('store_id').loc[store, 'model_path'])
        metadata = json.loads((model_path.parent / 'metadata.json').read_text(encoding='utf-8'))
        if (metadata.get('status') != 'completed' or metadata.get('mode') != 'train'
                or metadata.get('config') != config or metadata.get('scope') != 'full'
                or metadata.get('store_id') != store or metadata.get('n_items') != 3049
                or metadata.get('origin_day') != 1885 or metadata.get('fit_end_day') != 1885):
            raise ValueError(f"Mô hình Train không khớp C0/origin/phạm vi: {store}")
        prepared[store] = (model_path, metadata, hashlib.sha256(model_path.read_bytes()).hexdigest())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    batch_tag = f'validation_C0_full_{len(stores)}_stores_{run_id}'
    aggregate_dir = output_root / 'validation' / batch_tag
    metrics_dir = output_root / 'metrics' / batch_tag
    aggregate_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=False)
    sources = []
    save_json({'status':'running', 'config':config, 'stores':stores,
               'training_summary':str(training_summary.resolve()), 'batch_tag':batch_tag},
              metrics_dir / 'validation_manifest.json')
    calendar = load_calendar(data_dir)
    for store in stores:
        model_path, info, source_hash = prepared[store]
        bundle = verify_saved_models(model_path)
        items = info['selected_items']
        if (bundle['origin_day'] != 1885 or bundle['features'] != info['features']
                or bundle['categories'] != info['categories']
                or len(set(bundle['categories'])) != 3049
                or set(bundle['categories']) != set(items)):
            raise ValueError(f"Bundle không khớp metadata: {store}")
        LOGGER.info('VALIDATION SAVED C0 | %s | %s', store, model_path)
        wide = load_store(data_dir, store, 1885, chunk_size=config['chunk_size'], item_ids=items)
        if wide.item_id.tolist() != items or set(wide.state_id) != {info['state_id']}:
            raise ValueError(f"Thứ tự mã/bang không khớp mô hình: {store}")
        matrix = history_matrix(wide, 1885)
        del wide
        _, selected, component = training_patterns(matrix, items)
        tag = f'validation_full_{store}_{run_id}'
        locations = {name:output_root / name / tag for name in ['validation','models','metrics','figures']}
        for directory in locations.values():
            directory.mkdir(parents=True, exist_ok=False)
        common = dict(history=matrix, items=items, store=store, state=info['state_id'],
            calendar=calendar, origin=1885, features=bundle['features'], categories=bundle['categories'],
            split='validation', horizon=28)
        prediction, components = recursive_forecast(bundle['model'], **common, return_components=True)
        baseline = recursive_forecast(SeasonalNaive(), **common, model_name='seasonal_naive')
        export_forecasts(prediction, locations['validation'] / 'lightgbm_two_part.csv')
        export_forecasts(baseline, locations['validation'] / 'seasonal_naive.csv')
        # Nhãn Validation chỉ đọc sau khi đã dự báo xong cả hai phương pháp.
        actual = load_store(data_dir, store, 1913, start_day=1886, item_ids=items,
                            chunk_size=config['chunk_size'])
        context = dict(config=config, store_id=store, scope='full', split='validation', origin_day=1885,
            run_id=run_id, batch_tag=batch_tag, illustration_items=selected, component_item=component,
            mode='validation', status='completed', configuration_name='C0', model_reused=True,
            source_model_path=str(model_path.resolve()), source_model_sha256=source_hash,
            forecast_performed=True, evaluation_performed=True)
        evaluate_and_export(prediction, baseline, actual, matrix, items, bundle, info,
                            locations, context, components=components)
        for name in ['lightgbm_two_part.csv', 'seasonal_naive.csv', 'actual_vs_predictions.csv']:
            destination = aggregate_dir / name
            with (locations['validation'] / name).open('rb') as reader, destination.open('ab') as writer:
                if destination.stat().st_size:
                    reader.readline()  # Bỏ header/BOM; giữ nguyên byte dữ liệu, không parse/làm tròn lại.
                shutil.copyfileobj(reader, writer)
        sources.append(dict(store_id=store, run_tag=tag, source_model_path=str(model_path.resolve()),
            source_model_sha256=source_hash, actual_vs_predictions=str((locations['validation']/'actual_vs_predictions.csv').resolve()),
            summary=str((locations['metrics']/'summary.csv').resolve()), figures=str(locations['figures'].resolve())))
        LOGGER.info('VALIDATION COMPLETE | %s | %d dự báo/phương pháp', store, len(prediction))
        del bundle, matrix, common, prediction, baseline, components, actual
        gc.collect()
    workbook = export_validation_workbook(sources, metrics_dir)
    save_json({'status':'completed', 'config':config, 'stores':stores, 'n_items':3049*len(stores),
        'rows_per_model':3049*28*len(stores), 'origin_day':1885, 'target_days':[1886,1913],
        'models_retrained':False, 'test_performed':False, 'sources':sources,
        'training_summary':str(training_summary.resolve()), 'batch_tag':batch_tag,
        'actual_vs_predictions':str((aggregate_dir/'actual_vs_predictions.csv').resolve()),
        'workbook':str(workbook.resolve())}, metrics_dir/'validation_manifest.json')
    LOGGER.info('VALIDATION ALL COMPLETE | %s', workbook)
    return workbook


def build_parser():
    """Định nghĩa các lệnh công khai để người mới có thể xem --help."""
    parser = argparse.ArgumentParser(description="LightGBM Two-Part đệ quy M5, CPU")
    parser.add_argument(
        "--mode", choices=["train", "smoke", "validation", "test"], default="smoke"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--store", default=None)
    group.add_argument("--all-stores", action="store_true")
    parser.add_argument("--scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume-training", action="store_true",
                        help="Chỉ mode train: bỏ qua cửa hàng đã lưu/nạp xác minh với cùng config.")
    parser.add_argument("--from-training-summary",
                        help="Validation Full: nạp chính các bundle từ CSV bàn giao Train, không fit lại.")
    return parser


def main(argv=None):
    """Kiểm tra ý định chạy Test và xử lý tuần tự để hạn chế RAM."""
    # Đọc lựa chọn chạy và kiểm tra điều kiện trước khi bắt đầu sáu giai đoạn.
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.from_training_summary and (args.mode != "validation" or args.scope != "full"):
        parser.error("--from-training-summary chỉ dùng với --mode validation --scope full.")
    if args.resume_training and args.mode != "train":
        parser.error("--resume-training chỉ dùng với --mode train.")
    if args.mode == "train" and args.scope != "full":
        parser.error("Mode train yêu cầu --scope full, không lấy mẫu Smoke.")
    if args.mode == "test" and not args.confirm_final_test:
        parser.error("Test cuối cần --confirm-final-test sau khi đã chốt cấu hình.")
    if args.mode == "smoke" and args.all_stores:
        parser.error("Smoke chỉ chạy một cửa hàng; dùng --store CA_1.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
    )
    root = Path(__file__).resolve().parent
    try:
        config_path = Path(args.config)
        config = load_config(
            config_path if config_path.is_absolute() else root / config_path
        )
        if args.mode == "train":
            if config["use_prices"] or any(
                (config[name]["n_estimators"], config[name]["learning_rate"], config[name]["num_leaves"])
                != (200, 0.05, 31) for name in ["classifier", "regressor"]
            ):
                raise ValueError("Train C0 yêu cầu tắt giá và 200 cây/0.05/31 lá cho cả hai phần.")
        data_dir = root / config["data_dir"]
        output_root = root / config["results_dir"]
        check_files(data_dir, config["use_prices"])
        stores = list_stores(data_dir) if args.all_stores else [args.store or "CA_1"]
        if args.all_stores and len(stores) != 10:
            raise ValueError(f"M5 cần 10 cửa hàng, nhưng file hiện có {len(stores)}.")
        if args.from_training_summary:
            summary_path = Path(args.from_training_summary)
            run_saved_validation(config, data_dir, output_root, stores,
                                 summary_path if summary_path.is_absolute() else root / summary_path)
            return 0
        # run_store chạy đủ các giai đoạn 1–6 cho từng cửa hàng.
        # Chạy tuần tự để không giữ dữ liệu của 10 cửa hàng cùng lúc trong RAM.
        failures = []
        for store in stores:
            try:
                if args.resume_training:
                    completed = False
                    for path in sorted((output_root / "models").glob(f"train_full_{store}_*/metadata.json"), reverse=True):
                        metadata = json.loads(path.read_text(encoding="utf-8"))
                        if (metadata.get("status") == "completed" and metadata.get("config") == config
                                and metadata.get("origin_day") == 1885 and metadata.get("n_items") == 3049
                                and metadata.get("store_id") == store):
                            loaded = verify_saved_models(path.parent / "model.joblib")
                            if (loaded["origin_day"] != 1885 or loaded["features"] != metadata["features"]
                                    or loaded["categories"] != metadata["categories"]
                                    or set(loaded["categories"]) != set(metadata["selected_items"])):
                                raise ValueError(f"Bundle resume sai cấu trúc: {store}")
                            del loaded
                            LOGGER.info("TRAIN RESUME VERIFIED | %s | %s", store, path.parent)
                            completed = True
                            break
                    if completed:
                        continue
                run_store(config, args, data_dir, output_root, store)
            except Exception:
                if args.mode != "train":
                    raise
                failures.append(store)
                LOGGER.exception("TRAIN FAILED | %s; giữ kết quả khác, tiếp tục cửa hàng sau", store)
            finally:
                gc.collect()
        if failures:
            LOGGER.error("Cửa hàng chưa hoàn tất: %s. Chạy lại với --resume-training.", failures)
            return 1
    except MemoryError:
        LOGGER.error(
            "Không đủ RAM. Đóng ứng dụng khác, giảm chunk_size hoặc n_jobs; "
            "thử scope smoke để kiểm tra. Full không tự cắt sản phẩm/lịch sử."
        )
        return 1
    except (FileNotFoundError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
