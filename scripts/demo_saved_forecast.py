r"""Demo dự báo Validation bằng mô hình đã lưu; không fit, không chạy Test cuối.

Chạy từ thư mục project bằng PowerShell:
    & ..\.venv\Scripts\python.exe scripts/demo_saved_forecast.py

Luồng thực hiện:
    Nạp mô hình -> đọc lịch sử -> dự báo -> đối chiếu dự báo đã lưu
    -> đọc thực tế Validation -> tính sai số -> lưu CSV và biểu đồ.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# File nằm trong scripts/, nên đi lên hai cấp để tìm thư mục project.
# Thêm project vào đường dẫn import để gọi các chức năng trong src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_calendar, load_store
from src.reporting import evaluate_forecast
from src.preprocessing import history_matrix
from src.forecasting import SeasonalNaive, recursive_forecast
from src.forecasting import export_forecasts
from src.reporting import save_json
# visualization đã cấu hình backend lưu hình không cần mở cửa sổ đồ họa.
from src.reporting import plt


def main(argv=None):
    """Thực hiện lần lượt 12 bước demo cho một sản phẩm của một cửa hàng."""

    # BƯỚC 1: Đọc lựa chọn người dùng truyền từ Terminal.
    # --run chọn lần huấn luyện đã lưu; --item chọn sản phẩm muốn dự báo.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="validation_full_CA_1_20261106T152044652294Z",
        help="Tên thư mục mô hình Validation đã có trong results/models",
    )
    parser.add_argument(
        "--item",
        help="Mã sản phẩm; mặc định là mã đầu tiên theo alphabet",
    )
    args = parser.parse_args(argv)

    # Chỉ nhận tên lần chạy nội bộ, không nhận đường dẫn mô hình bên ngoài.
    if Path(args.run).name != args.run or not args.run.startswith("validation_full_"):
        parser.error("--run phải là tên một lần Validation Full, không phải đường dẫn.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # BƯỚC 2: Đọc metadata để kiểm tra mô hình và thiết kế thí nghiệm.
    # Chỉ dùng Validation thật, origin d_1885, horizon 28 và không dùng giá bán.
    model_dir = ROOT / "results/models" / args.run
    metadata_path = model_dir / "metadata.json"
    metadata_text = metadata_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)

    experiment = (
        metadata["split"],
        metadata["origin_day"],
        metadata["demo"],
        metadata["config"]["horizon"],
    )

    if experiment != ("validation", 1885, False, 28):
        parser.error("Demo chỉ hỗ trợ mô hình Validation thật, origin 1885, horizon 28.")

    if metadata["config"]["use_prices"]:
        parser.error("Demo này dùng các mô hình đã lưu không sử dụng giá bán.")

    # BƯỚC 3: Chọn một sản phẩm đã xuất hiện trong tập huấn luyện.
    # Nếu không truyền --item thì chọn mã đầu theo alphabet, không chọn theo điểm số.
    items = [args.item or sorted(metadata["selected_items"])[0]]

    if items[0] not in metadata["selected_items"]:
        parser.error("Sản phẩm không thuộc tập sản phẩm của mô hình đã lưu.")

    # BƯỚC 4: Nạp model.joblib nội bộ của project, không huấn luyện lại.
    # bundle chứa mô hình, thứ tự đặc trưng và các mã sản phẩm dạng phân loại.
    logging.info(
        "Nạp mô hình đã lưu: %s | sản phẩm %s (không huấn luyện)",
        args.run,
        items[0],
    )
    bundle = joblib.load(model_dir / "model.joblib")

    # BƯỚC 5: Đọc lịch sử thật đến d_1885 và lịch ngày.
    # matrix có dạng số sản phẩm × số ngày lịch sử (ở đây là 1 × 1885).
    # Chưa đọc doanh số thật của 28 ngày sắp dự báo.
    data_dir = ROOT / metadata["config"]["data_dir"]
    store = metadata["store_id"]

    history = load_store(
        data_dir,
        store,
        1885,
        item_ids=items,
    )
    matrix = history_matrix(history, 1885)
    calendar = load_calendar(data_dir)

    # BƯỚC 6: Dự báo bằng LightGBM–Tweedie và Seasonal Naive.
    # common chứa cùng các đầu vào để hai mô hình dự báo trên cùng điều kiện.
    # recursive_forecast gọi feature_engineering tạo đặc trưng cho mỗi ngày.
    # Kết quả dự báo được thêm vào lịch sử tạm để dự báo ngày tiếp theo.
    common = {
        "history": matrix,
        "items": items,
        "store": store,
        "state": history.state_id.iloc[0],
        "calendar": calendar,
        "origin": 1885,
        "features": bundle["features"],
        "categories": bundle["categories"],
        "split": "validation",
        "horizon": 28,
    }

    # **common truyền từng đầu vào trong dictionary vào hàm dự báo.
    prediction = recursive_forecast(
        bundle["model"],
        **common,
    )

    # Seasonal Naive lấy lag_7, tương đương lặp tuần cuối Train bốn lần.
    baseline = recursive_forecast(
        SeasonalNaive(),
        **common,
        model_name="seasonal_naive",
    )

    # BƯỚC 7: Đối chiếu cả hai dự báo với CSV của cùng lần chạy nguồn.
    # Ghép theo mã sản phẩm và horizon để so sánh đúng từng dự báo.
    # Cho phép chênh lệch rất nhỏ do biểu diễn số khi lưu và đọc CSV.
    differences = {}

    for forecast in [prediction, baseline]:
        model = forecast.model.iloc[0]
        saved_path = ROOT / "results/validation" / args.run / f"{model}.csv"
        old = pd.read_csv(saved_path)
        old = old.loc[old.item_id.isin(items)]

        joined = forecast.merge(
            old,
            on=["item_id", "horizon"],
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )

        if len(joined) != 28 or not np.allclose(
            joined.y_pred_new,
            joined.y_pred_old,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(f"Dự báo phát lại khác kết quả đã lưu: {model}")

        absolute_difference = np.abs(joined.y_pred_new - joined.y_pred_old)
        differences[model] = float(absolute_difference.max())

    # BƯỚC 8: Đọc doanh số thật Validation và tính sai số của hai mô hình.
    # Chỉ đọc sau khi dự báo xong để không dùng nhãn tương lai làm đầu vào.
    # Khoảng đọc là d_1886–d_1913, không đọc Test cuối.
    actual = load_store(
        data_dir,
        store,
        1913,
        start_day=1886,
        item_ids=items,
    )

    results = []

    for forecast in [prediction, baseline]:
        # Hàm trả về chỉ số, bảng từng chuỗi, thực tế và dự báo.
        # Demo chỉ cần chỉ số; dấu _ bỏ qua ba kết quả còn lại.
        score, _, _, _ = evaluate_forecast(
            forecast,
            actual,
            matrix,
            items,
        )

        # Không đo fit nên bỏ cột này, tránh hiểu nhầm mô hình fit trong 0 giây.
        score.pop("training_time_seconds")
        results.append(score)

    # BƯỚC 9: Tạo thư mục mới và lưu các bảng CSV.
    # Tên thư mục dùng thời điểm UTC; không ghi đè thư mục kết quả đã có.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results/demo_predictions" / timestamp
    output.mkdir(parents=True, exist_ok=False)

    # Hai tệp dự báo giữ định dạng tám cột chuẩn của project.
    export_forecasts(prediction, output / "lightgbm_tweedie.csv")
    export_forecasts(baseline, output / "seasonal_naive.csv")

    # Bảng minh họa: mỗi dòng là một ngày, gồm hai dự báo và doanh số thật.
    comparison_columns = ["item_id", "target_day", "horizon", "y_pred"]
    comparison = prediction[comparison_columns].rename(
        columns={"y_pred": "lightgbm_tweedie"},
    )
    comparison["seasonal_naive"] = baseline.y_pred.to_numpy()

    validation_days = [f"d_{day}" for day in range(1886, 1914)]
    comparison["actual"] = actual[validation_days].to_numpy().ravel()

    # Nối mã ngày M5 với ngày lịch để người xem dễ đọc kết quả.
    comparison = comparison.merge(
        calendar[["d", "date"]],
        left_on="target_day",
        right_on="d",
        validate="one_to_one",
    )
    comparison = comparison.drop(columns="d")
    comparison.to_csv(
        output / "comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Các chỉ số chỉ mô tả một sản phẩm trong 28 ngày của demo.
    metrics = pd.DataFrame(results)
    metrics.to_csv(
        output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # BƯỚC 10: Vẽ ba đường để so sánh thực tế với dự báo hai mô hình.
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_series = [
        ("actual", "Thực tế"),
        ("lightgbm_tweedie", "LightGBM–Tweedie"),
        ("seasonal_naive", "Seasonal Naive"),
    ]

    for name, label in plot_series:
        ax.plot(
            comparison.horizon,
            comparison[name],
            marker="o",
            markersize=3,
            label=label,
        )

    ax.set(
        title=(
            f"Demo phát lại Validation: {store} / {items[0]}\n"
            "Mô hình đã lưu; không huấn luyện lại"
        ),
        xlabel="Horizon (ngày từ origin d_1885)",
        ylabel="Lượng bán",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "prediction_demo.png", dpi=220)
    plt.close(fig)

    # BƯỚC 11: Lưu thông tin nguồn để truy vết và kiểm tra lại buổi demo.
    # Ghi rõ nguồn mô hình, sản phẩm, mốc dự báo và sai lệch với CSV cũ.
    provenance = {
        "source_run": args.run,
        "store_id": store,
        "items": items,
        "origin_day": 1885,
        "forecast_days": [1886, 1913],
        "retrained": False,
        "final_test_run": False,
        "max_difference_from_saved_csv": differences,
        "selection_rule": "explicit item or first alphabetically; no metric selection",
        "scope": "one series demonstration, not full-store evaluation",
    }
    save_json(provenance, output / "provenance.json")

    # BƯỚC 12: In chỉ số và đường dẫn thư mục để người dùng mở kết quả.
    print(metrics.to_string(index=False))
    print(f"OUTPUT_DIR={output}")

    return output


# Chỉ chạy demo khi gọi trực tiếp file; import file không tự chạy main().
if __name__ == "__main__":
    main()
