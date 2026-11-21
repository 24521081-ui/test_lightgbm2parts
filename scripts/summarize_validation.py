"""Tổng hợp hiện vật Validation Full, chỉ đọc dữ liệu, không nạp hay chạy mô hình.

Đọc CSV nguồn một lần theo khối; tính sai số bằng float64. Mỗi lần thực hiện
tạo thư mục riêng, kiểm tra SHA256 để bảo toàn mọi kết quả đã có.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.forecasting import FORECAST_COLUMNS, validate_forecasts
from src.reporting import save_rmsse_comparisons

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
MODELS = ["lightgbm_tweedie", "seasonal_naive"]
METRICS = ["RMSSE_mean_unweighted", "MAE", "RMSE"]
LABELS = {"lightgbm_tweedie": "LightGBM–Tweedie", "seasonal_naive": "Seasonal Naive"}
LOG = logging.getLogger("summary")


def file_hash(path):
    """Tính SHA256 theo khối, không tải tệp lớn vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(data, path):
    """Lưu bằng chứng ở thư mục tổng hợp mới."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pooled_metrics(item_rows):
    """Gộp theo tổng sai số và số điểm, tuyệt đối không lấy trung bình RMSE."""
    rows = []
    for model, group in item_rows.groupby("model", sort=False):
        count = int(group.n_predictions.sum())
        valid = group.RMSSE.notna()
        rows.append({"model": model, "split": "validation", "n_stores": group.store_id.nunique(),
            "n_items": len(group), "n_predictions": count,
            "MAE": group.absolute_error_sum.sum() / count,
            "RMSE": np.sqrt(group.squared_error_sum.sum() / count),
            "RMSSE_mean_unweighted": group.loc[valid, "RMSSE"].mean(),
            "rmsse_valid_items": int(valid.sum()), "rmsse_undefined_items": int((~valid).sum())})
    return pd.DataFrame(rows)


def discover_runs(root):
    """Đọc mọi summary Full Validation; chọn lần hoàn tất mới nhất mỗi cửa hàng."""
    candidates = []
    for path in sorted((root / "results/metrics").glob("validation_full_*/summary.csv")):
        tag = path.parent.name
        metadata_path = root / "results/models" / tag / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"Thiếu metadata của {tag}; cần kiểm tra trước khi tổng hợp.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(path)
        if "RMSSE_mean_unweighted" not in summary:
            summary = summary.rename(columns={"RMSSE": "RMSSE_mean_unweighted"})
        store = metadata["store_id"]
        if store not in STORES or metadata["demo"] or metadata["scope"] != "full":
            raise ValueError(f"Lần chạy không thuộc Full M5 thật: {tag}")
        if (metadata["split"], metadata["origin_day"], metadata["n_items"],
                metadata["n_train_rows"]) != ("validation", 1885, 3049, 5661993):
            raise ValueError(f"Sai thiết kế hoặc quy mô Train: {tag}")
        if set(summary.model) != set(MODELS) or len(summary) != 2:
            raise ValueError(f"Summary không đủ đúng hai mô hình: {tag}")
        if not summary.store_id.eq(store).all() or not summary.split.eq("validation").all() or not summary.n_items.eq(3049).all():
            raise ValueError(f"Summary sai cửa hàng, split hoặc số sản phẩm: {tag}")
        if not np.isfinite(summary[METRICS + ["training_time_seconds"]]).all().all() or (summary[METRICS + ["training_time_seconds"]] < 0).any().any():
            raise ValueError(f"Chỉ số summary không hợp lệ: {tag}")
        files = [root / "results/validation" / tag / f"{model}.csv" for model in MODELS]
        files += [root / "results/metrics" / tag / f"{model}_per_item.csv" for model in MODELS]
        if not all(p.is_file() for p in files):
            raise ValueError(f"Lần chạy chưa đủ hiện vật: {tag}")
        candidates.append({"tag": tag, "store": store, "run_id": metadata["run_id"],
                           "metadata": metadata, "summary": summary})
    selected = {}
    for store in STORES:
        choices = [row for row in candidates if row["store"] == store]
        if not choices:
            raise ValueError(f"Chưa có Validation Full hoàn tất cho {store}.")
        selected[store] = max(choices, key=lambda row: row["run_id"])
    # Chỉ gộp các thí nghiệm có cùng thiết kế, đặc trưng và tham số.
    reference = selected[STORES[0]]["metadata"]
    for row in selected.values():
        metadata = row["metadata"]
        for key in ["features", "config"]:
            if metadata[key] != reference[key]:
                raise ValueError(f"Không đồng nhất {key} giữa các lần chạy, không tự gộp: {row['tag']}")
    return candidates, selected


def load_forecasts(root, selected):
    """Kiểm tra khóa/ngày và chỉ giữ ma trận dự báo 3049×28 cho từng mô hình."""
    matrices = {}
    for store in STORES:
        row = selected[store]
        items = sorted(row["metadata"]["selected_items"])
        if len(items) != 3049 or len(set(items)) != 3049:
            raise ValueError(f"Metadata trùng/thiếu item: {store}")
        for model in MODELS:
            frame = pd.read_csv(root / "results/validation" / row["tag"] / f"{model}.csv")
            if list(frame.columns) != FORECAST_COLUMNS:
                raise ValueError(f"CSV sai thứ tự cột: {store}/{model}")
            validate_forecasts(frame, 28)
            if not (frame.model.eq(model).all() and frame.split.eq("validation").all()
                    and frame.store_id.eq(store).all() and frame.origin_day.eq("d_1885").all()):
                raise ValueError(f"CSV sai model/store/split/origin: {store}/{model}")
            if set(frame.item_id) != set(items) or len(frame) != 85372:
                raise ValueError(f"CSV sai số item hoặc dự báo: {store}/{model}")
            order_ok = frame.groupby("item_id", sort=False).horizon.apply(list).apply(
                lambda values: values == list(range(1, 29))).all()
            if not order_ok:
                raise ValueError(f"Horizon không theo thứ tự 1–28: {store}/{model}")
            matrix = frame.pivot(index="item_id", columns="horizon", values="y_pred").loc[items, range(1, 29)]
            matrices[store, model] = matrix
        LOG.info("Đã kiểm tra CSV %s: 3049 chuỗi, 85372 dự báo/mô hình", store)
    return matrices


def recalculate(root, selected, matrices):
    """Đọc Train/Validation một lần theo khối; không đọc d_1914 trở đi."""
    days = [f"d_{d}" for d in range(1, 1914)]
    data_path = root / selected[STORES[0]]["metadata"]["config"]["data_dir"] / "sales_train_evaluation.csv"
    pieces, seen = [], set()
    for chunk in pd.read_csv(data_path, usecols=["store_id", "item_id"] + days, chunksize=500,
                             dtype={day: "float32" for day in days}):
        for store, part in chunk.groupby("store_id", sort=False):
            if store not in STORES:
                raise ValueError(f"Cửa hàng nguồn không mong đợi: {store}")
            items = part.item_id.tolist()
            for item in items:
                key = (store, item)
                if key in seen:
                    raise ValueError(f"Dữ liệu thật trùng khóa: {key}")
                seen.add(key)
            sales = part[days].to_numpy(dtype="float64")
            if not np.isfinite(sales).all() or (sales < 0).any():
                raise ValueError("Doanh số nguồn thiếu/vô hạn/âm.")
            scales = np.full(len(part), np.nan)
            for index, history in enumerate(sales[:, :1885]):
                positives = np.flatnonzero(history > 0)
                if len(positives) and positives[0] < 1884:
                    scale = np.mean(np.square(np.diff(history[positives[0]:])))
                    if scale > 0:
                        scales[index] = scale
            for model in MODELS:
                prediction = matrices[store, model].loc[items].to_numpy()
                error = sales[:, 1885:] - prediction
                square_sum = np.square(error).sum(axis=1)
                scores = np.full(len(part), np.nan)
                valid = np.isfinite(scales)
                scores[valid] = np.sqrt(square_sum[valid] / 28 / scales[valid])
                if model == "seasonal_naive":
                    expected = np.tile(sales[:, 1878:1885], (1, 4))
                    if not np.array_equal(expected, prediction):
                        raise ValueError(f"Seasonal Naive khác mẫu lặp 7 ngày: {store}")
                pieces.append(pd.DataFrame({"model": model, "store_id": store, "item_id": items,
                    "n_predictions": 28, "absolute_error_sum": np.abs(error).sum(axis=1),
                    "squared_error_sum": square_sum, "scale": scales, "RMSSE": scores}))
    expected_keys = {(store, item) for store in STORES for item in matrices[store, MODELS[0]].index}
    if seen != expected_keys:
        raise ValueError("Tập chuỗi doanh số thật không khớp đủ 30490 chuỗi dự báo.")
    return pd.concat(pieces, ignore_index=True), data_path


def store_tables(root, selected, item_rows):
    """Đối chiếu summary và RMSSE từng chuỗi cũ, xuất số tính lại không sửa tệp cũ."""
    rows, comparisons = [], []
    for store in STORES:
        subset = item_rows.loc[item_rows.store_id == store]
        calculated = pooled_metrics(subset)
        summary = selected[store]["summary"].set_index("model")
        for _, row in calculated.iterrows():
            model = row["model"]
            source = summary.loc[model]
            differences = {metric: float(row[metric] - source[metric]) for metric in METRICS}
            if max(abs(value) for value in differences.values()) > 1e-7:
                raise ValueError(f"Chỉ số không khớp summary: {store}/{model}: {differences}")
            per_item = pd.read_csv(root / "results/metrics" / selected[store]["tag"] / f"{model}_per_item.csv")
            scores = subset.loc[subset.model == model].set_index("item_id")
            if per_item.item_id.duplicated().any() or set(per_item.item_id) != set(scores.index):
                raise ValueError(f"Bảng chi tiết thiếu hoặc trùng item: {store}/{model}")
            old = per_item.set_index("item_id").loc[scores.index]
            if not np.allclose(old.scale, scores.scale, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError(f"Mẫu số RMSSE không khớp: {store}/{model}")
            if not np.allclose(old.RMSSE, scores.RMSSE, rtol=1e-6, atol=1e-7, equal_nan=True):
                raise ValueError(f"RMSSE từng chuỗi không khớp: {store}/{model}")
            record = row.to_dict()
            record.pop("n_stores")
            record.update(store_id=store, training_time_seconds=float(source.training_time_seconds),
                          source_run=selected[store]["tag"])
            rows.append(record)
            comparisons.append({"store_id": store, "model": model,
                **{f"delta_{key}": value for key, value in differences.items()},
                "scale_check": "PASS", "per_item_rmsse_check": "PASS"})
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def improvement_table(by_store):
    """Mức giảm sai số tương đối so với baseline; số dương nghĩa là LightGBM tốt hơn."""
    rows = []
    for store in STORES:
        values = by_store.loc[by_store.store_id == store].set_index("model")
        rows.append({"store_id": store, **{f"{metric}_reduction_percent":
            100 * (1 - values.loc[MODELS[0], metric] / values.loc[MODELS[1], metric]) for metric in METRICS}})
    return pd.DataFrame(rows)


def figures(by_store, overall, output):
    """Vẽ biểu đồ chuẩn để xuất vào luận văn; ghi PNG 300 dpi và PDF."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    save_rmsse_comparisons(by_store, overall, output, STORES, MODELS, LABELS)
    colors = ["#2563a6", "#e39836"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    columns = METRICS + ["training_time_seconds"]
    titles = ["Chỉ số chính: RMSSE trung bình không trọng số", "MAE bổ sung theo cửa hàng", "RMSE bổ sung theo cửa hàng", "Thời gian fit đã ghi nhận"]
    positions = np.arange(len(STORES))
    for ax, metric, title in zip(axes.flat, columns, titles):
        for index, model in enumerate(MODELS):
            frame = by_store.loc[by_store.model == model].set_index("store_id").loc[STORES]
            ax.bar(positions + (index - 0.5) * 0.36, frame[metric], width=0.36,
                   color=colors[index], label=LABELS[model])
        ax.set_xticks(positions, STORES, rotation=35)
        ax.set_title(title)
        ax.set_ylabel("Giây" if metric == "training_time_seconds" else metric)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Validation d_1886–d_1913 • 3.049 chuỗi mỗi cửa hàng", y=0.94)
    fig.text(0.5, 0.01, "Seasonal Naive: thời gian fit ghi bằng 0; chưa đo thời gian dự báo. RMSSE không phải WRMSSE M5.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 0.91])
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"comparison_by_store.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    indexed = overall.set_index("model").loc[MODELS]
    for ax, metric in zip(axes, METRICS):
        bars = ax.bar([LABELS[model] for model in MODELS], indexed[metric], color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_ylim(0, indexed[metric].max() * 1.18)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Kết quả gộp: 30.490 chuỗi • 853.720 dự báo/mô hình")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"comparison_overall.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame):
    """Tạo bảng Markdown mà không yêu cầu cài thêm thư viện tabulate."""
    def cell(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def report(by_store, overall, gains, inventory, checks, output):
    """Soạn phần kết quả và giới hạn bằng tiếng Việt, có thể trích sang Chương 5."""
    global_values = overall.set_index("model")
    lgb, naive = global_values.loc[MODELS[0]], global_values.loc[MODELS[1]]
    reduction = {key: 100 * (1 - lgb[key] / naive[key]) for key in METRICS}
    best = by_store.loc[by_store.model == MODELS[0]].sort_values("RMSSE_mean_unweighted")
    run_inventory = inventory[["store_id", "source_run"]].drop_duplicates()
    duplicate_stores = run_inventory.loc[run_inventory.duplicated("store_id", keep=False), "store_id"].unique()
    view = by_store[["store_id", "model"] + METRICS + ["training_time_seconds"]]
    text = f"""# Chương 5 — Kết quả Validation Full trên 10 cửa hàng Walmart M5

## 5.1. Phạm vi và dữ liệu thực nghiệm

Báo cáo được tổng hợp từ dự báo và hiện vật đã lưu, không huấn luyện lại mô hình và không chạy Test cuối. Train dùng d_1–d_1885; Validation dùng d_1886–d_1913, dự báo đệ quy 28 ngày. Có 10 cửa hàng, mỗi cửa hàng 3.049 sản phẩm: **30.490 chuỗi sản phẩm–cửa hàng, 853.720 dự báo mỗi mô hình**, tổng 1.707.440 dự báo của hai mô hình.

Các lần được chọn có cùng bộ đặc trưng, cùng cấu hình LightGBM–Tweedie và đều tắt giá bán. Mỗi cửa hàng dùng một LightGBM chung; baseline Seasonal Naive chu kỳ 7 ngày. Kết quả chỉ mô tả cửa sổ Validation này.

Đọc tất cả {len(run_inventory)} summary Full Validation hiện có. Với cửa hàng có nhiều lần hoàn tất ({', '.join(duplicate_stores)}), chọn lần mới nhất theo run_id UTC, không chọn theo điểm số. CA_1 cũ được loại khỏi phép gộp để tránh đếm hai lần, vẫn giữ nguyên mọi tệp. Danh sách chọn/loại và đường dẫn nguồn có trong `source_runs.csv`.

## 5.2. Cách tính và kiểm tra số liệu

### Vai trò các chỉ số và kiểm tra mã nguồn

**Chỉ số so sánh chính của báo cáo là `RMSSE_mean_unweighted`; MAE và RMSE là chỉ số bổ sung.** Việc đổi thứ tự trình bày không thay đổi công thức đánh giá hoặc cấu hình đã huấn luyện.

- `configs/config.yaml`: `objective: tweedie`, `tweedie_variance_power: 1.2`, `n_estimators: 200`. `run.py::load_config` kiểm tra objective và miền 1 < power < 2.
- `src/training.py::train_model`: truyền tham số vào `LGBMRegressor(**parameters)` và gọi `fit(X, y, categorical_feature=["item_id"])`. Mục tiêu tối ưu là loss Tweedie; không dùng MAE, RMSE hoặc RMSSE làm objective. Không truyền `eval_set`, `eval_metric` hay callback early stopping.
- `run.py` trực tiếp điều phối 6 giai đoạn cho mỗi cửa hàng; mỗi cửa hàng fit một cấu hình cố định. Không có vòng tìm kiếm tham số, so sánh cấu hình, chọn best iteration hay chọn mô hình tốt nhất tự động. Không có bằng chứng trong luồng mã này rằng MAE quyết định tham số hoặc kết quả được giữ lại. Cấu hình được ấn định sẵn trong YAML; mã nguồn không chứng minh tiêu chí người dùng đã dùng để chọn cấu hình đó trước đây.
- `src/reporting.py` tính ba chỉ số sau khi có dự báo. `src/reporting.py` giữ biểu đồ MAE để tham khảo và thêm biểu đồ RMSSE chính. Biểu đồ/chỉ số không điều khiển huấn luyện hoặc chọn cấu hình.
- `scripts/summarize_validation.py::discover_runs` chọn lần hoàn tất mới nhất theo `run_id` ở từng cửa hàng để tránh đếm trùng. Đây là chọn hiện vật để tổng hợp theo thời gian, không phải chọn mô hình theo MAE hay RMSSE. Xếp thứ tự cửa hàng theo RMSSE trong phần phân tích chỉ mô tả kết quả.

Với sai số `e = y - y_hat`, N điểm đánh giá, horizon H = 28, T là ngày cuối Train và k là ngày bán dương đầu tiên của chuỗi i, mã hiện tại thực hiện đúng:

```text
MAE = sum(abs(e)) / N
RMSE = sqrt(sum(e^2) / N)
scale_i = sum((y_i,t - y_i,t-1)^2, t=k+1..T) / (T-k)
RMSSE_i = sqrt((sum(e_i,h^2, h=1..H) / H) / scale_i)
RMSSE_mean_unweighted = sum(RMSSE_i với scale xác định) / số chuỗi hợp lệ
```

Không lấy căn sau khi gộp các tỷ số MSE/scale của nhiều chuỗi. Chuỗi toàn 0, hằng hoặc còn ít hơn hai điểm từ k có scale không xác định; mã trả NaN, loại riêng khi lấy trung bình RMSSE và báo số lượng, không thêm epsilon. MAE/RMSE vẫn dùng mọi điểm đánh giá. Không phát hiện lỗi công thức trong phạm vi kiểm tra nên giữ nguyên `reporting.py`, tham số huấn luyện và dữ liệu gốc.

WRMSSE chính thức đánh giá các chuỗi ở đủ 12 cấp phân cấp và dùng trọng số doanh thu từ 28 ngày cuối Train. Trung bình đều RMSSE của các chuỗi cấp đáy hiện tại không có các thành phần đó và không được diễn giải thành điểm WRMSSE hoặc thứ hạng M5.

Doanh số thật d_1–d_1913 được đọc từ CSV nguồn theo khối, không đọc 28 nhãn Test cuối. Dự báo được căn theo store_id, item_id và horizon. Kiểm tra mỗi tệp đúng tám cột quy định, 3.049 item, 28 horizon liên tục từ 1 đến 28, origin d_1885, không thiếu/trùng khóa, NaN, vô cực hoặc âm. Hai mô hình dùng cùng tập chuỗi và nhãn thật. Seasonal Naive được đối chiếu chính xác với bảy ngày cuối Train lặp bốn lần.

- **MAE gộp** = tổng sai số tuyệt đối trên tất cả cửa hàng chia tổng số dự báo.
- **RMSE gộp** = căn của tổng bình phương sai số chia tổng số dự báo. **Không lấy trung bình đơn giản của RMSE theo cửa hàng.** Cách tương đương là `sqrt(sum(N_s * RMSE_s²) / sum(N_s))` nếu dùng cùng độ chính xác dữ liệu.
- **RMSSE_mean_unweighted** = trung bình số học của RMSSE từng chuỗi. Scale từng chuỗi là trung bình bình phương sai phân Train kể từ lần bán dương đầu tiên, chỉ dựa trên d_1–d_1885. Không bỏ tiền tố 0 khỏi dữ liệu huấn luyện.

Đã tính lại RMSSE từng chuỗi và đối chiếu scale/RMSSE với bảng chi tiết đã lưu. Các chỉ số theo cửa hàng tính lại khớp summary trong ngưỡng 1e-7; độ lệch lớn nhất là {checks.filter(like='delta_').abs().to_numpy().max():.3g}. Độ lệch rất nhỏ có thể xuất hiện khi chuỗi thập phân của dự báo float32 được đọc lại bằng float64. Bảng mới dùng trực tiếp dự báo thập phân trong CSV, tính sai số bằng float64, không làm tròn dự báo.

**Chỉ số RMSSE ở đây không phải WRMSSE chính thức của M5**: chưa đánh giá đủ 12 cấp phân cấp hoặc dùng trọng số doanh thu. Số chuỗi RMSSE hợp lệ và không xác định được ghi trong các CSV. Nếu có scale bằng 0/không đủ lịch sử thì RMSSE không xác định và phải báo riêng; kết quả thực tế có {int(lgb.rmsse_valid_items):,} chuỗi hợp lệ và {int(lgb.rmsse_undefined_items)} không xác định cho mỗi mô hình.

## 5.3. Kết quả theo cửa hàng

Các giá trị dưới đây được tính lại từ dự báo; thời gian fit lấy nguyên từ summary của lần chạy được chọn. Dấu chấm là dấu thập phân, các giá trị hiển thị sáu chữ số thập phân.

{markdown_table(view)}

![Chỉ số chính: RMSSE trung bình theo cửa hàng](rmsse_mean_by_store.png)

Biểu đồ bổ sung giữ MAE, RMSE và thời gian fit:

![So sánh theo cửa hàng](comparison_by_store.png)

Thời gian của LightGBM chỉ đo `model.fit`, chưa gồm đọc dữ liệu, xây đặc trưng, dự báo và đánh giá. Seasonal Naive có `training_time_seconds=0` theo quy ước vì không có bước fit; **không có nghĩa toàn bộ thời gian chạy hoặc thời gian dự báo bằng 0**.

## 5.4. Kết quả gộp trên toàn bộ 30.490 chuỗi

{markdown_table(overall[['model', 'n_items', 'n_predictions'] + METRICS + ['training_time_seconds']])}

Cột thời gian trong bảng gộp là **tổng thời gian fit** của 10 mô hình cửa hàng được chọn, không phải thời gian thực hiện toàn bộ quy trình. Tổng fit LightGBM là {lgb.training_time_seconds:.6f} giây, tương đương {lgb.training_time_seconds / 60:.2f} phút; thời gian fit trung bình là {lgb.training_time_seconds / 10:.6f} giây/cửa hàng.

Theo chỉ số chính, LightGBM–Tweedie giảm **RMSSE_mean_unweighted {reduction['RMSSE_mean_unweighted']:.2f}%** so với Seasonal Naive. Hai chỉ số bổ sung giảm **MAE {reduction['MAE']:.2f}%** và **RMSE {reduction['RMSE']:.2f}%** trên cửa sổ Validation đang xét. Mức giảm tính bằng `100 × (1 − chỉ_số_LightGBM / chỉ_số_baseline)`.

![Chỉ số chính: RMSSE trung bình toàn bộ 10 cửa hàng](rmsse_mean_overall.png)

![So sánh kết quả gộp](comparison_overall.png)

## 5.5. Phân tích và giới hạn

LightGBM có sai số thấp hơn baseline ở {int((gains.filter(like='_reduction_percent') > 0).all(axis=1).sum())}/10 cửa hàng xét đồng thời cả ba chỉ số. Theo RMSSE trung bình của LightGBM, cửa hàng có điểm thấp nhất là **{best.iloc[0].store_id} ({best.iloc[0].RMSSE_mean_unweighted:.6f})**, cao nhất là **{best.iloc[-1].store_id} ({best.iloc[-1].RMSSE_mean_unweighted:.6f})**. MAE/RMSE tuyệt đối còn phụ thuộc quy mô lượng bán, nên không nên dùng riêng hai chỉ số này để kết luận cửa hàng nào dễ dự báo hơn.

Kết quả cho thấy mô hình có khả năng cải thiện so với baseline tuần trên dữ liệu và giai đoạn đã kiểm tra. Một lần chia thời gian chưa đủ chứng minh mức cải thiện ổn định trên các giai đoạn khác; chưa có kiểm định ý nghĩa thống kê, nhiều cửa sổ backtest hoặc đo chi phí tồn kho/thất thoát. Không diễn giải trực tiếp phần trăm giảm sai số thành phần trăm giảm tồn kho hay chi phí bổ sung hàng. Chưa sử dụng Test cuối để lựa chọn hay xác nhận cấu hình.

Đây là bản LightGBM–Tweedie tự triển khai với cấu hình hiện tại, không phải tái lập đầy đủ phương pháp đội thắng M5. Không suy ra thứ hạng M5 từ RMSSE trung bình của 30.490 chuỗi cấp đáy.

## 5.6. Tệp bàn giao và khả năng truy vết

- `metrics_by_store.csv`: 20 dòng, đủ hai mô hình ở 10 cửa hàng; tên lần chạy nguồn và số quan sát.
- `metrics_overall.csv`: hai dòng; MAE/RMSE gộp, RMSSE trung bình, tổng thời gian fit.
- `metrics_per_series.csv`: 60.980 dòng, gồm scale, RMSSE và tổng sai số từng chuỗi/mô hình.
- `improvement_by_store.csv`: mức giảm ba loại sai số so với Seasonal Naive.
- `source_runs.csv`: toàn bộ lần Full Validation phát hiện, cờ chọn/loại và số liệu summary gốc.
- `summary_crosscheck.csv`: chênh lệch giữa tính lại và summary, kiểm tra scale/RMSSE từng chuỗi.
- `comparison_by_store.png/.pdf`, `comparison_overall.png/.pdf`: biểu đồ PNG 300 dpi và bản PDF để đưa vào luận văn.
- `rmsse_mean_overall.png/.pdf`, `rmsse_mean_by_store.png/.pdf`: hai biểu đồ riêng cho chỉ số so sánh chính.
- `provenance.json`, `existing_results_sha256.json`, `integrity_check.json`: thiết kế, đường dẫn, dấu vân tay tệp và kiểm tra bảo toàn.

Mọi tệp cũ được giữ nguyên. Mỗi lần tổng hợp tạo một thư mục mới bên trong `results/aggregated/`; tệp `integrity_check.json` ghi kết quả so sánh SHA256 trước/sau. Script không import module huấn luyện, không gọi `fit()` hoặc `predict()`.
"""
    (output / "CHUONG_5_VALIDATION_REPORT.md").write_text(text, encoding="utf-8")


def main(argv=None):
    """Điểm vào: chỉ tổng hợp Validation, không có tùy chọn mở Test cuối."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    candidates, selected = discover_runs(root)
    # Chụp mọi tệp kết quả đã tồn tại, trước khi tạo thư mục đầu ra mới.
    existing = {str(p.relative_to(root)): file_hash(p) for p in (root / "results").rglob("*") if p.is_file()}
    output = root / "results/aggregated" / ("validation_full_10_stores_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(existing, output / "existing_results_sha256.json")
    matrices = load_forecasts(root, selected)
    LOG.info("Tính lại sai số và scale từ CSV doanh số thật, chỉ đến d_1913")
    item_rows, data_path = recalculate(root, selected, matrices)
    by_store, checks = store_tables(root, selected, item_rows)
    overall = pooled_metrics(item_rows)
    if not overall.n_items.eq(30490).all() or not overall.n_predictions.eq(853720).all():
        raise ValueError("Quy mô tổng hợp không đủ 30490 chuỗi hoặc 853720 dự báo/mô hình.")
    times = by_store.groupby("model").training_time_seconds.sum()
    overall["training_time_seconds"] = overall.model.map(times)
    overall["training_time_mean_per_store_seconds"] = overall.training_time_seconds / 10
    inventory_rows = []
    for candidate in candidates:
        for _, row in candidate["summary"].iterrows():
            inventory_rows.append({**row.to_dict(), "source_run": candidate["tag"],
                "selected": selected[candidate["store"]]["tag"] == candidate["tag"],
                "selection_rule": "latest_completed_run_per_store"})
    inventory = pd.DataFrame(inventory_rows)
    gains = improvement_table(by_store)
    for name, frame in {"metrics_by_store": by_store, "metrics_overall": overall,
            "metrics_per_series": item_rows, "source_runs": inventory,
            "summary_crosscheck": checks, "improvement_by_store": gains}.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    figures(by_store, overall, output)
    report(by_store, overall, gains, inventory, checks, output)
    write_json({"selected_runs": {s: selected[s]["tag"] for s in STORES},
        "origin_day": 1885, "validation_days": [1886, 1913], "n_stores": 10,
        "n_series_per_model": 30490, "n_predictions_per_model": 853720,
        "input_sales_file": str(data_path), "input_sales_sha256": file_hash(data_path),
        "config": selected[STORES[0]]["metadata"]["config"],
        "features": selected[STORES[0]]["metadata"]["features"],
        "numeric_policy": "CSV predictions parsed float64; errors/scales/aggregation float64",
        "training_time_policy": "sum of recorded per-store fit seconds; excludes prediction and preprocessing",
        "official_wrmsse": False, "retrained": False, "final_test_run": False}, output / "provenance.json")
    changed = [name for name, digest in existing.items() if file_hash(root / name) != digest]
    write_json({"n_existing_files": len(existing), "n_changed": len(changed), "changed": changed}, output / "integrity_check.json")
    if changed:
        raise RuntimeError("Có tệp kết quả cũ thay đổi trong quá trình tổng hợp.")
    LOG.info("Hoàn tất: %s", output)
    print(overall.to_string(index=False))
    return output


if __name__ == "__main__":
    main()
