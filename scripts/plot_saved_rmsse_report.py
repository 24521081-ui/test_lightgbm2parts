"""Vẽ RMSSE từ CSV Validation đã lưu; không nạp mô hình hoặc dữ liệu gốc."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
MODELS = ["lightgbm_tweedie", "seasonal_naive"]
LABELS = ["LightGBM–Tweedie", "Seasonal Naive"]
METRIC = "RMSSE_mean_unweighted"
NOTE = "RMSSE trung bình không trọng số, không phải WRMSSE chính thức"


def sha256(path):
    """Ghi dấu vân tay để kiểm tra các nguồn đã đọc không bị thay đổi."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_verified_aggregate():
    """Chọn bản tổng hợp mới nhất đã có biên bản kiểm chứng thành công."""
    candidates = sorted(
        (ROOT / "results/aggregated").glob("validation_full_10_stores_*"),
        reverse=True,
    )
    for directory in candidates:
        integrity_path = directory / "integrity_check.json"
        required = ["metrics_overall.csv", "metrics_by_store.csv",
                    "metrics_per_series.csv", "summary_crosscheck.csv"]
        if not integrity_path.is_file() or not all((directory / n).is_file() for n in required):
            continue
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity.get("n_changed") == 0 and integrity.get("changed") == []:
            return directory
    raise ValueError("Không tìm thấy tổng hợp Validation Full đã kiểm chứng.")


def main():
    # Bước 1: Chỉ đọc CSV kết quả và biên bản kiểm chứng trong results/.
    source = latest_verified_aggregate()
    input_paths = [source / name for name in [
        "metrics_overall.csv", "metrics_by_store.csv", "metrics_per_series.csv",
        "summary_crosscheck.csv", "integrity_check.json",
    ]]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    overall = pd.read_csv(source / "metrics_overall.csv")
    by_store = pd.read_csv(source / "metrics_by_store.csv")
    series = pd.read_csv(source / "metrics_per_series.csv")
    crosscheck = pd.read_csv(source / "summary_crosscheck.csv")

    # Bước 2: Kiểm tra đủ hai mô hình, mười cửa hàng và đối chiếu RMSSE từng chuỗi.
    if len(overall) != 2 or set(overall.model) != set(MODELS):
        raise ValueError("Bảng tổng hợp không có đúng hai mô hình.")
    if len(by_store) != 20 or by_store.duplicated(["model", "store_id"]).any():
        raise ValueError("Bảng cửa hàng thiếu hoặc trùng khóa.")
    if not overall.split.eq("validation").all() or not by_store.split.eq("validation").all():
        raise ValueError("Chỉ hỗ trợ Validation.")
    if not overall.n_stores.eq(10).all() or not overall.n_items.eq(30490).all():
        raise ValueError("Không đúng phạm vi 10 cửa hàng / 30.490 chuỗi.")
    if len(series) != 60980 or series.duplicated(["model", "store_id", "item_id"]).any():
        raise ValueError("Bảng từng chuỗi thiếu hoặc trùng khóa.")
    if set(series.model) != set(MODELS) or not series.n_predictions.eq(28).all():
        raise ValueError("Bảng từng chuỗi sai mô hình hoặc horizon.")
    delta_columns = crosscheck.filter(like="delta_")
    if len(crosscheck) != 20 or delta_columns.empty:
        raise ValueError("Thiếu đối chiếu với summary nguồn.")
    if not np.isfinite(delta_columns.to_numpy()).all() or (delta_columns.abs() > 1e-7).any().any():
        raise ValueError("Đối chiếu summary không đạt ngưỡng 1e-7.")

    indexed = overall.set_index("model").loc[MODELS]
    for model in MODELS:
        rows = series.loc[series.model == model]
        stores = by_store.loc[by_store.model == model].set_index("store_id")
        if len(rows) != 30490 or set(stores.index) != set(STORES):
            raise ValueError("Thiếu chuỗi hoặc cửa hàng.")
        valid = np.isfinite(rows.scale) & (rows.scale > 0)
        expected_scores = np.sqrt(
            rows.loc[valid, "squared_error_sum"] / 28 / rows.loc[valid, "scale"]
        )
        if not np.allclose(expected_scores, rows.loc[valid, "RMSSE"], rtol=1e-10, atol=1e-12):
            raise ValueError("RMSSE từng chuỗi không khớp tổng sai số / scale.")
        if rows.loc[~valid, "RMSSE"].notna().any():
            raise ValueError("Chuỗi scale không xác định có điểm RMSSE.")
        if not np.isclose(rows.RMSSE.mean(), indexed.loc[model, METRIC], rtol=1e-10):
            raise ValueError("RMSSE tổng hợp không khớp trung bình từng chuỗi.")
        for store in STORES:
            subset = rows.loc[rows.store_id == store]
            if len(subset) != 3049 or not np.isclose(
                subset.RMSSE.mean(), stores.loc[store, METRIC], rtol=1e-10
            ):
                raise ValueError("RMSSE theo cửa hàng không khớp.")

    # Bước 3: Kiểm tra schema CSV dự báo nguồn, không đọc dữ liệu sales gốc.
    # Các tệp hiện tại chỉ có y_pred; tổng lỗi 28 ngày không đủ để tách theo ngày.
    forecast_schemas = []
    for row in by_store.itertuples(index=False):
        run = row.source_run
        if Path(run).name != run or not run.startswith(f"validation_full_{row.store_id}_"):
            raise ValueError("Tên lần chạy nguồn không hợp lệ.")
        path = ROOT / "results/validation" / run / f"{row.model}.csv"
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        forecast_schemas.append({"path": str(path.relative_to(ROOT)), "columns": columns})
    if any("y_true" in entry["columns"] or "actual" in entry["columns"] for entry in forecast_schemas):
        raise ValueError("Schema nguồn đã thay đổi; cần kiểm tra dữ liệu horizon trước khi vẽ.")

    # Bước 4: Tạo thư mục mới. Nếu đã tồn tại báo cáo thì dùng thư mục con mới.
    output = ROOT / "results/figures/rmsse_report"
    if output.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = output / timestamp
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    colors = ["#2563a6", "#e39836"]

    def save_figure(fig, name):
        fig.text(0.5, 0.02, NOTE, ha="center", fontsize=10)
        fig.tight_layout(rect=[0, 0.065, 1, 1])
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Bước 5: Biểu đồ RMSSE toàn bộ chuỗi; dùng giá trị đầy đủ trong CSV.
    reduction = 100 * (1 - indexed.loc[MODELS[0], METRIC] / indexed.loc[MODELS[1], METRIC])
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(LABELS, indexed[METRIC], color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.6f", padding=6)
    ax.set_ylim(0, indexed[METRIC].max() * 1.28)
    ax.set_title("So sánh RMSSE trung bình trên 10 cửa hàng – Validation", pad=18)
    ax.set_xlabel("Mô hình")
    ax.set_ylabel("RMSSE trung bình không trọng số")
    reduction_label = f"{reduction:.2f}".replace(".", ",")
    reduction_text = f"LightGBM–Tweedie giảm {reduction_label}% sai số so với Seasonal Naive"
    ax.text(0.5, 0.96, reduction_text,
            transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save_figure(fig, "rmsse_comparison_overall")

    # Bước 6: Hai cột mỗi cửa hàng, giữ nguyên thứ tự CA -> TX -> WI.
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = np.arange(len(STORES))
    for index, model in enumerate(MODELS):
        values = by_store.loc[by_store.model == model].set_index("store_id").loc[STORES, METRIC]
        bars = ax.bar(positions + (index - 0.5) * 0.38, values,
                      width=0.38, color=colors[index], label=LABELS[index])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(positions, STORES)
    ax.set_ylim(0, by_store[METRIC].max() * 1.22)
    ax.set_title("So sánh RMSSE trung bình theo cửa hàng – Validation", pad=18)
    ax.set_xlabel("Cửa hàng")
    ax.set_ylabel("RMSSE trung bình không trọng số")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save_figure(fig, "rmsse_comparison_by_store")

    # Bước 7: Ghi nguồn và giới hạn. Không vẽ đường horizon khi thiếu nhãn thật.
    source_list = "\n".join(f"- `{entry['path']}`" for entry in forecast_schemas)
    readme = f"""# Biểu đồ RMSSE — Validation Full 10 cửa hàng

## Phạm vi và nguồn

Nguồn tổng hợp mới nhất đã được kiểm chứng: `{source.relative_to(ROOT)}`.
Validation d_1886–d_1913; 30.490 chuỗi sản phẩm–cửa hàng, 853.720 dự báo/mô hình.
Không huấn luyện lại, không nạp mô hình, không chạy Test cuối, không đọc dữ liệu
gốc trong data/ hoặc dataset/. Chỉ đọc CSV kết quả và JSON biên bản kiểm chứng.
Không sử dụng CSV demo hoặc dữ liệu giả lập.

- `metrics_overall.csv`: điểm RMSSE tổng hợp, giữ đầy đủ độ chính xác khi vẽ.
- `metrics_by_store.csv`: điểm RMSSE của hai mô hình ở mỗi cửa hàng, tên lần chạy nguồn.
- `metrics_per_series.csv`: RMSSE, scale Train và tổng bình phương sai số của 28 ngày.
- `summary_crosscheck.csv`: đối chiếu chỉ số tính lại với summary nguồn.
- `integrity_check.json`: biên bản kiểm chứng của bản tổng hợp, số tệp cũ thay đổi bằng 0.

Các tên CSV trên đều nằm trong thư mục nguồn tổng hợp đã nêu. Đã đối chiếu lại
điểm tổng hợp và từng cửa hàng bằng trung bình RMSSE trong metrics_per_series.csv;
đồng thời kiểm tra `sqrt(squared_error_sum / 28 / scale)` cho từng chuỗi hợp lệ.

## 5.6 – Kết quả thực nghiệm

Chèn hai hình sau (có bản PDF cùng tên):

1. **rmsse_comparison_overall.png**: so sánh trên toàn bộ 30.490 chuỗi.
   LightGBM–Tweedie = {indexed.loc[MODELS[0], METRIC]:.6f}; Seasonal Naive =
   {indexed.loc[MODELS[1], METRIC]:.6f}. Mức giảm sai số = {reduction:.2f}%, khoảng 23,29%,
   tính bằng `100 * (1 - điểm_LightGBM / điểm_Seasonal_Naive)`.
2. **rmsse_comparison_by_store.png**: mỗi cửa hàng có hai cột, mỗi cột là trung bình
   RMSSE 28 ngày của 3.049 chuỗi. Nhãn cột hiển thị ba chữ số thập phân.

Chỉ số dùng là **RMSSE_mean_unweighted**. Càng thấp càng tốt trên cùng tập đánh giá.
Điểm tổng hợp là trung bình các chuỗi hợp lệ, không phải căn của trung bình tỷ số
MSE/scale. MAE và RMSE vẫn là chỉ số bổ sung trong kết quả cũ, được giữ nguyên.
Đây không phải phần trăm sai số hoặc độ chính xác; giảm 23,29% không có nghĩa
độ chính xác tăng 23,29 điểm phần trăm.

**{NOTE}.**
Không đánh giá đủ 12 cấp phân cấp và trọng số doanh thu của WRMSSE M5.

## 5.7 – Trực quan hóa kết quả dự báo: chưa đủ dữ liệu horizon

**Không tạo rmsse_by_horizon.png hoặc rmsse_by_horizon.pdf**, vì thiếu nhãn thật
theo từng chuỗi và từng ngày trong CSV kết quả Validation Full đã lưu.
Đã kiểm tra header của 20 CSV dự báo nguồn bên dưới: chúng có y_pred nhưng
không có y_true/actual. Bảng metrics_per_series.csv có scale và tổng sai số
28 ngày; không thể phân rã duy nhất tổng này thành sai số từng horizon.
comparison.csv trong kết quả demo một sản phẩm không đủ cho toàn bộ 30.490 chuỗi.

Công thức cần dùng khi có đủ dữ liệu:

```text
V = các chuỗi có scale Train hữu hạn và > 0
mean_scaled_absolute_error_by_horizon(h)
    = mean_i_in_V(sqrt((y_true[i,h] - y_pred[i,h])^2 / scale[i]))
    = mean_i_in_V(abs(y_true[i,h] - y_pred[i,h]) / sqrt(scale[i]))
```

Đây là **RMSSE một bước theo horizon, trung bình trên các chuỗi**. Nó khác
RMSSE 28 ngày của một chuỗi: `sqrt(mean_h(error[i,h]^2) / scale[i])`.
Trung bình 28 giá trị trên đường horizon không đồng nhất với RMSSE_mean_unweighted
28 ngày. Không chia MAE cho hằng số, không lấy điểm tổng hợp làm đường phẳng.

Dữ liệu còn thiếu là CSV kết quả chứa `store_id, item_id, target_day/horizon,
y_true` cho đủ 30.490 × 28 điểm. Có thể nối CSV đó với y_pred và scale đã lưu;
không cần huấn luyện lại. Báo cáo này không đọc sales gốc để bổ sung nhãn,
đúng giới hạn chỉ đọc CSV kết quả của yêu cầu hiện tại.

Khi đủ nhãn, **rmsse_by_horizon.png** phù hợp mục **5.7**. Hiện tại có thể dùng
biểu đồ dự báo–thực tế đã có cho một sản phẩm để minh họa tại 5.7, kèm phạm vi
một chuỗi; không gọi đó là biểu đồ RMSSE horizon của 10 cửa hàng.

## Danh sách CSV dự báo đã kiểm tra schema

{source_list}

## Định dạng và bảo toàn

PNG 300 dpi, PDF dạng vector; tiêu đề, trục và ghi chú tiếng Việt.
Chèn PNG vào Google Docs; PDF phù hợp lưu trữ/in ấn. Điểm thấp hơn là tốt hơn.
`verification.json` ghi dấu SHA256 của nguồn, kiểm tra nguồn không thay đổi,
schema dự báo và lý do chưa tạo biểu đồ horizon. Mọi tệp cũ được giữ nguyên.
Script tái tạo: `scripts/plot_saved_rmsse_report.py`. Khi thư mục rmsse_report
đã tồn tại, lần chạy sau tạo thư mục con theo thời điểm UTC để tránh ghi đè.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    changed = [name for name, digest in before.items() if sha256(ROOT / name) != digest]
    evidence = {
        "source": str(source.relative_to(ROOT)), "input_sha256": before,
        "changed_inputs": changed, "series_crosscheck_passed": True,
        "forecast_schemas": forecast_schemas, "retrained": False,
        "final_test_run": False, "raw_data_read": False,
        "horizon_status": "not_generated_missing_per_day_actuals",
        "overall_reduction_percent": float(reduction),
    }
    (output / "verification.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if changed:
        raise RuntimeError("Nguồn đã thay đổi trong quá trình vẽ.")
    print(f"OUTPUT_DIR={output}")
    # Terminal Windows có thể dùng cp1252; nội dung tiếng Việt nằm trong README/hình.
    print("Created 2 PNG/PDF charts. Horizon chart unavailable: missing daily actuals.")


if __name__ == "__main__":
    main()
