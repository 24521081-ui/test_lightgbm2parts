"""Vẽ hình tiếng Việt từ kết quả hoàn tất; không huấn luyện hoặc ghi results/."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STORES = {"CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"}
C5 = "validation_C5_full_10_stores_20261121T140034335643Z"
C6 = "validation_C6_platt_10_stores_20261121T145343079375Z"
KEYS = ["origin_day", "store_id", "item_id", "target_day", "horizon"]
PERIOD = "Validation d_1886–d_1913 · 10 cửa hàng · 30.490 chuỗi"
COLORS = {"C5": "#4477aa", "C6": "#228833", "C7": "#cc6677"}
SOURCES = set()


def read_json(path):
    SOURCES.add(path.relative_to(ROOT).as_posix())
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    SOURCES.add(path.relative_to(ROOT).as_posix())
    return pd.read_csv(path, float_precision="round_trip")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def completed(path, origin):
    m = read_json(path)
    require(m.get("status") in ("complete", "completed"), f"Chưa hoàn tất: {path}")
    require(set(m["stores"]) == STORES, f"Thiếu cửa hàng: {path}")
    require(m.get("origin", m.get("validation_origin")) == origin, f"Sai gốc: {path}")
    return m


def save(fig, output, name):
    fig.savefig(output / name, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(table, title, output, name):
    require(not table.empty and np.isfinite(table.to_numpy()).all(), "Heatmap thiếu RMSSE")
    fig, ax = plt.subplots(figsize=(9, max(4, len(table) * .45)))
    picture = ax.imshow(table.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for row in range(len(table)):
        for col in range(len(table.columns)):
            value = table.iloc[row, col]
            color = "white" if picture.norm(value) > .55 else "black"
            ax.text(col, row, f"{value:.4f}", ha="center", va="center", color=color)
    ax.set_title(title + "\n" + PERIOD, pad=12)
    fig.colorbar(picture, ax=ax, label="RMSSE trung bình không trọng số")
    save(fig, output, name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    results = args.results.resolve()
    output = ROOT / "reports" / "bieu_do_quan_trong"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    analysis = results / "analysis"
    validation = analysis / "stage_C7_validation_d1885_lambda075"
    test = analysis / "final_test_C6_C7_frozen" / "comparison_C6_C7"
    vm = completed(validation / "manifest.json", 1885)
    require(vm["lambda"] == .75 and vm["horizon"] == [1886, 1913], "Sai C7 Validation")
    cm = completed(results / "metrics" / C6 / "manifest.json", 1885)
    require(cm["validation_days"] == [1886, 1913], "Sai thời kỳ C6")
    require(vm["C6_batch"] == C6 and vm["C5_batch"] == C5 and cm["c5_batch"] == C5,
            "Nguồn C5/C6/C7 không khớp")
    c5m = read_json(results / "metrics" / C5 / "manifest.json")
    require(c5m["status"] == "completed" and c5m["origin_day"] == 1885
            and c5m["target_days"] == [1886, 1913] and c5m["n_items"] == 30490,
            "C5 chưa hoàn tất đúng phạm vi")
    vm_test = completed(test / "manifest_C6_C7.json", 1913)
    require(vm_test["test_days"] == [1914, 1941] and vm_test["lambda_C7"] == .75,
            "Sai C7 Test")
    val = read_csv(validation / "01_by_origin.csv").iloc[0]
    tes = read_csv(test / "01_by_origin.csv").iloc[0]
    stores = read_csv(validation / "02_by_store.csv")
    require(len(stores) == 10 and set(stores.store_id) == STORES
            and (stores.n_items == 3049).all() and (stores.origin == 1885).all(),
            "Sai phạm vi bảng cửa hàng")
    stage_a = analysis / C6 / "stage_A"

    # Nhóm đã có trong báo cáo gốc; không đặt lại ngưỡng theo holdout.
    groups = read_csv(stage_a / "09_by_sales_pattern.csv")
    require(groups.n_series.sum() == 30490, "Nhóm lượng bán không đủ 30.490 chuỗi")
    labels = {"gian_doan": "Gián đoạn", "it_gian_doan": "Ít gián đoạn",
              "rat_gian_doan": "Rất gián đoạn"}
    grouped = groups.set_index("sales_pattern")[["RMSSE_C5", "RMSSE_C6"]]
    grouped.index = grouped.index.map(labels)
    grouped.columns = ["C5", "C6"]
    heatmap(grouped, "RMSSE theo nhóm lượng bán lịch sử", output,
            "02_RMSSE_theo_nhom_luong_ban.png")
    table = stores.set_index("store_id")[["RMSSE_C5", "RMSSE_C6", "RMSSE_C7"]].sort_index()
    table.columns = ["C5", "C6", "C7 (λ = 0,75)"]
    heatmap(table, "RMSSE theo 10 cửa hàng", output, "03_RMSSE_theo_10_cua_hang.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (data, models, split) in enumerate([(val, ["C5", "C6", "C7"], "Validation d_1886–d_1913"),
                                               (tes, ["C6", "C7"], "Test d_1914–d_1941")]):
        for col, (metric, label) in enumerate([("RMSSE", "RMSSE trung bình"),
                                               ("MAE", "MAE (đơn vị bán)"),
                                               ("bias", "Sai lệch tổng (%)")]):
            values = [data[f"{metric}_{m}_pct" if metric == "bias" else f"{metric}_{m}"] for m in models]
            require(np.isfinite(values).all(), "Chỉ số không hữu hạn")
            axes[row, col].bar(models, values, color=[COLORS[m] for m in models])
            axes[row, col].set_title(split)
            axes[row, col].set_ylabel(label)
            if metric == "bias":
                axes[row, col].axhline(0, color="black", lw=.8)
    fig.suptitle("So sánh C5, C6 và C7 (λ = 0,75) · 10 cửa hàng · 30.490 chuỗi")
    fig.tight_layout()
    save(fig, output, "04_so_sanh_C5_C6_C7.png")

    reliability = read_csv(stage_a / "06_reliability_overall.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Hiệu chuẩn lý tưởng")
    for kind, label in [("raw_on_C6_path", "Xác suất gốc trên đường C6"),
                         ("cal_on_C6_path", "Platt trên đường C6")]:
        group = reliability.loc[reliability.probability_type == kind].sort_values("bin")
        require(group.n.sum() == 853720 and not group.bin.duplicated().any(), "Thiếu mẫu hiệu chuẩn")
        require(group[["mean_probability", "actual_positive_rate"]].ge(0).all().all()
                and group[["mean_probability", "actual_positive_rate"]].le(1).all().all(), "Xác suất sai")
        ax.plot(group.mean_probability, group.actual_positive_rate, "o-", label=label)
    ax.set(xlabel="Xác suất dự báo trung bình", ylabel="Tỷ lệ có bán thực tế",
           title="Độ tin cậy xác suất ngoài mẫu\n" + PERIOD)
    ax.legend(fontsize=9)
    save(fig, output, "05_hieu_chuan_xac_suat.png")

    daily = []
    for store in sorted(STORES):
        folder = validation / "folds" / store
        done = read_json(folder / "completed.json")
        require(done["origin"] == 1885 and done["store_id"] == store
                and done["version"] == vm["version"], "C7 fold không hợp lệ")
        detail = read_csv(results / "validation" / f"{C6}_{store}" / "actual_vs_predictions.csv")
        forecast = read_csv(folder / "forecast_C7.csv")
        require((detail.split == "validation").all() and (forecast.split == "validation").all(),
                "Lẫn split")
        paired = detail.merge(forecast[KEYS + ["y_pred"]], on=KEYS, how="outer",
                              validate="one_to_one", indicator=True)
        require(len(paired) == 3049 * 28 and paired._merge.eq("both").all(), "Không cùng khóa dự báo")
        require((paired.store_id == store).all() and (paired.origin_day == "d_1885").all()
                and set(paired.target_day) == {f"d_{d}" for d in range(1886, 1914)}
                and paired.item_id.nunique() == 3049, "Sai phạm vi dự báo")
        cols = ["y_actual", "y_pred_two_part", "y_pred"]
        require(np.isfinite(paired[cols]).all().all() and paired[cols].ge(0).all().all(), "Dữ liệu lượng bán sai")
        daily.append(paired.groupby("horizon")[cols].sum())
    daily = pd.concat(daily).groupby(level=0).sum()
    require(np.isclose(daily.y_actual.sum(), val.actual, rtol=0, atol=1e-9), "Tổng thực tế không khớp")
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, label in [("y_actual", "Thực tế"), ("y_pred_two_part", "C6"), ("y_pred", "C7 (λ = 0,75)")]:
        ax.plot(daily.index, daily[name], label=label, linewidth=2)
    ax.set(xlabel="Ngày dự báo (1–28)", ylabel="Tổng lượng bán (đơn vị)",
           title="Thực tế và dự báo toàn hệ thống trong 28 ngày\n" + PERIOD)
    ax.legend()
    save(fig, output, "06_thuc_te_va_du_bao_28_ngay.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, models, label in [(axes[0], val, ["C5", "C6", "C7"], "Validation d_1886–d_1913"),
                                    (axes[1], tes, ["C6", "C7"], "Test d_1914–d_1941")]:
        values = [data[f"bias_{m}_pct"] for m in models]
        ax.bar(models, values, color=["#cc6677" if v > 0 else "#4477aa" for v in values])
        ax.axhline(0, color="black", linewidth=1)
        ax.set(title=label, ylabel="Sai lệch tổng lượng bán (%)")
    fig.suptitle("Dự báo thừa (+) và thiếu (−) · 10 cửa hàng · C7 λ = 0,75")
    fig.tight_layout()
    save(fig, output, "07_sai_lech_tong_luong_ban.png")
    manifest = {"period": PERIOD, "sources": sorted(SOURCES),
                "figures": sorted(p.name for p in output.glob("*.png")),
                "missing": {"01_so_sanh_RMSSE_6_mo_hinh.png":
                            "Chưa có kết quả ES_bu, Croston–SBA, Random Forest, LightGBM–Tweedie cùng tập đánh giá."},
                "limitations": ["Hình 02 chỉ có C5/C6 theo nhóm lịch sử đã lưu.",
                                 "Chưa lưu xác suất từng ngày C7; hình 05 chỉ so sánh raw/Platt trên cùng đường C6.",
                                 "Hình 06 dùng C6/C7; không ghép thêm kết quả khác thời kỳ."]}
    (output / "nguon_du_lieu.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã tạo {len(manifest['figures'])} hình: {output}")
    for name, reason in manifest["missing"].items():
        print(f"THIẾU {name}: {reason}")


if __name__ == "__main__":
    main()
