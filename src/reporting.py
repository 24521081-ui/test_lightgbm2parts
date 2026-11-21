"""6. Đánh giá và xuất kết quả: chỉ số, biểu đồ và thông tin thực nghiệm."""

import json
import logging
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

# Lưu cache trong project; dùng backend xuất ảnh không cần cửa sổ đồ họa.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .forecasting import validate_forecasts

LOGGER = logging.getLogger("m5")
RMSSE_MEAN_COLUMN = "RMSSE_mean_unweighted"


# 6.1. Tính các chỉ số đánh giá trên dự báo đã hoàn thành.
def mae_rmse(actual, predicted):
    """MAE là sai lệch tuyệt đối trung bình; RMSE phạt sai số lớn mạnh hơn."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(
        predicted, dtype=float
    )
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Hai mảng phải cùng kích thước và không rỗng.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Không tính sai số trên NaN hoặc vô hạn.")
    error = actual - predicted
    return float(np.abs(error).mean()), float(np.sqrt(np.square(error).mean()))


def rmsse_scale(train):
    """Mẫu số RMSSE theo M5: sai phân Train từ lần bán dương đầu tiên.

    Chỉ bỏ tiền tố 0 khi tính mẫu số chỉ số; dữ liệu Train vẫn giữ nguyên.
    Chuỗi hằng, toàn 0 hoặc chưa đủ hai điểm có mẫu số không xác định.
    """
    train = np.asarray(train, dtype=float)
    if train.ndim != 1 or not np.isfinite(train).all() or (train < 0).any():
        raise ValueError(
            "Lịch sử tính scale phải là chuỗi một chiều, hữu hạn, không âm."
        )
    positive = np.flatnonzero(train > 0)
    if not len(positive) or len(train) - positive[0] < 2:
        return np.nan
    denominator = np.square(np.diff(train[positive[0] :])).mean()
    return float(denominator) if denominator > 0 else np.nan


def evaluate_forecast(forecast, actual_wide, train_history, items, training_time=0):
    """Nối theo mã và ngày sau dự báo; không dựa vào thứ tự dòng ngẫu nhiên."""
    if forecast.empty or any(
        forecast[key].nunique() != 1
        for key in ["model", "split", "store_id", "origin_day"]
    ):
        raise ValueError("Mỗi lần đánh giá cần đúng một model/split/cửa hàng/origin.")
    if len(set(items)) != len(items) or set(forecast.item_id) != set(items):
        raise ValueError("Danh sách sản phẩm dự báo không khớp tập đánh giá.")
    origin = int(forecast.origin_day.iloc[0][2:])
    horizon = int(forecast.horizon.max())
    validate_forecasts(forecast, horizon)
    train_history = np.asarray(train_history)
    if train_history.shape != (len(items), origin):
        raise ValueError("Lịch sử tính scale phải có đúng số sản phẩm và origin ngày.")
    if (
        "store_id" in actual_wide
        and not actual_wide.store_id.eq(forecast.store_id.iloc[0]).all()
    ):
        raise ValueError("Doanh số thật không thuộc đúng cửa hàng dự báo.")
    if actual_wide.item_id.duplicated().any() or set(actual_wide.item_id) != set(items):
        raise ValueError("Tập doanh số thật có mã trùng, thiếu hoặc thừa.")
    actual = actual_wide.set_index("item_id", verify_integrity=True).loc[
        list(items), [f"d_{d}" for d in range(origin + 1, origin + horizon + 1)]
    ]
    prediction = forecast.pivot(
        index="item_id", columns="horizon", values="y_pred"
    ).loc[list(items), list(range(1, horizon + 1))]
    truth, predicted = actual.to_numpy(dtype=float), prediction.to_numpy(dtype=float)
    mae, rmse = mae_rmse(truth, predicted)
    per_item_mse = np.square(truth - predicted).mean(axis=1)
    scales = np.array([rmsse_scale(row) for row in train_history])
    valid = np.isfinite(scales)
    scores = np.full(len(items), np.nan)
    scores[valid] = np.sqrt(per_item_mse[valid] / scales[valid])
    # Không thay mẫu số 0 bằng epsilon: cách đó tạo chỉ số khó diễn giải.
    # Trung bình chỉ trên chuỗi xác định, đồng thời báo số chuỗi bị loại.
    result = {
        "model": forecast.model.iloc[0],
        "split": forecast.split.iloc[0],
        "store_id": forecast.store_id.iloc[0],
        "n_items": len(items),
        "MAE": mae,
        "RMSE": rmse,
        RMSSE_MEAN_COLUMN: float(scores[valid].mean()) if valid.any() else np.nan,
        "training_time_seconds": training_time,
        "rmsse_valid_items": int(valid.sum()),
        "rmsse_undefined_items": int((~valid).sum()),
    }
    per_item = pd.DataFrame({"item_id": list(items), "RMSSE": scores, "scale": scales})
    return result, per_item, truth, predicted


# 6.2. Vẽ và lưu biểu đồ phục vụ báo cáo.
def save_rmsse_comparisons(by_store, overall, directory, stores, models, labels):
    """Biểu đồ chính từ chỉ số đã tính, không tính lại hoặc thay đổi công thức."""
    metric = "RMSSE_mean_unweighted"
    colors = ["#2563a6", "#e39836"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    indexed = overall.set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([labels[m] for m in models], indexed[metric], color=colors)
    ax.bar_label(bars, fmt="%.6f", padding=5)
    ax.set_ylim(0, indexed[metric].max() * 1.2)
    ax.set_title(
        "RMSSE trung bình trên toàn bộ 10 cửa hàng\nValidation d_1886–d_1913 • 30.490 chuỗi/mô hình"
    )
    figures = [(fig, ax, "rmsse_mean_overall")]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    positions = np.arange(len(stores))
    for index, model in enumerate(models):
        frame = by_store.loc[by_store.model == model].set_index("store_id").loc[stores]
        bars = ax.bar(
            positions + (index - 0.5) * 0.38,
            frame[metric],
            width=0.38,
            color=colors[index],
            label=labels[model],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(positions, stores)
    ax.set_ylim(0, by_store[metric].max() * 1.2)
    ax.set_title(
        "RMSSE trung bình theo cửa hàng\nValidation d_1886–d_1913 • 3.049 chuỗi/cửa hàng"
    )
    ax.legend(ncol=2)
    figures.append((fig, ax, "rmsse_mean_by_store"))
    for fig, ax, name in figures:
        ax.set_ylabel(metric + " (thấp hơn tốt hơn)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.text(
            0.5,
            0.015,
            "Trung bình RMSSE từng chuỗi hợp lệ; không phải WRMSSE chính thức M5.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        for extension in ["png", "pdf"]:
            fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


# 6.3. Xuất thông tin truy vết và kết quả của lần chạy.
def save_json(data, path):
    """Lưu cấu hình và thông tin lần chạy để có thể đối chiếu về sau."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )



def training_patterns(matrix, items):
    """Quy tắc cố định chỉ nhìn Train; không chọn ví dụ theo sai số holdout."""
    zero = (matrix == 0).mean(axis=1)
    groups = np.select([zero < .25, zero < .75], ['ít gián đoạn', 'gián đoạn'], default='rất gián đoạn')
    frame = pd.DataFrame({'item_id': list(items), 'zero_fraction_train': zero, 'sales_pattern': groups})
    # Mỗi nhóm lấy mã đầu theo thứ tự từ điển; hình thành phần lấy tỷ lệ 0 cao nhất.
    selected = frame.sort_values('item_id').groupby('sales_pattern', sort=True).head(1).item_id.tolist()
    component = frame.sort_values(['zero_fraction_train', 'item_id'], ascending=[False, True]).item_id.iloc[0]
    return frame, selected, component


def save_two_part_figures(directory, detail, per_item, patterns, selected, component, context):
    directory.mkdir(parents=True, exist_ok=True)
    scope = 'SMOKE (mẫu nhỏ)' if context['scope'] == 'smoke' else 'FULL'
    title = f"{scope} | {context['store_id']} | {detail.item_id.nunique()} sản phẩm | d_{context['origin_day']+1}–d_{context['origin_day']+28}"
    days = np.arange(1, 29)
    blue, orange = '#1874b4', '#e38724'
    def finish(fig, name):
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(directory / name, dpi=220)
        plt.close(fig)
    def lines(ax, frame):
        ax.bar(days, frame.y_actual, color='0.65', label='Thực tế', alpha=.7)
        ax.plot(days, frame.y_pred_two_part, 'o-', color=blue, markersize=3, label='Two-Part')
        ax.plot(days, frame.y_pred_seasonal_naive, '--', color=orange, label='Seasonal Naïve')
        ax.set_ylabel('Lượng bán')
        ax.set_xlim(.3, 28.7)
        ax.grid(axis='y', alpha=.2)
    def residuals(ax, frame):
        ax.plot(days, frame.signed_error_two_part, 'o-', color=blue, markersize=3, label='Two-Part')
        ax.plot(days, frame.signed_error_seasonal_naive, '--', color=orange, label='Seasonal Naïve')
        ax.axhline(0, color='black', linewidth=1)
        ax.set(xlim=(.3, 28.7), ylabel='Sai lệch',
               title='Sai lệch = dự báo − thực tế | Dương: dự báo thừa; âm: dự báo thiếu')
        ax.grid(axis='y', alpha=.2)
    fig, axes = plt.subplots(2*len(selected), 1, figsize=(12, 5*len(selected)), squeeze=False,
                             gridspec_kw={'height_ratios': [2, 1]*len(selected)})
    for index, item in enumerate(selected):
        ax, residual_ax = axes[2*index, 0], axes[2*index+1, 0]
        frame = detail.loc[detail.item_id == item].sort_values('horizon')
        lines(ax, frame)
        ax.set_title(item)
        ax.legend(ncol=3, fontsize=8)
        residuals(residual_ax, frame)
    axes[-1, 0].set_xlabel('Bước dự báo (ngày 1–28; cột bằng 0 là ngày không bán)')
    finish(fig, '01_actual_vs_forecast.png')

    aggregate = detail.groupby('horizon')[['y_actual', 'y_pred_two_part', 'y_pred_seasonal_naive']].sum().cumsum()
    fig, ax = plt.subplots(figsize=(11, 5))
    for col, label, color, style in [('y_actual', 'Thực tế', '0.4', '-'), ('y_pred_two_part', 'Two-Part', blue, '-'), ('y_pred_seasonal_naive', 'Seasonal Naïve', orange, '--')]:
        ax.plot(days, aggregate[col], style, color=color, label=label)
    delta = aggregate.iloc[-1] - aggregate.y_actual.iloc[-1]
    ax.set_title(f"Cộng dồn toàn bộ mẫu; chênh ngày 28: Two-Part {delta.y_pred_two_part:+.2f}, Naïve {delta.y_pred_seasonal_naive:+.2f} đơn vị")
    ax.set(xlabel='Horizon', ylabel='Lượng bán cộng dồn')
    ax.legend()
    finish(fig, '02_cumulative_units.png')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, label, color in [('y_pred_two_part', 'Two-Part', blue), ('y_pred_seasonal_naive', 'Seasonal Naïve', orange)]:
        errors = detail.assign(error=(detail.y_actual-detail[col]).abs()).groupby('horizon').error.mean()
        ax.plot(errors.index, errors, 'o-', color=color, label=label, markersize=3)
    ax.set(xlabel='Horizon', ylabel='MAE', title='Sai số trên cùng toàn bộ tập sản phẩm')
    ax.legend()
    finish(fig, '03_mae_by_horizon.png')

    gain = per_item['lightgbm_two_part'][['item_id', 'RMSSE']].merge(per_item['seasonal_naive'][['item_id', 'RMSSE']], on='item_id', suffixes=('_two_part', '_naive'), validate='one_to_one').merge(patterns, on='item_id', validate='one_to_one')
    gain['difference'] = gain.RMSSE_two_part - gain.RMSSE_naive
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    for index, group in enumerate(['ít gián đoạn', 'gián đoạn', 'rất gián đoạn']):
        values = gain.loc[gain.sales_pattern == group, 'difference'].dropna().to_numpy()
        n = len(values)
        labels.append(f'{group}\nn={n}; thắng={(values<0).mean():.1%}' if n else f'{group}\nn=0; thắng: không xác định')
        if n >= 5:
            ax.boxplot([values], positions=[index], widths=.5)
        if n:
            jitter = np.linspace(-.12, .12, n) if n > 1 else np.array([0.])
            ax.scatter(index+jitter, values, s=18, alpha=.7, color=blue)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_xticks(range(3), labels)
    ax.set(ylabel='RMSSE Two-Part − RMSSE Seasonal Naïve', title='Âm: Two-Part tốt hơn; loại RMSSE không xác định')
    finish(fig, '04_model_gain_by_sales_pattern.png')
    gain.to_csv(directory / 'sales_pattern_gain.csv', index=False)

    frame = detail.loc[detail.item_id == component].sort_values('horizon')
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    axes[0].plot(days, frame.probability_positive, 'o-', color=blue, label='P(có bán)')
    axes[0].scatter(days, (frame.y_actual > 0).astype(int), marker='x', color='0.3', label='Thực tế có bán (0/1)')
    axes[0].set(ylabel='Xác suất', ylim=(-.05, 1.05), title=component)
    axes[1].plot(days, frame.conditional_units, 'o-', color=blue, label='E(lượng bán | có bán)')
    positive = frame.y_actual.to_numpy() > 0
    axes[1].scatter(days[positive], frame.y_actual.to_numpy()[positive], color='0.3', label='Thực tế ngày bán dương')
    axes[1].set_ylabel('Lượng bán khi có bán')
    lines(axes[2], frame)
    residuals(axes[3], frame)
    axes[3].set_xlabel('Horizon')
    for ax in axes:
        ax.legend(fontsize=8)
    finish(fig, '05_two_part_components.png')


def evaluate_and_export(prediction, baseline, actual_wide, matrix, items, bundle, info, locations, context, components=None):
    """Ghép khóa 1:1; kiểm tra thành phần trước khi xuất báo cáo."""
    results, per_items = [], {}
    for forecast, seconds in [(prediction, info['training_time_seconds']), (baseline, 0)]:
        result, per_item, truth, predicted = evaluate_forecast(forecast, actual_wide, matrix, items, seconds)
        per_item['MAE'] = np.abs(truth-predicted).mean(axis=1)
        per_item['RMSE'] = np.sqrt(np.square(truth-predicted).mean(axis=1))
        per_item['store_id'] = context['store_id']
        per_item.to_csv(locations['metrics'] / f"{result['model']}_per_item.csv", index=False)
        per_items[result['model']] = per_item
        results.append(result)
    keys = ['split', 'origin_day', 'item_id', 'store_id', 'target_day', 'horizon']
    detail = prediction.drop(columns='model').rename(columns={'y_pred': 'y_pred_two_part'}).merge(
        baseline.drop(columns='model').rename(columns={'y_pred': 'y_pred_seasonal_naive'}), on=keys, how='outer', validate='one_to_one', indicator=True)
    if not detail['_merge'].eq('both').all():
        raise ValueError('Hai mô hình thiếu khóa dự báo đối ứng.')
    detail = detail.drop(columns='_merge')
    actual = actual_wide.melt(id_vars=['item_id', 'store_id'], value_vars=[f"d_{context['origin_day']+i}" for i in range(1, 29)], var_name='target_day', value_name='y_actual')
    join_keys = ['item_id', 'store_id', 'target_day']
    for other in [actual, components]:
        if other is None:
            raise ValueError('Thiếu bảng thành phần Two-Part.')
        detail = detail.merge(other, on=join_keys, how='outer', validate='one_to_one', indicator=True)
        if not detail['_merge'].eq('both').all():
            raise ValueError('Thiếu/thừa khóa thực tế hoặc thành phần.')
        detail = detail.drop(columns='_merge')
    numeric = detail.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise ValueError('Bảng chi tiết có giá trị không hợp lệ.')
    if not detail.probability_positive.between(0, 1).all() or not np.allclose(detail.y_pred_two_part, detail.probability_positive*detail.conditional_units, rtol=1e-12, atol=1e-12):
        raise ValueError('Dự báo không khớp tích hai thành phần.')
    detail = add_forecast_errors(detail)
    patterns, selected, component = training_patterns(matrix, items)
    if selected != context['illustration_items'] or component != context['component_item']:
        raise ValueError('Lựa chọn minh họa khác lựa chọn đã khóa trước khi đọc holdout.')
    detail = detail.merge(patterns, on='item_id', validate='many_to_one').sort_values(['item_id', 'horizon'])
    detail.to_csv(locations[context['split']] / 'actual_vs_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(results).to_csv(locations['metrics'] / 'summary.csv', index=False)
    if context.get('save_figures', True):
        save_two_part_figures(locations['figures'], detail, per_items, patterns, selected, component, context)
    metadata = {**info, **context, 'demo': False, 'selected_items': items,
        'python_version': platform.python_version(),
        'libraries': {name: version(name) for name in ['numpy', 'pandas', 'lightgbm', 'scikit-learn', 'matplotlib', 'PyYAML', 'joblib']},
        'rmsse_policy': 'M5: bỏ tiền tố 0 chỉ khi tính scale; không thay scale 0 bằng epsilon',
        'aggregate_rmsse_column': RMSSE_MEAN_COLUMN, 'official_wrmsse': False,
        'illustration_rule': 'Mỗi nhóm lấy mã đầu từ điển; hình 05 lấy tỷ lệ ngày 0 Train cao nhất, hòa thì mã đầu từ điển.',
        'pattern_rule': 'Tỷ lệ ngày 0 trên toàn Train: <25%, [25%,75%), >=75%.',
        'illustration_items': selected, 'component_item': component,
        'checks': {'horizons_per_item': 28, 'unique_keys': True, 'finite_nonnegative': True, 'actual_join_one_to_one': True, 'component_product': True},
        'artifacts': {key: str(value.resolve()) for key, value in locations.items()}}
    save_json(metadata, locations['models'] / 'metadata.json')
    LOGGER.info('Kết quả:\n%s', pd.DataFrame(results).to_string(index=False))
    LOGGER.info('Đã lưu bảng chỉ số: %s', locations['metrics'] / 'summary.csv')
    return pd.DataFrame(results)


ERROR_METHODS = [('two_part', 'lightgbm_two_part'), ('seasonal_naive', 'seasonal_naive')]


def add_forecast_errors(detail):
    """Sai lệch = dự báo - thực tế; giữ float64, không làm tròn dữ liệu."""
    detail = detail.copy()
    actual = detail['y_actual'].to_numpy(dtype=np.float64)
    for suffix, _ in ERROR_METHODS:
        error = detail[f'y_pred_{suffix}'].to_numpy(dtype=np.float64) - actual
        detail[f'signed_error_{suffix}'] = error
        detail[f'absolute_error_{suffix}'] = np.abs(error)
    return detail


def validation_error_tables(detail):
    """Tổng và sai lệch tính từ các dòng gốc; không lấy trị tuyệt đối của tổng để tính MAE."""
    totals, horizons = [], []
    for suffix, model in ERROR_METHODS:
        frame = detail[['store_id', 'item_id', 'horizon', 'y_actual', f'y_pred_{suffix}',
                        f'signed_error_{suffix}', f'absolute_error_{suffix}']]
        aggregates = dict(n_observations=('y_actual', 'size'), n_items=('item_id', 'nunique'),
                          total_actual=('y_actual', 'sum'), total_prediction=(f'y_pred_{suffix}', 'sum'),
                          signed_error_sum=(f'signed_error_{suffix}', 'sum'),
                          absolute_error_sum=(f'absolute_error_{suffix}', 'sum'))
        for keys, output in [(['store_id'], totals), (['store_id', 'horizon'], horizons)]:
            table = frame.groupby(keys, observed=True).agg(**aggregates).reset_index()
            table.insert(0, 'model', model)
            table['total_difference'] = table.total_prediction - table.total_actual
            table['mean_signed_error'] = table.signed_error_sum / table.n_observations
            table['MAE'] = table.absolute_error_sum / table.n_observations
            output.append(table)
    return pd.concat(totals, ignore_index=True), pd.concat(horizons, ignore_index=True)


def export_validation_workbook(sources, directory):
    """Gộp các cửa hàng đã hoàn tất; Excel chỉ định dạng hiển thị, không round giá trị."""
    from openpyxl.styles import Font, PatternFill
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    totals, horizons, scores = [], [], []
    for source in sources:
        detail = pd.read_csv(source['actual_vs_predictions'], float_precision='round_trip')
        checked = add_forecast_errors(detail)
        for suffix, _ in ERROR_METHODS:
            for prefix in ['signed_error', 'absolute_error']:
                col = f'{prefix}_{suffix}'
                if not np.array_equal(detail[col], checked[col]):
                    raise ValueError(f'Sai cột {col}: {source["store_id"]}')
        total, horizon = validation_error_tables(detail)
        totals.append(total)
        horizons.append(horizon)
        scores.append(pd.read_csv(source['summary'], float_precision='round_trip'))
    by_store = pd.concat(totals, ignore_index=True)
    store_horizon = pd.concat(horizons, ignore_index=True)
    metrics = pd.concat(scores, ignore_index=True)
    additive = ['n_observations', 'n_items', 'total_actual', 'total_prediction', 'signed_error_sum', 'absolute_error_sum']
    overall = by_store.groupby('model', observed=True)[additive].sum().reset_index()
    by_horizon = store_horizon.groupby(['model', 'horizon'], observed=True)[additive].sum().reset_index()
    for table in [overall, by_horizon]:
        table['total_difference'] = table.total_prediction - table.total_actual
        table['mean_signed_error'] = table.signed_error_sum / table.n_observations
        table['MAE'] = table.absolute_error_sum / table.n_observations
    pooled = []
    for model, frame in metrics.groupby('model', observed=True):
        weights = frame.n_items * 28
        valid = frame.rmsse_valid_items
        pooled.append(dict(model=model, n_items=int(frame.n_items.sum()),
            MAE=float(np.average(frame.MAE, weights=weights)),
            RMSE=float(np.sqrt(np.average(frame.RMSE**2, weights=weights))),
            RMSSE_mean_unweighted=float((frame[RMSSE_MEAN_COLUMN].fillna(0)*valid).sum()/valid.sum()) if valid.sum() else np.nan,
            rmsse_valid_items=int(valid.sum()), rmsse_undefined_items=int(frame.rmsse_undefined_items.sum())))
    guide = pd.DataFrame([
        ('Phạm vi', 'Validation C0 Full; d_1886–d_1913; dùng mô hình đã học đến d_1885, không fit lại; không chạy Test.'),
        ('Dấu sai lệch', 'signed_error = dự báo − thực tế. Dương: dự báo thừa; âm: dự báo thiếu.'),
        ('Sai lệch tuyệt đối', 'absolute_error = abs(signed_error). MAE = trung bình absolute_error trên các cặp sản phẩm–ngày.'),
        ('Chênh tổng', 'total_difference = total_prediction − total_actual; có thể triệt tiêu giữa ngày/sản phẩm.'),
        ('Độ chính xác', 'Tính từ float64 chưa làm tròn. CSV giữ độ chính xác gốc; Excel định dạng hiển thị 6 chữ số thập phân.'),
        ('Các sheet', 'Totals_all: cả 10 cửa hàng/28 ngày; Totals_by_store: từng cửa hàng/28 ngày; Errors_by_horizon: 1–28 trên toàn bộ mã; Errors_store_horizon: từng cửa hàng và horizon.'),
        ('Chỉ số', 'Metrics_all và Metrics_by_store: MAE, RMSE, RMSSE trung bình không trọng số; không phải WRMSSE chính thức M5.'),
        ('Nguồn', 'Model_sources ghi đường dẫn mô hình, CSV và hình cho từng cửa hàng. Chưa phân tích hoặc chọn cấu hình mới.'),
    ], columns=['Nội dung', 'Giải thích'])
    tables = {'Guide':guide, 'Totals_all':overall, 'Totals_by_store':by_store,
              'Errors_by_horizon':by_horizon, 'Errors_store_horizon':store_horizon,
              'Metrics_all':pd.DataFrame(pooled), 'Metrics_by_store':metrics,
              'Model_sources':pd.DataFrame(sources)}
    destination = directory / 'validation_summary.xlsx'
    with pd.ExcelWriter(destination, engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill('solid', fgColor='245A81')
            for column in sheet.columns:
                header = str(column[0].value)
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(18, len(header)+3))
                for cell in column[1:]:
                    if isinstance(cell.value, (float, np.floating)):
                        cell.number_format = '#,##0.000000;[Red]-#,##0.000000'
    for name, table in tables.items():
        if name != 'Guide':
            table.to_csv(directory / f'{name}.csv', index=False, encoding='utf-8-sig')
    LOGGER.info('Đã xuất workbook: %s', destination)
    return destination
