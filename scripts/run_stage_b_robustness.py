
"""Stage B | C5 vs C6 LightGBM Two-Part, temporal-origin robustness.

Retrospective folds (28-day calibration, 28-day validation):
  auxiliary d_1..1801; calibrate 1802..1829; main train d_1..1829; test 1830..1857
  auxiliary d_1..1829; calibrate 1830..1857; main train d_1..1857; test 1858..1885
  existing C5/C6 at origin 1885: calibrate 1858..1885; validate 1886..1913

No training on any target's future labels. All price/rolling features are causal.
Per-store calibrator: LogisticRegression(C=1.0) on logit(raw p), as in C6.
Reuse main model at d_1829 as auxiliary at d_1829; reuse existing
C6 auxiliary (trained through d_1857) as main for origin d_1857 if available.

Runs are resumable; outputs are isolated from C0-C6.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LogisticRegression

from src.data_loader import check_files, load_calendar, load_prices, load_store
from src.preprocessing import history_matrix
from src.feature_engineering import next_day_features
from src.forecasting import FORECAST_COLUMNS, validate_forecasts
from src.reporting import evaluate_forecast
from src.training import train_model, verify_saved_models
from run_c5_price_features import (
    EXTRAS, KEYS, build_weekly_features, make_training_data,
    forecast_c5,
)
from run_c6_platt import (
    logit_probability, apply_platt, prepare_c5_features,
)

LOG = logging.getLogger('stage_b')
C5_TAG = 'validation_C5_full_10_stores_20261121T140034335643Z'
FOLDS = (1829, 1857)
H = 28
VERSION = 'B1_20261121'
STORE_CODES = (
    'CA_1', 'CA_2', 'CA_3', 'CA_4',
    'TX_1', 'TX_2', 'TX_3',
    'WI_1', 'WI_2', 'WI_3'
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    p = Path(path)
    require(p.is_file(), f'Missing JSON file: {p}')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def find_c6_batch(results, requested=None):
    if requested:
        tag = requested
        manifest = read_json(
            results / 'metrics' / tag / 'manifest.json'
        )
        require(
            manifest.get('status') == 'completed'
            and manifest.get('c5_batch') == C5_TAG,
            'Selected C6 batch does not match the final C5 validation'
        )
        return tag, manifest

    eligible = []

    for path in (
        results / 'metrics'
    ).glob('validation_C6_platt_10_stores_*/manifest.json'):
        m = read_json(path)
        if (
            m.get('status') == 'completed'
            and m.get('configuration') == 'C6'
            and m.get('c5_batch') == C5_TAG
            and len(m.get('stores', [])) == 10
        ):
            eligible.append((path.parent.name, m))

    require(
        len(eligible) == 1,
        f'Expected exactly one matching full C6 batch, '
        f'found {len(eligible)}; specify --c6-batch'
    )
    return eligible[0]


# ============================================================
# 1. FEATURES GIA TAI TUNG GOC DU BAO
# ============================================================

def historical_price_arrays(week_features, calendar, items, origin):
    """Origin price from completed weeks, without melting the full daily train.

    C5's historical daily sell_price is forward-filled from past days,
    while its extra two price features are left as-is.

    Reproduce these semantics using the already-lagged weekly table.
    """
    row = calendar.loc[
        calendar['d'].eq(f'd_{origin}'),
        'wm_yr_wk'
    ]

    require(
        len(row) == 1,
        f'No calendar week for origin d_{origin}'
    )

    current_week = int(row.iloc[0])

    historical = week_features.loc[
        week_features.wm_yr_wk.le(current_week)
    ]

    require(
        not historical.duplicated(KEYS).any(),
        'Duplicated historical weekly prices'
    )

    latest = (
        historical.sort_values('wm_yr_wk')
        .drop_duplicates('item_id', keep='last')
        .set_index('item_id')
    )

    current = (
        historical.loc[
            historical.wm_yr_wk.eq(current_week)
        ]
        .set_index('item_id')
    )

    require(
        not latest.index.duplicated().any()
        and not current.index.duplicated().any(),
        'Duplicate historical price item'
    )

    result = {
        'sell_price': (
            latest['sell_price']
            .reindex(items)
            .to_numpy(dtype=np.float32)
        )
    }

    for feature in EXTRAS:
        result[feature] = (
            current[feature]
            .reindex(items)
            .to_numpy(dtype=np.float32)
        )

    for feature, values in result.items():
        require(
            not np.isinf(values).any(),
            f'Infinite origin price feature: {feature}'
        )

    missing = int(np.isnan(result['sell_price']).sum())

    if missing:
        LOG.warning(
            'Origin d_%d: %d/%d products have no past known '
            'prices; keep NaN',
            origin, missing, len(items)
        )

    return result


# ============================================================
# 2. KIEM TRA THU TU DU BAO
# ============================================================

def check_prediction_order(comps, origin, items):
    n = len(items)

    require(
        len(comps) == n * H,
        'Incorrect 28-day components row count'
    )

    expected_days = np.repeat(
        np.arange(origin + 1, origin + H + 1),
        n
    )

    got_days = (
        comps['target_day']
        .str[2:]
        .astype(int)
        .to_numpy()
    )

    require(
        np.array_equal(expected_days, got_days),
        'Component rows are not horizon-major'
    )

    require(
        np.array_equal(
            comps['item_id'].to_numpy().reshape(H, n)[0],
            np.array(items)
        ),
        'Component item order mismatch'
    )


# ============================================================
# 3. HOC PLATT SCALING
# ============================================================

def fit_platt(comps, actual_cal, aux_origin, origin, items):

    require(
        aux_origin + H == origin,
        'Calibration period is not exactly one 28-day block'
    )

    check_prediction_order(comps, aux_origin, items)

    y = (
        np.asarray(actual_cal, dtype=float)
        .T.reshape(-1)
    )

    z = (y > 0).astype(np.int8)

    raw_p = (
        comps['probability_positive']
        .to_numpy(dtype=float)
    )

    require(
        z.min() == 0 and z.max() == 1,
        'Calibration data has just one class'
    )

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0
    )

    model.fit(
        logit_probability(raw_p),
        z
    )

    slope = float(model.coef_[0, 0])

    require(
        slope > 0,
        'Platt calibration has nonpositive slope; '
        'manual review required'
    )

    p_cal = apply_platt(model, raw_p)

    metadata = {
        'aux_origin': aux_origin,
        'calibration_start': aux_origin + 1,
        'calibration_end': origin,
        'n_calibration_rows': int(len(z)),
        'cal_positive_rate': float(z.mean()),
        'cal_mean_p_raw': float(raw_p.mean()),
        'cal_mean_p_platt': float(p_cal.mean()),
        'cal_brier_raw': float(
            np.mean((raw_p - z) ** 2)
        ),
        'cal_brier_platt_in_sample': float(
            np.mean((p_cal - z) ** 2)
        ),
        'platt_slope': slope,
        'platt_intercept': float(model.intercept_[0]),
    }

    return model, metadata


# ============================================================
# 4. DU BAO C6 TAI GOC BAT KY
# ============================================================

def forecast_c6_at_origin(
    bundle,
    calibrator,
    matrix,
    items,
    store,
    state,
    calendar,
    origin,
    frozen_prices
):
    """Same C6 recursion as original, but origin is configurable."""

    cal = calendar.set_index(
        'd',
        verify_integrity=True
    )

    features = bundle['features']

    for feature in ('sell_price', *EXTRAS):
        require(
            feature in features,
            f'C5 missing feature {feature}'
        )

    history = np.asarray(
        matrix[:, -H:],
        dtype=np.float32
    ).copy()

    require(
        history.shape == (len(items), H),
        'Wrong recursive history'
    )

    preds = []
    comps = []

    for horizon in range(1, H + 1):

        day = f'd_{origin + horizon}'

        x = next_day_features(
            history,
            items,
            cal.loc[day],
            state,
            bundle['categories'],
            last_prices=frozen_prices['sell_price']
        )

        for feature in EXTRAS:
            x[feature] = frozen_prices[feature]

        require(
            x[features].columns.to_list() == features,
            'Feature order mismatch'
        )

        p_raw, mu = (
            bundle['model']
            .predict_components(x[features])
        )

        p_cal = apply_platt(
            calibrator,
            p_raw
        )

        yhat = np.maximum(
            p_cal * mu,
            0.0
        )

        require(
            np.isfinite(yhat).all(),
            'C6 prediction contains NaN/Inf'
        )

        preds.append(
            pd.DataFrame({
                'model': 'lightgbm_two_part',
                'split': 'validation',
                'origin_day': f'd_{origin}',
                'item_id': items,
                'store_id': store,
                'target_day': day,
                'horizon': horizon,
                'y_pred': yhat,
            })
        )

        comps.append(
            pd.DataFrame({
                'item_id': items,
                'store_id': store,
                'target_day': day,
                'probability_positive': p_cal,
                'probability_raw': p_raw,
                'conditional_units': mu,
            })
        )

        history = np.column_stack((
            history[:, 1:],
            yhat
        ))

    result = pd.concat(
        preds,
        ignore_index=True
    )[FORECAST_COLUMNS]

    validate_forecasts(result)

    return (
        result,
        pd.concat(comps, ignore_index=True)
    )


# ============================================================
# 5. TAI SU DUNG HOAC HUAN LUYEN MO HINH
# ============================================================

def get_model(
    store,
    fit_origin,
    wide,
    calendar,
    weekly,
    items,
    params,
    c6_tag,
    results,
    destination
):
    """Cache models by store and fit origin."""

    model_path = (
        destination
        / 'models'
        / store
        / f'd_{fit_origin}'
        / 'model.joblib'
    )

    if model_path.exists():

        bundle = verify_saved_models(model_path)

        require(
            bundle['origin_day'] == fit_origin
            and bundle['categories'] == items,
            'Cached model has incorrect train origin '
            'or item ordering'
        )

        return bundle, 'cached_stage_b'

    if fit_origin == 1857 and len(items) == 3049:

        original_path = (
            results
            / 'models'
            / f'{c6_tag}_{store}'
            / 'aux_model.joblib'
        )

        if original_path.exists():

            bundle = verify_saved_models(
                original_path
            )

            require(
                bundle['origin_day'] == fit_origin
                and bundle['categories'] == items,
                'Original C6 auxiliary incompatible '
                'with Stage B'
            )

            require(
                all(
                    x in bundle['features']
                    for x in ['sell_price', *EXTRAS]
                ),
                'Original C6 auxiliary is not a C5 '
                'feature model'
            )

            LOG.info(
                'REUSE original C6 historical model '
                'for %s at d_1857',
                store
            )

            return (
                bundle,
                'reused_existing_c6_aux'
            )

    hist = prepare_c5_features(
        wide,
        calendar,
        weekly,
        fit_origin
    )

    training = make_training_data(hist)

    require(
        training['fit_end_day'] == fit_origin,
        'Train rows include future labels'
    )

    require(
        training['categories'] == items,
        'Model item categories are not frozen'
    )

    del hist
    gc.collect()

    bundle, info = train_model(
        training,
        params,
        model_path
    )

    del training
    gc.collect()

    require(
        bundle['origin_day'] == fit_origin,
        'Saved model origin mismatch'
    )

    LOG.info(
        'TRAIN DONE %s d_%s: %.1f s',
        store,
        fit_origin,
        info['training_time_seconds']
    )

    return bundle, 'trained_stage_b'


# ============================================================
# 6. TONG HOP KET QUA THEO GOC
# ============================================================

def aggregate_origin_records(df):
    """Mean RMSSE across all valid item-store series."""

    valid = df[
        'valid_rmsse_items'
    ].to_numpy(dtype=float)

    actual = df.actual.sum()
    predicted5 = df.predicted_C5.sum()
    predicted6 = df.predicted_C6.sum()

    return {
        'origin': int(df.origin.iloc[0]),
        'n_stores': int(len(df)),
        'n_items': int(df.n_items.sum()),
        'valid_rmsse_items': int(
            df.valid_rmsse_items.sum()
        ),
        'actual': float(actual),
        'predicted_C5': float(predicted5),
        'predicted_C6': float(predicted6),
        'bias_C5_pct': float(
            100 * (predicted5 / actual - 1)
        ),
        'bias_C6_pct': float(
            100 * (predicted6 / actual - 1)
        ),
        'MAE_C5': float(
            np.average(
                df.MAE_C5,
                weights=df.n_items
            )
        ),
        'MAE_C6': float(
            np.average(
                df.MAE_C6,
                weights=df.n_items
            )
        ),
        'RMSSE_C5': float(
            np.average(
                df.RMSSE_C5,
                weights=valid
            )
        ),
        'RMSSE_C6': float(
            np.average(
                df.RMSSE_C6,
                weights=valid
            )
        ),
        'positive_rate': float(
            np.average(
                df.positive_rate,
                weights=df.n_items
            )
        ),
        'Brier_C5_path': float(
            np.average(
                df.Brier_C5_path,
                weights=df.n_items
            )
        ),
        'Brier_raw_C6_path': float(
            np.average(
                df.Brier_raw_C6_path,
                weights=df.n_items
            )
        ),
        'Brier_cal_C6_path': float(
            np.average(
                df.Brier_cal_C6_path,
                weights=df.n_items
            )
        ),
    }


# ============================================================
# 7. DANH GIA TUNG CUA HANG
# ============================================================

def record_from_prediction(
    store,
    origin,
    items,
    train_matrix,
    actual_wide,
    prediction5,
    components5,
    prediction6,
    components6,
    cal_info,
    model5_source,
    aux_source
):
    score5, item5, truth5, pred5 = evaluate_forecast(
        prediction5,
        actual_wide,
        train_matrix,
        items
    )

    score6, item6, truth6, pred6 = evaluate_forecast(
        prediction6,
        actual_wide,
        train_matrix,
        items
    )

    np.testing.assert_array_equal(
        truth5,
        truth6
    )

    np.testing.assert_allclose(
        item5['scale'],
        item6['scale'],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True
    )

    n = len(items)

    check_prediction_order(
        components5,
        origin,
        items
    )

    check_prediction_order(
        components6,
        origin,
        items
    )

    actual = truth5

    z = (
        (actual > 0)
        .astype(float)
        .T.reshape(-1)
    )

    p5 = (
        components5['probability_positive']
        .to_numpy(dtype=float)
    )

    praw = (
        components6['probability_raw']
        .to_numpy(dtype=float)
    )

    pcal = (
        components6['probability_positive']
        .to_numpy(dtype=float)
    )

    def brier(p):
        return float(
            np.mean((p - z) ** 2)
        )

    record = {
        'origin': origin,
        'store_id': store,
        'n_items': n,
        'valid_rmsse_items': int(
            score5['rmsse_valid_items']
        ),
        'actual': float(actual.sum()),
        'predicted_C5': float(pred5.sum()),
        'predicted_C6': float(pred6.sum()),
        'bias_C5_pct': float(
            100 * (
                pred5.sum() / actual.sum() - 1
            )
        ),
        'bias_C6_pct': float(
            100 * (
                pred6.sum() / actual.sum() - 1
            )
        ),
        'RMSSE_C5': float(
            score5['RMSSE_mean_unweighted']
        ),
        'RMSSE_C6': float(
            score6['RMSSE_mean_unweighted']
        ),
        'MAE_C5': float(score5['MAE']),
        'MAE_C6': float(score6['MAE']),
        'positive_rate': float(z.mean()),
        'mean_p_C5': float(p5.mean()),
        'mean_p_raw_C6_path': float(praw.mean()),
        'mean_p_cal_C6_path': float(pcal.mean()),
        'Brier_C5_path': brier(p5),
        'Brier_raw_C6_path': brier(praw),
        'Brier_cal_C6_path': brier(pcal),
        'model_C5_source': model5_source,
        'aux_model_source': aux_source,
        **cal_info,
    }

    scale = item5['scale'].to_numpy()

    rows = []

    for start in range(0, H, 7):

        a = actual[:, start:start + 7]
        f5 = pred5[:, start:start + 7]
        f6 = pred6[:, start:start + 7]

        slice_z = (
            (a > 0)
            .astype(float)
            .T.reshape(-1)
        )

        seg = slice(
            start * n,
            (start + 7) * n
        )

        pr = praw[seg]
        pc = pcal[seg]

        valid = np.isfinite(scale)

        m5 = np.square(a - f5).mean(axis=1)
        m6 = np.square(a - f6).mean(axis=1)

        rows.append({
            'origin': origin,
            'store_id': store,
            'week': start // 7 + 1,
            'actual': float(a.sum()),
            'predicted_C5': float(f5.sum()),
            'predicted_C6': float(f6.sum()),
            'bias_C5_pct': float(
                100 * (
                    f5.sum() / a.sum() - 1
                )
            ),
            'bias_C6_pct': float(
                100 * (
                    f6.sum() / a.sum() - 1
                )
            ),
            'MAE_C5': float(
                np.abs(a - f5).mean()
            ),
            'MAE_C6': float(
                np.abs(a - f6).mean()
            ),
            'RMSSE_C5': (
                float(
                    np.sqrt(
                        m5[valid] / scale[valid]
                    ).mean()
                )
                if valid.any()
                else np.nan
            ),
            'RMSSE_C6': (
                float(
                    np.sqrt(
                        m6[valid] / scale[valid]
                    ).mean()
                )
                if valid.any()
                else np.nan
            ),
            'Brier_raw_C6_path': float(
                np.mean(
                    (pr - slice_z) ** 2
                )
            ),
            'Brier_cal_C6_path': float(
                np.mean(
                    (pc - slice_z) ** 2
                )
            ),
        })

    per_item = (
        item5[['item_id', 'RMSSE', 'scale']]
        .rename(columns={'RMSSE': 'RMSSE_C5'})
    )

    per_item['RMSSE_C6'] = (
        item6['RMSSE'].to_numpy()
    )

    per_item['origin'] = origin
    per_item['store_id'] = store

    per_item['actual_units'] = (
        actual.sum(axis=1)
    )

    per_item['predicted_C5'] = (
        pred5.sum(axis=1)
    )

    per_item['predicted_C6'] = (
        pred6.sum(axis=1)
    )

    per_item['C6_improves'] = (
        per_item.RMSSE_C6
        < per_item.RMSSE_C5
    )

    return record, rows, per_item


# ============================================================
# 8. CHAY MOT GOC DU BAO
# ============================================================

def run_fold(
    store,
    origin,
    params,
    calendar,
    results,
    data_dir,
    c6_tag,
    destination,
    chunk_size,
    smoke_items=None
):
    """Train auxiliary/main separately and evaluate without leakage."""

    folder = (
        destination
        / 'folds'
        / f'd_{origin}'
        / store
    )

    done = folder / 'completed.json'

    if done.exists():

        r = read_json(done)

        require(
            r.get('version') == VERSION
            and r.get('origin') == origin
            and r.get('store_id') == store,
            'Cached fold not compatible'
        )

        for required in (
            'per_item.csv',
            'weekly.csv'
        ):
            require(
                (folder / required).exists(),
                f'Missing cached output: {required}'
            )

        LOG.info(
            'RESUME %s d_%d from completed results',
            store,
            origin
        )

        return (
            r['record'],
            pd.read_csv(folder / 'weekly.csv'),
            pd.read_csv(folder / 'per_item.csv')
        )

    aux_origin = origin - H

    # Only historical data through current origin.
    wide = load_store(
        data_dir,
        store,
        origin,
        chunk_size=chunk_size
    )

    if smoke_items is not None:
        wide = wide.head(smoke_items).copy()

    items = wide.item_id.to_list()

    require(
        wide.item_id.nunique() == len(items),
        'Duplicate item IDs'
    )

    require(
        smoke_items is not None
        or len(items) == 3049,
        f'Expected 3049 items, found {len(items)} at {store}'
    )

    state = str(wide.state_id.iloc[0])

    matrix = history_matrix(
        wide,
        origin
    )

    prices = load_prices(
        data_dir,
        store
    )

    weekly = build_weekly_features(
        prices,
        calendar
    )

    del prices
    gc.collect()

    # Train auxiliary through origin - 28.
    aux_bundle, aux_source = get_model(
        store,
        aux_origin,
        wide,
        calendar,
        weekly,
        items,
        params,
        c6_tag,
        results,
        destination
    )

    aux_price = historical_price_arrays(
        weekly,
        calendar,
        items,
        aux_origin
    )

    _, cal_comps = forecast_c5(
        aux_bundle,
        matrix[:, :aux_origin],
        items,
        store,
        state,
        calendar,
        aux_origin,
        aux_price
    )

    cal_actual = wide[
        [
            f'd_{d}'
            for d in range(
                aux_origin + 1,
                origin + 1
            )
        ]
    ].to_numpy()

    calibrator, cal_info = fit_platt(
        cal_comps,
        cal_actual,
        aux_origin,
        origin,
        items
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        calibrator,
        folder / 'platt.joblib'
    )

    del aux_bundle, cal_comps
    gc.collect()

    # Train main C5 architecture through current origin.
    main_bundle, main_source = get_model(
        store,
        origin,
        wide,
        calendar,
        weekly,
        items,
        params,
        c6_tag,
        results,
        destination
    )

    main_price = historical_price_arrays(
        weekly,
        calendar,
        items,
        origin
    )

    del weekly
    gc.collect()

    pred5, comp5 = forecast_c5(
        main_bundle,
        matrix,
        items,
        store,
        state,
        calendar,
        origin,
        main_price
    )

    pred6, comp6 = forecast_c6_at_origin(
        main_bundle,
        calibrator,
        matrix,
        items,
        store,
        state,
        calendar,
        origin,
        main_price
    )

    # Read holdout labels only after forecasts are created.
    actual_wide = load_store(
        data_dir,
        store,
        origin + H,
        chunk_size=chunk_size,
        start_day=origin + 1,
        item_ids=items
    )

    require(
        actual_wide.item_id.to_list() == items,
        'Holdout item ordering mismatch'
    )

    record, weeks, per_item = record_from_prediction(
        store,
        origin,
        items,
        matrix,
        actual_wide,
        pred5,
        comp5,
        pred6,
        comp6,
        cal_info,
        main_source,
        aux_source
    )

    per_item.to_csv(
        folder / 'per_item.csv',
        index=False,
        encoding='utf-8-sig'
    )

    pd.DataFrame(weeks).to_csv(
        folder / 'weekly.csv',
        index=False,
        encoding='utf-8-sig'
    )

    write_json(
        done,
        {
            'version': VERSION,
            'origin': origin,
            'store_id': store,
            'created_utc': datetime.now(
                timezone.utc
            ).isoformat(),
            'record': record,
        }
    )

    LOG.info(
        '%s d_%d: RMSSE %.6f -> %.6f | '
        'bias %.2f%% -> %.2f%%',
        store,
        origin,
        record['RMSSE_C5'],
        record['RMSSE_C6'],
        record['bias_C5_pct'],
        record['bias_C6_pct']
    )

    del (
        wide,
        matrix,
        main_bundle,
        pred5,
        pred6,
        comp5,
        comp6,
        actual_wide
    )

    gc.collect()

    return (
        record,
        pd.DataFrame(weeks),
        per_item
    )


# ============================================================
# 9. SU DUNG LAI KET QUA VALIDATION GOC d_1885
# ============================================================

def load_original_validation(
    results,
    c6_tag,
    destination,
    selected_stores
):
    """Import existing C5/C6 validation."""

    mdir = (
        results
        / 'metrics'
        / c6_tag
    )

    stage_a = (
        results
        / 'analysis'
        / c6_tag
        / 'stage_A'
    )

    original_metrics = pd.read_csv(
        mdir / 'comparison_by_store.csv'
    )

    original_a = pd.read_csv(
        stage_a / '02_by_store.csv'
    )

    c5_detail = pd.read_csv(
        results
        / 'metrics'
        / C5_TAG
        / 'C5_metrics_by_store.csv'
    )

    c5_detail = c5_detail.loc[
        c5_detail.model.eq(
            'lightgbm_two_part'
        )
    ]

    require(
        len(c5_detail) == 10,
        'Missing C5 per-store metrics for reference'
    )

    cal_detail = pd.read_csv(
        mdir / 'calibration_by_store.csv'
    )

    reference = (
        original_metrics
        .merge(
            original_a,
            on='store_id',
            validate='one_to_one'
        )
        .merge(
            c5_detail[
                [
                    'store_id',
                    'rmsse_valid_items',
                    'n_items'
                ]
            ],
            on='store_id',
            validate='one_to_one'
        )
        .merge(
            cal_detail[
                [
                    'store_id',
                    'calibration_train_end',
                    'calibration_days',
                    'observed_positive_rate',
                    'mean_probability_raw',
                    'mean_probability_calibrated',
                    'platt_slope',
                    'platt_intercept',
                    'brier_raw',
                    'brier_cal_in_sample'
                ]
            ],
            on='store_id',
            validate='one_to_one'
        )
    )

    rows = []

    for _, v in reference.loc[
        reference.store_id.isin(
            selected_stores
        )
    ].iterrows():

        rows.append({
            'origin': 1885,
            'store_id': v.store_id,
            'n_items': int(v.n_items),
            'valid_rmsse_items': int(
                v.rmsse_valid_items
            ),
            'actual': float(v.actual),
            'predicted_C5': float(
                v.forecast_C5
            ),
            'predicted_C6': float(
                v.forecast_C6
            ),
            'bias_C5_pct': float(
                v.bias_C5_pct_y
                if 'bias_C5_pct_y' in v
                else v.bias_C5_pct
            ),
            'bias_C6_pct': float(
                v.bias_C6_pct_y
                if 'bias_C6_pct_y' in v
                else v.bias_C6_pct
            ),
            'RMSSE_C5': float(v.RMSSE_C5),
            'RMSSE_C6': float(v.RMSSE_C6),
            'MAE_C5': float(v.MAE_C5),
            'MAE_C6': float(v.MAE_C6),
            'positive_rate': float(
                v.positive_rate_pct / 100
            ),
            'mean_p_C5': float(
                v.p_C5_mean_pct / 100
            ),
            'mean_p_raw_C6_path': float(
                v.p_raw_C6_path_pct / 100
            ),
            'mean_p_cal_C6_path': float(
                v.p_cal_C6_path_pct / 100
            ),
            'Brier_C5_path': float(
                v.Brier_C5_path
            ),
            'Brier_raw_C6_path': float(
                v.Brier_raw_C6_path
            ),
            'Brier_cal_C6_path': float(
                v.Brier_cal_C6_path
            ),
            'model_C5_source': 'existing_C5_d1885',
            'aux_model_source': 'existing_C6_aux_d1857',
            'aux_origin': 1857,
            'calibration_start': 1858,
            'calibration_end': 1885,
            'n_calibration_rows': (
                28 * int(v.n_items)
            ),
            'cal_positive_rate': float(
                v.observed_positive_rate
            ),
            'cal_mean_p_raw': float(
                v.mean_probability_raw
            ),
            'cal_mean_p_platt': float(
                v.mean_probability_calibrated
            ),
            'cal_brier_raw': float(
                v.brier_raw
            ),
            'cal_brier_platt_in_sample': float(
                v.brier_cal_in_sample
            ),
            'platt_slope': float(
                v.platt_slope
            ),
            'platt_intercept': float(
                v.platt_intercept
            ),
        })

    return rows


# ============================================================
# 10. CHUONG TRINH CHINH
# ============================================================

def run(args):

    config = yaml.safe_load(
        (
            ROOT / 'configs' / 'config.yaml'
        ).read_text(encoding='utf-8-sig')
    )

    results = (
        ROOT / config['results_dir']
    )

    data_dir = (
        ROOT / config['data_dir']
    )

    check_files(
        data_dir,
        use_prices=True
    )

    c5_manifest = read_json(
        results
        / 'metrics'
        / C5_TAG
        / 'manifest.json'
    )

    require(
        c5_manifest.get('status') == 'completed'
        and c5_manifest.get('configuration') == 'C5'
        and c5_manifest.get('origin_day') == 1885,
        'C5 reference is invalid'
    )

    c6_tag, c6_manifest = find_c6_batch(
        results,
        args.c6_batch
    )

    require(
        c6_manifest.get('calibration_train_end') == 1857
        and c6_manifest.get('validation_origin') == 1885
        and c6_manifest.get(
            'validation_not_used_for_calibration'
        ) is True,
        'C6 temporal design does not match Stage B'
    )

    require(
        c5_manifest.get('price_policy')
        == 'previous_completed_week',
        'C5 prices policy does not match Stage B'
    )

    stores = sorted(
        args.stores or STORE_CODES
    )

    require(
        len(set(stores)) == len(stores)
        and set(stores).issubset(STORE_CODES),
        'Invalid store names'
    )

    origins = sorted(
        set(args.origins)
    )

    require(
        origins
        and set(origins).issubset(FOLDS),
        'Origins must be 1829 and/or 1857'
    )

    if args.smoke_items:
        require(
            1 <= args.smoke_items <= 3049,
            'Invalid --smoke-items'
        )

    out_name = (
        f'stage_B_C5_C6_smoke_{args.smoke_items}_items'
        if args.smoke_items
        else 'stage_B_C5_C6_full'
    )

    destination = (
        results
        / 'analysis'
        / out_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    params = {
        name: {
            **c5_manifest['config'][name],
            'importance_type': 'gain'
        }
        for name in (
            'classifier',
            'regressor'
        )
    }

    calendar = load_calendar(
        data_dir
    )

    records = []
    weeks = []
    item_parts = []

    # ----------------------------------------------------
    # CHAY 2 GOC LICH SU MOI
    # ----------------------------------------------------

    for store in stores:

        for origin in origins:

            row, week, per_item = run_fold(
                store,
                origin,
                params,
                calendar,
                results,
                data_dir,
                c6_tag,
                destination,
                config['chunk_size'],
                smoke_items=args.smoke_items
            )

            records.append(row)
            weeks.append(week)
            item_parts.append(per_item)

    # ----------------------------------------------------
    # GHEP VALIDATION d_1885 DA CO
    # ----------------------------------------------------

    if not args.smoke_items:
        records.extend(
            load_original_validation(
                results,
                c6_tag,
                destination,
                stores
            )
        )

    by_store = (
        pd.DataFrame(records)
        .sort_values(
            ['origin', 'store_id']
        )
    )

    by_store['delta_RMSSE'] = (
        by_store.RMSSE_C6
        - by_store.RMSSE_C5
    )

    by_store['C6_improves_RMSSE'] = (
        by_store.delta_RMSSE < 0
    )

    by_store['delta_Brier_C6_path'] = (
        by_store.Brier_cal_C6_path
        - by_store.Brier_raw_C6_path
    )

    # ----------------------------------------------------
    # TONG HOP THEO GOC
    # ----------------------------------------------------

    by_origin = pd.DataFrame([
        aggregate_origin_records(group)

        for _, group in by_store.groupby(
            'origin',
            sort=True
        )
    ]).sort_values('origin')

    by_origin['delta_RMSSE'] = (
        by_origin.RMSSE_C6
        - by_origin.RMSSE_C5
    )

    by_origin['relative_RMSSE_change_pct'] = (
        100
        * by_origin.delta_RMSSE
        / by_origin.RMSSE_C5
    )

    by_origin['delta_MAE'] = (
        by_origin.MAE_C6
        - by_origin.MAE_C5
    )

    by_origin['delta_Brier_C6_path'] = (
        by_origin.Brier_cal_C6_path
        - by_origin.Brier_raw_C6_path
    )

    # ----------------------------------------------------
    # BANG DO BEN
    # ----------------------------------------------------

    summary = pd.DataFrame([
        {
            'origins_evaluated': int(
                len(by_origin)
            ),
            'origin_list': ','.join(
                map(str, by_origin.origin)
            ),
            'stores_evaluated': len(stores),

            'RMSSE_C5_mean': float(
                by_origin.RMSSE_C5.mean()
            ),
            'RMSSE_C6_mean': float(
                by_origin.RMSSE_C6.mean()
            ),

            'RMSSE_C5_std': float(
                by_origin.RMSSE_C5.std(ddof=1)
            ),
            'RMSSE_C6_std': float(
                by_origin.RMSSE_C6.std(ddof=1)
            ),

            'origins_C6_better_RMSSE': int(
                (by_origin.delta_RMSSE < 0).sum()
            ),

            'stores_origin_C6_better_RMSSE': int(
                by_store.C6_improves_RMSSE.sum()
            ),

            'stores_origin_total': int(
                len(by_store)
            ),

            'origins_C6_better_MAE': int(
                (by_origin.delta_MAE < 0).sum()
            ),

            'origins_cal_better_Brier': int(
                (
                    by_origin.delta_Brier_C6_path < 0
                ).sum()
            ),

            'max_abs_bias_C5_pct': float(
                by_origin.bias_C5_pct.abs().max()
            ),
            'max_abs_bias_C6_pct': float(
                by_origin.bias_C6_pct.abs().max()
            ),

            'mean_abs_bias_C5_pct': float(
                by_origin.bias_C5_pct.abs().mean()
            ),
            'mean_abs_bias_C6_pct': float(
                by_origin.bias_C6_pct.abs().mean()
            ),
        }
    ])

    # ----------------------------------------------------
    # XUAT CAC FILE BAO CAO
    # ----------------------------------------------------

    by_origin.to_csv(
        destination / '01_by_origin.csv',
        index=False,
        encoding='utf-8-sig'
    )

    by_store.to_csv(
        destination / '02_by_store_origin.csv',
        index=False,
        encoding='utf-8-sig'
    )

    summary.to_csv(
        destination / '03_robustness_summary.csv',
        index=False,
        encoding='utf-8-sig'
    )

    calibration_columns = [
        'origin',
        'store_id',
        'cal_positive_rate',
        'cal_mean_p_raw',
        'cal_mean_p_platt',
        'platt_slope',
        'platt_intercept',
        'Brier_raw_C6_path',
        'Brier_cal_C6_path',
        'delta_Brier_C6_path',
        'positive_rate',
        'mean_p_raw_C6_path',
        'mean_p_cal_C6_path',
    ]

    by_store[
        [
            col for col in calibration_columns
            if col in by_store
        ]
    ].to_csv(
        destination
        / '04_calibration_by_store_origin.csv',
        index=False,
        encoding='utf-8-sig'
    )

    if weeks:
        pd.concat(
            weeks,
            ignore_index=True
        ).to_csv(
            destination / '05_by_week_origin.csv',
            index=False,
            encoding='utf-8-sig'
        )

    if item_parts:
        pd.concat(
            item_parts,
            ignore_index=True
        ).to_csv(
            destination / '06_per_item_historical.csv',
            index=False,
            encoding='utf-8-sig'
        )

    write_json(
        destination / 'manifest.json',
        {
            'version': VERSION,
            'created_utc': datetime.now(
                timezone.utc
            ).isoformat(),
            'origins_requested': origins,
            'stores': stores,
            'smoke_items': args.smoke_items,
            'origins_in_summary': (
                by_origin.origin.to_list()
            ),
            'C5_batch': C5_TAG,
            'C6_batch': c6_tag,
            'historical_calibration_horizon': 28,
            'validation_horizon': 28,
            'existing_validation_reused': (
                not bool(args.smoke_items)
            ),
            'test_performed': False,
            'regression_and_classifier_hyperparams_frozen': True,
            'status': (
                'complete'
                if (
                    not args.smoke_items
                    and set(origins) == set(FOLDS)
                    and len(stores) == 10
                    and len(by_origin) == 3
                )
                else 'partial'
            ),
        }
    )

    # ----------------------------------------------------
    # IN BAO CAO NGAN GON
    # ----------------------------------------------------

    print(
        '\n=== STAGE B: SUMMARY BY ORIGIN ==='
    )

    print(
        by_origin[
            [
                'origin',
                'RMSSE_C5',
                'RMSSE_C6',
                'bias_C5_pct',
                'bias_C6_pct',
                'Brier_raw_C6_path',
                'Brier_cal_C6_path',
            ]
        ].to_string(index=False)
    )

    print(
        '\n=== STAGE B: ROBUSTNESS ==='
    )

    print(
        summary.to_string(index=False)
    )

    print(
        f'\nFiles saved: {destination}'
    )

    if (
        len(by_origin) != 3
        or len(stores) != 10
    ):
        print(
            'PARTIAL: run all ten stores at origins '
            '1829 and 1857 for full conclusion.'
        )


# ============================================================
# 11. DIEM CHAY
# ============================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--c6-batch',
        default=None
    )

    parser.add_argument(
        '--stores',
        nargs='+',
        default=None
    )

    parser.add_argument(
        '--origins',
        nargs='+',
        type=int,
        default=[1829, 1857]
    )

    parser.add_argument(
        '--smoke-items',
        type=int,
        default=None,
        help='Smoke-only number of item series/store; outputs isolated'
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    run(arguments)
