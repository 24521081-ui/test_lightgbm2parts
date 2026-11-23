"""Shared experiment helpers; entrypoints retain their frozen configurations."""
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnostics(frame):
    positive = frame.y_actual > 0
    return dict(
        actual=float(frame.y_actual.sum()),
        prediction=float(frame.y_pred_two_part.sum()),
        bias_pct=100 * (frame.y_pred_two_part.sum() / frame.y_actual.sum() - 1),
        p_gap_pp=100 * (frame.probability_positive.mean() - positive.mean()),
        mu_bias_pct=100 * (
            frame.loc[positive, "conditional_units"].sum()
            / frame.loc[positive, "y_actual"].sum() - 1
        ),
    )


def csv_save(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def print_validation_summary(row, configuration, origin, store, output, status="COMPLETE"):
    print(
        f"{status} {configuration} | d_{origin} | {store} "
        f"| valid={row['valid_rmsse_items']} "
        f"| RMSSE={row['RMSSE_C7']:.6f} | MAE={row['MAE_C7']:.6f} "
        f"| bias={row['bias_C7_pct']:+.2f}% | {output}",
        flush=True,
    )


def predict_blended(bundle, platt, matrix, items, store, state, calendar, origin,
                    prices, lam, *, require_finite_mu=False):
    """Run a separate 28-day path for each frozen lambda, updating its own lags."""
    from src.feature_engineering import next_day_features
    from src.forecasting import FORECAST_COLUMNS, validate_forecasts
    from run_c5_price_features import EXTRAS
    from run_c6_platt import apply_platt

    cal = calendar.set_index("d", verify_integrity=True)
    history = np.asarray(matrix[:, -28:], dtype=np.float32).copy()
    features = bundle["features"]
    require(history.shape == (len(items), 28), "History can dung 28 ngay.")
    forecasts, components = [], []
    for horizon in range(1, 29):
        day = f"d_{origin + horizon}"
        X = next_day_features(history, items, cal.loc[day], state,
                              bundle["categories"], last_prices=prices["sell_price"])
        for col in EXTRAS:
            X[col] = prices[col]
        require(X[features].columns.tolist() == features, "Feature order khac Stage B.")
        p_raw, mu = bundle["model"].predict_components(X[features])
        p_raw = np.asarray(p_raw, dtype=float)
        mu = np.asarray(mu, dtype=float)
        p_platt = apply_platt(platt, p_raw)
        p = (1 - lam) * p_raw + lam * p_platt
        yhat = np.maximum(p * mu, 0.0)
        require(np.isfinite(yhat).all() and np.isfinite(p).all()
                and ((p >= 0) & (p <= 1)).all(), f"C7 khong hop le o {day}")
        if require_finite_mu:
            require(np.isfinite(mu).all(), f"C7 chua NaN/Inf tai {day}.")
        forecasts.append(pd.DataFrame({
            "model": "lightgbm_two_part", "split": "validation",
            "origin_day": f"d_{origin}", "item_id": items, "store_id": store,
            "target_day": day, "horizon": horizon, "y_pred": yhat,
        }))
        components.append(pd.DataFrame({
            "target_day": day, "horizon": horizon, "item_id": items,
            "p_raw": p_raw, "p_platt": p_platt, "p_C7": p,
            "mu": mu, "yhat_C7": yhat,
        }))
        # Keep the original float32 start and float64 promotion after the first step.
        history = np.column_stack((history[:, 1:], yhat))
    pred = pd.concat(forecasts, ignore_index=True)[FORECAST_COLUMNS]
    validate_forecasts(pred)
    return pred, pd.concat(components, ignore_index=True)
