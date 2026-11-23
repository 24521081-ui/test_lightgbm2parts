"""Regression checks for frozen C7 paths, independent of trained artifacts."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_common import diagnostics, predict_blended
from run_c7_lambda050 import predict_c7 as predict_050
from run_c7_lambda075 import predict_c7 as predict_075
import run_c7_lambda050
import run_c7_lambda075


class EchoLag:
    def predict_components(self, X):
        return np.full(len(X), 0.5), X.lag_1.to_numpy()


class ConstantPlatt:
    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))


def arguments():
    return (
        {"features": ["lag_1"], "categories": ["B", "A"], "model": EchoLag()},
        ConstantPlatt(), np.array([np.arange(60), np.arange(60) + 10]),
        ["B", "A"], "CA_1", "CA",
        pd.DataFrame({"d": [f"d_{i}" for i in range(61, 89)],
                      "day_of_week": np.arange(28) % 7, "month": 1,
                      "event_flag": 0, "snap_CA": 0}),
        60, {"sell_price": np.ones(2), "price_change_1w": np.zeros(2),
             "price_ratio_4w": np.ones(2)},
    )


@pytest.mark.parametrize("predict,lam", [(predict_050, 0.5), (predict_075, 0.75)])
def test_frozen_lambda_recurses_with_own_product(predict, lam):
    args = arguments()
    forecast, components = predict(*args)
    p = (1 - lam) * 0.5 + lam * 0.25
    assert len(forecast) == 56
    for item, start in [("B", 59), ("A", 69)]:
        expected = []
        for _ in range(28):
            # next_day_features exposes lag columns as float32 in the original code.
            start = float(np.float32(start)) * p
            expected.append(start)
        np.testing.assert_array_equal(
            forecast.loc[forecast.item_id == item, "y_pred"], expected)
    np.testing.assert_array_equal(components.p_C7, np.full(56, p))
    np.testing.assert_array_equal(forecast.y_pred, components.p_C7 * components.mu)
    np.testing.assert_array_equal(args[2][0], np.arange(60))


def test_rejects_invalid_calibrator_probability():
    class InvalidPlatt:
        def predict_proba(self, X):
            return np.tile([-1, 2], (len(X), 1))
    args = list(arguments())
    args[1] = InvalidPlatt()
    with pytest.raises(ValueError, match="Calibrated probability invalid"):
        predict_blended(*args, 0.75)


def test_diagnostics_preserves_signed_bias():
    frame = pd.DataFrame({"y_actual": [0., 2.], "y_pred_two_part": [1., 2.],
                          "probability_positive": [0.5, 0.5], "conditional_units": [2., 4.]})
    assert diagnostics(frame) == {"actual": 2., "prediction": 3., "bias_pct": 50.,
                                  "p_gap_pp": 0., "mu_bias_pct": 100.}


@pytest.mark.parametrize("module", [run_c7_lambda050, run_c7_lambda075])
@pytest.mark.parametrize("stale", [False, True])
def test_resume_reuses_valid_cache_and_rejects_stale(tmp_path, monkeypatch, capsys, module, stale):
    stage_b = tmp_path / "stage_b"
    reference = stage_b / "folds/d_1829/CA_1"
    reference.mkdir(parents=True)
    (reference / "completed.json").write_text(json.dumps({
        "version": module.B_VERSION, "origin": 1829, "store_id": "CA_1", "record": {},
    }))
    output = tmp_path / "output"
    folder = output / "folds/d_1829/CA_1"
    folder.mkdir(parents=True)
    row = {"RMSSE_C7": .7, "MAE_C7": 1., "bias_C7_pct": -2., "valid_rmsse_items": 3}
    (folder / "completed.json").write_text(json.dumps({
        "version": module.C7_VERSION, "lambda": -1 if stale else module.LAM,
        "origin": 1829, "store_id": "CA_1", "record": row,
    }))
    for name in ["per_item.csv", "by_week.csv", "forecast_C7.csv"]:
        (folder / name).write_text("item_id,value\nA,1\n")
    def forbid_load(*args, **kwargs):
        pytest.fail("Resume must not load data or run inference")
    monkeypatch.setattr(module, "load_store", forbid_load)
    args = ("CA_1", 1829, tmp_path, {}, None, tmp_path, stage_b, "unused", output)
    if stale:
        with pytest.raises(ValueError):
            module.run_store(*args)
    else:
        result, weeks, items = module.run_store(*args)
        assert result == row and len(weeks) == len(items) == 1
        text = capsys.readouterr().out
        assert "REUSE" in text and "valid=3" in text and "MAE=1.000000" in text
