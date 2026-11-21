import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate_final_test import calculate_deciles, predict, score_model
from src.modeling.train_baseline import evaluate
from src.modeling.train_expanded_model import risk_deciles


def test_metric_values_for_perfect_predictions():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert m["roc_auc"] == 1.0
    assert m["average_precision"] == 1.0
    assert m["brier_score"] == 0.0


def test_metric_values_for_reversed_predictions():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([1.0, 0.0, 1.0, 0.0]))
    assert m["roc_auc"] == 0.0
    assert m["brier_score"] == 1.0


def test_final_deciles_preserve_counts_and_order():
    y = np.array([int(i % 7 == 0) for i in range(100)])
    p = np.linspace(0.02, 0.9, 100)
    result = calculate_deciles(y, p, "synthetic")
    assert result["decile"].tolist() == list(range(1, 11))
    assert result["loans"].tolist() == [10] * 10
    assert result["charged_off"].sum() == y.sum()
    assert result["mean_prediction"].is_monotonic_increasing
    assert set(result["model"]) == {"synthetic"}
    assert result["predicted_pct"].between(0, 100).all()


def test_tied_predictions_produce_ten_groups():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    result = calculate_deciles(y, p, "tied")
    assert len(result) == 10
    assert result["loans"].tolist() == [10] * 10
    assert (result["mean_prediction"] == 0.5).all()


def test_training_and_final_decile_implementations_agree():
    y = pd.Series([0, 1] * 50)
    p = np.linspace(0.01, 0.99, 100)
    training = risk_deciles(y, p)
    final = calculate_deciles(y.to_numpy(), p, "synthetic")
    for col in ["decile", "loans", "charged_off", "predicted_pct", "observed_pct"]:
        np.testing.assert_allclose(training[col], final[col])


def test_model_score_bias_and_top_decile_capture():
    y = np.array([0] * 90 + [1] * 10)
    p = np.linspace(0.01, 0.95, 100)
    metrics, deciles = score_model("synthetic", y, p)
    assert metrics["top_decile_capture_pct"] == pytest.approx(100.0)
    assert metrics["top_bottom_risk_ratio"] is None
    assert metrics["observed_pct"] == pytest.approx(10.0)
    assert metrics["prediction_bias_pp"] == pytest.approx(100 * (p.mean() - 0.1))
    assert len(deciles) == 10


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features):
        p = self.probabilities
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("probabilities,invalid", [
    ([0.1, 0.9], False),
    ([float("nan"), 0.1], True),
    ([1.1, 0.1], True),
    ([-0.1, 0.9], True),
    ([float("inf"), 0.9], True),
])
def test_prediction_probability_contract(monkeypatch, tmp_path, probabilities, invalid):
    import src.modeling.evaluate_final_test as final
    monkeypatch.setattr(final.joblib, "load", lambda path: FakeModel(probabilities))
    path = tmp_path / "fake.joblib"
    features = pd.DataFrame({"x": [1, 2]})
    if invalid:
        with pytest.raises(ValueError, match="Invalid predictions|outside"):
            predict(path, features)
    else:
        np.testing.assert_allclose(predict(path, features), probabilities)
