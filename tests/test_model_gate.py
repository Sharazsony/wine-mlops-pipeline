"""
tests/test_model_gate.py
------------------------
MLOps Quality Gate — asserts the champion model meets production thresholds:
  1. Validation Macro F1-score >= 0.88
  2. Batch inference latency <= 30 ms
  3. Output schema: predictions must only be class indices {0, 1, 2}
"""

import time
import pytest
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score

from src.data import load_and_split

REGISTRY_MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"

F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0
VALID_CLASS_INDICES = {0, 1, 2}


@pytest.fixture(scope="module")
def champion_model():
    """Load the champion model once for all gate tests."""
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    model_uri = f"models:/{REGISTRY_MODEL_NAME}@{CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


@pytest.fixture(scope="module")
def test_data():
    """Load test split once for all gate tests."""
    _, X_test, _, y_test, _ = load_and_split()
    return X_test, y_test


class TestMetricThresholdGate:
    """Gate 1 — Macro F1-score must be >= 0.88 on the test split."""

    def test_f1_above_threshold(self, champion_model, test_data):
        X_test, y_test = test_data
        y_pred = champion_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")

        assert f1 >= F1_THRESHOLD, (
            f"Quality gate FAILED: Macro F1 {f1:.4f} < threshold {F1_THRESHOLD}. "
            "Model is not production-ready."
        )
        print(f"\n[gate] F1 gate PASSED: {f1:.4f} >= {F1_THRESHOLD}")


class TestInferenceLatencyGate:
    """Gate 2 — Batch inference on the test split must complete in <= 30 ms."""

    def test_latency_under_threshold(self, champion_model, test_data):
        X_test, _ = test_data

        # Warm-up run (avoids JIT / import overhead)
        champion_model.predict(X_test)

        # Timed run
        start = time.perf_counter()
        champion_model.predict(X_test)
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms <= LATENCY_THRESHOLD_MS, (
            f"Latency gate FAILED: {latency_ms:.2f} ms > {LATENCY_THRESHOLD_MS} ms."
        )
        print(f"\n[gate] Latency gate PASSED: {latency_ms:.2f} ms <= {LATENCY_THRESHOLD_MS} ms")


class TestOutputSchemaGate:
    """Gate 3 — All predictions must be class indices 0, 1, or 2 only."""

    def test_predictions_are_valid_class_indices(self, champion_model, test_data):
        X_test, _ = test_data
        y_pred = champion_model.predict(X_test)

        unique_preds = set(int(v) for v in y_pred)
        invalid = unique_preds - VALID_CLASS_INDICES

        assert not invalid, (
            f"Schema gate FAILED: unexpected class index(es) {invalid} in predictions. "
            f"Valid indices are {VALID_CLASS_INDICES}."
        )
        print(f"\n[gate] Schema gate PASSED: predictions={sorted(unique_preds)}")

    def test_prediction_count_matches_input(self, champion_model, test_data):
        X_test, _ = test_data
        y_pred = champion_model.predict(X_test)

        assert len(y_pred) == len(X_test), (
            f"Schema gate FAILED: expected {len(X_test)} predictions, got {len(y_pred)}."
        )

    def test_predictions_are_integers(self, champion_model, test_data):
        X_test, _ = test_data
        y_pred = champion_model.predict(X_test)

        assert all(isinstance(int(v), int) for v in y_pred), (
            "Schema gate FAILED: predictions contain non-integer values."
        )
