"""
src/evaluate.py
---------------
Load the registered 'champion' model from MLflow Model Registry and
compute final evaluation metrics on the held-out test split.
"""

import time
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    f1_score, accuracy_score, log_loss, classification_report
)

from src.data import load_and_split

REGISTRY_MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"


def evaluate_champion():
    """
    Load champion model from registry and evaluate on the test split.

    Returns
    -------
    dict with test metrics
    """
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    model_uri = f"models:/{REGISTRY_MODEL_NAME}@{CHAMPION_ALIAS}"
    print(f"[evaluate] Loading model: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    _, X_test, _, y_test, _ = load_and_split()

    # ---- Inference + latency ----
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    latency_ms = (time.perf_counter() - start) * 1000

    y_proba = model.predict_proba(X_test)

    # ---- Metrics ----
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_acc = accuracy_score(y_test, y_pred)
    test_logloss = log_loss(y_test, y_proba)

    print("\n=== Champion Model — Test Split Evaluation ===")
    print(f"  Macro F1-Score  : {test_f1:.4f}")
    print(f"  Accuracy        : {test_acc:.4f}")
    print(f"  Log Loss        : {test_logloss:.4f}")
    print(f"  Latency (batch) : {latency_ms:.2f} ms")
    report = classification_report(
        y_test, y_pred, target_names=["class_0", "class_1", "class_2"]
    )
    print("\n" + report)

    # ---- Output schema check ----
    unique_preds = set(int(v) for v in y_pred)
    valid_classes = {0, 1, 2}
    assert unique_preds.issubset(valid_classes), (
        f"Unexpected class indices in predictions: {unique_preds - valid_classes}"
    )
    print("[evaluate] Output schema OK — all predictions are in {0, 1, 2}.")

    return {
        "test_f1_macro": test_f1,
        "test_accuracy": test_acc,
        "test_log_loss": test_logloss,
        "inference_latency_ms": latency_ms,
        "predictions": y_pred.tolist(),
    }


if __name__ == "__main__":
    metrics = evaluate_champion()
