"""
src/evaluate.py
---------------
Load the registered 'champion' model from MLflow Model Registry and
compute final evaluation metrics on the held-out test split.

Compatible with: mlflow==2.16.2, scikit-learn==1.5.2, Python 3.10
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
    """Load champion model from registry, evaluate on test split."""
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    model_uri = f"models:/{REGISTRY_MODEL_NAME}@{CHAMPION_ALIAS}"
    print(f"[evaluate] Loading: {model_uri}")

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

    print("\n=== Champion Model — Test Split ===")
    print(f"  Macro F1-Score  : {test_f1:.4f}")
    print(f"  Accuracy        : {test_acc:.4f}")
    print(f"  Log Loss        : {test_logloss:.4f}")
    print(f"  Latency (batch) : {latency_ms:.2f} ms")

    report = classification_report(
        y_test, y_pred,
        target_names=["class_0", "class_1", "class_2"]
    )
    print("\n" + report)

    # ---- Output schema check ----
    unique_preds = set(int(v) for v in y_pred)
    assert unique_preds.issubset({0, 1, 2}), (
        f"Unexpected classes: {unique_preds - {0, 1, 2}}"
    )
    print("[evaluate] Schema OK — predictions in {0, 1, 2}.")

    return {
        "test_f1_macro":        test_f1,
        "test_accuracy":        test_acc,
        "test_log_loss":        test_logloss,
        "inference_latency_ms": latency_ms,
        "predictions":          y_pred.tolist(),
    }


if __name__ == "__main__":
    evaluate_champion()
