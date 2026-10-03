"""
src/train.py
------------
Train RandomForestClassifier and GradientBoostingClassifier on the Wine
dataset. Logs all hyperparameter configurations to MLflow, runs 5-fold
stratified cross-validation, and registers the best model as 'WineClassifier'
with alias 'champion'.

Compatible with: mlflow==2.16.2, scikit-learn==1.5.2, Python 3.10
"""

import time
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score, log_loss

from src.data import load_and_split

RANDOM_STATE = 42
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTRY_MODEL_NAME = "WineClassifier"
CV_FOLDS = 5

# ------------------------------------------------------------------
# Hyperparameter search grids (3 configs per family as required)
# ------------------------------------------------------------------
RF_GRID = [
    {"n_estimators": 50,  "max_depth": 3,    "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 4},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
]

GBM_GRID = [
    {"n_estimators": 50,  "learning_rate": 0.1,  "max_depth": 2},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 150, "learning_rate": 0.01, "max_depth": 4},
]


def _cv_metrics(model, X_train, y_train):
    """Run 5-fold stratified CV and return mean metrics dict."""
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=["accuracy", "f1_macro", "neg_log_loss"],
        return_train_score=True,
        n_jobs=-1
    )
    return {
        "cv_train_accuracy": float(np.mean(cv_results["train_accuracy"])),
        "cv_val_accuracy":   float(np.mean(cv_results["test_accuracy"])),
        "cv_train_f1_macro": float(np.mean(cv_results["train_f1_macro"])),
        "cv_val_f1_macro":   float(np.mean(cv_results["test_f1_macro"])),
        "cv_val_log_loss":   float(-np.mean(cv_results["test_neg_log_loss"])),
    }


def train_and_log(
    model_family, params, X_train, y_train, X_test, y_test
):
    """Train one config, log to MLflow, return run metadata."""
    params_str = "_".join(str(v) for v in params.values())
    run_name = f"{model_family}_{params_str}"

    with mlflow.start_run(run_name=run_name) as run:

        # ---- Build model ----
        if model_family == "RandomForest":
            model = RandomForestClassifier(
                random_state=RANDOM_STATE, **params
            )
        else:
            model = GradientBoostingClassifier(
                random_state=RANDOM_STATE, **params
            )

        # ---- Cross-validation ----
        cv_metrics = _cv_metrics(model, X_train, y_train)

        # ---- Full fit on training split ----
        model.fit(X_train, y_train)

        # ---- Test split metrics ----
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        test_acc = accuracy_score(y_test, y_pred)
        test_logloss = log_loss(y_test, y_proba)

        # ---- Inference latency ----
        start = time.perf_counter()
        model.predict(X_test)
        latency_ms = (time.perf_counter() - start) * 1000

        # ---- MLflow: tags, params, metrics ----
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("cv_folds", str(CV_FOLDS))
        mlflow.log_params({"model_family": model_family, **params})
        mlflow.log_metrics({
            **cv_metrics,
            "test_f1_macro":        test_f1,
            "test_accuracy":        test_acc,
            "test_log_loss":        test_logloss,
            "inference_latency_ms": latency_ms,
        })

        # ---- Model artifact + signature ----
        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print(
            f"  [{model_family}] {params} | "
            f"cv_f1={cv_metrics['cv_val_f1_macro']:.4f} | "
            f"test_f1={test_f1:.4f} | "
            f"lat={latency_ms:.1f}ms | "
            f"run={run.info.run_id[:8]}"
        )

        return {
            "run_id":          run.info.run_id,
            "cv_val_f1_macro": cv_metrics["cv_val_f1_macro"],
            "model_family":    model_family,
            "params":          params,
        }


def run_experiment():
    """Run all configs, register champion model."""
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, _ = load_and_split()
    all_runs = []

    print("\n=== RandomForest Configurations ===")
    for params in RF_GRID:
        result = train_and_log(
            "RandomForest", params, X_train, y_train, X_test, y_test
        )
        all_runs.append(result)

    print("\n=== GradientBoosting Configurations ===")
    for params in GBM_GRID:
        result = train_and_log(
            "GradientBoosting", params, X_train, y_train, X_test, y_test
        )
        all_runs.append(result)

    # ---- Champion: best cv_val_f1_macro ----
    champion = max(all_runs, key=lambda r: r["cv_val_f1_macro"])
    print(
        f"\n=== Champion: {champion['model_family']} | "
        f"cv_f1={champion['cv_val_f1_macro']:.4f} | "
        f"run={champion['run_id'][:8]} ==="
    )

    # ---- Register in Model Registry ----
    model_uri = f"runs:/{champion['run_id']}/model"
    registered = mlflow.register_model(
        model_uri=model_uri, name=REGISTRY_MODEL_NAME
    )

    # MLflow 2.x: use transition_model_version_stage or set alias
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=REGISTRY_MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print(
        f"[registry] '{REGISTRY_MODEL_NAME}' "
        f"v{registered.version} -> alias='champion'"
    )
    return champion


if __name__ == "__main__":
    run_experiment()
