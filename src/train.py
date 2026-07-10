"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost on the Heart Disease
UCI dataset, tunes each with GridSearchCV (5-fold CV), logs everything to
MLflow, and persists the best pipeline (preprocessing + model bundled
together, so there is no train/serve skew) to models/model.joblib.

Run: python src/train.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import build_preprocessor, load_dataset

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "heart_disease.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("heart-disease-classification")

RANDOM_STATE = 42

MODEL_GRID = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
        },
    ),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
    ),
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_confusion_matrix(y_true, y_pred, run_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix — {run_name}")
    path = FIG_DIR / f"confusion_matrix_{run_name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def log_roc_curve(y_true, y_proba, run_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"ROC curve — {run_name}")
    path = FIG_DIR / f"roc_curve_{run_name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    X, y = load_dataset(str(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] train={X_train.shape[0]} test={X_test.shape[0]}")

    results = []

    for name, (estimator, param_grid) in MODEL_GRID.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ])
            grid = GridSearchCV(
                pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True
            )
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_

            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            # log_params requires plain values, not e.g. numpy types
            clean_params = {k: (v if v is None or isinstance(v, (int, float, str, bool)) else str(v))
                             for k, v in grid.best_params_.items()}
            mlflow.log_params(clean_params)
            mlflow.log_metric("cv_best_roc_auc", grid.best_score_)
            mlflow.log_metrics(metrics)

            cm_path = log_confusion_matrix(y_test, y_pred, name)
            roc_path = log_roc_curve(y_test, y_proba, name)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            trusted_types = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
            mlflow.sklearn.log_model(best_pipeline, name="model", skops_trusted_types=trusted_types)

            run_id = mlflow.active_run().info.run_id
            results.append({"model": name, "run_id": run_id, "best_params": grid.best_params_, **metrics,
                             "cv_best_roc_auc": grid.best_score_, "pipeline": best_pipeline})

            print(f"[train] {name}: {metrics} (best_params={grid.best_params_})")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
    results_df = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    print("\n[train] model comparison (sorted by test ROC-AUC):")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_best_roc_auc"]]
          .to_string(index=False))

    best = max(results, key=lambda r: r["roc_auc"])
    best_pipeline = best["pipeline"]

    import joblib
    joblib.dump(best_pipeline, MODELS_DIR / "model.joblib")

    metadata = {
        "best_model": best["model"],
        "mlflow_run_id": best["run_id"],
        "metrics": {k: best[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        "best_params": best["best_params"],
        "feature_columns": list(X.columns),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n[train] BEST MODEL: {best['model']} (test ROC-AUC={best['roc_auc']:.4f})")
    print(f"[train] saved -> {MODELS_DIR / 'model.joblib'}")
    print(f"[train] mlflow run id -> {best['run_id']}")


if __name__ == "__main__":
    main()
