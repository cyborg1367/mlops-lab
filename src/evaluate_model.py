
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


EXPERIMENT_NAME = "breast-cancer-classification"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def main():

    # 1. Configure MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3. Load test data
    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # 4. Load trained model
    model_path = Path("models/model.joblib")
    model = joblib.load(model_path)

    # 5. Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 6. Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # 7. Calculate confusion matrix
    # Class 0 = Benign
    # Class 1 = Malignant

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    metrics["test_samples"] = int(len(y_test))

    # 8. Save local evaluation outputs
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign (0)", "Malignant (1)"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()

    figure_path = reports_dir / "confusion_matrix.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    model_params = model.get_params()

    # 9. Create MLflow run
    run_name = (
        f"rf-n{model_params['n_estimators']}"
        f"-depth{model_params['max_depth']}"
    )

    with mlflow.start_run(run_name=run_name) as run:

        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "random_state": model_params["random_state"],
        })

        # Log scalar metrics
        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

        mlflow.log_metrics(scalar_metrics)

        # Log metadata
        mlflow.set_tags({
            "model_type": "RandomForestClassifier",
            "dataset": "Breast Cancer Wisconsin",
            "source": "DVC Pipeline",
            "dvc_stage": "evaluate_model",
        })

        # Log evaluation artifacts
        mlflow.log_artifact(
            str(figure_path),
            artifact_path="evaluation",
        )

        mlflow.log_dict(
            metrics,
            "evaluation/metrics.json",
        )

        # Log trained model
        signature = infer_signature(X_test, y_pred)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random-forest-model",
            signature=signature,
            input_example=X_test.head(5),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
            pip_requirements="requirements-model.txt",
        )

        print("\nMLflow run completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Model URI: {model_info.model_uri}")

    # 10. Print evaluation results
    print("\nModel evaluation completed.")
    print(f"Test samples: {len(y_test)}")

    print("\nEvaluation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {figure_path}")


if __name__ == "__main__":
    main()
