
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "breast-cancer-classification"


def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def load_parameters():
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config["train"]


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return metrics, cm, y_pred


def main():
    # 1. Connect to MLflow Tracking Server
    mlflow.set_tracking_uri(TRACKING_URI)

    # 2. Select or create an experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3. Load data and model parameters
    X_train, X_test, y_train, y_test = load_data()
    params = load_parameters()

    # 4. Train the model
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 5. Evaluate the model
    metrics, cm, y_pred = evaluate_model(
        model, X_test, y_test
    )

    # 6. Start an MLflow run and log everything
    with mlflow.start_run(
        run_name="random-forest-baseline"
    ) as run:

        # Log hyperparameters
        mlflow.log_params(params)

        # Log dataset information
        mlflow.log_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log useful metadata
        mlflow.set_tags({
            "model_type": "RandomForestClassifier",
            "dataset": "Breast Cancer Wisconsin",
            "positive_class": "malignant",
            "framework": "scikit-learn",
        })

        # Log confusion matrix as an artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = Path(tmp_dir) / "confusion_matrix.png"

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Benign (0)", "Malignant (1)"],
            )

            display.plot(cmap="Blues")
            plt.title("Random Forest - Confusion Matrix")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close()

            mlflow.log_artifact(
                str(cm_path),
                artifact_path="evaluation",
            )

        # Log metrics as a JSON artifact
        mlflow.log_dict(
            metrics,
            "evaluation/metrics.json",
        )

        # Infer model signature
        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        # Log the trained model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random-forest-model",
            signature=signature,
            input_example=X_train.head(5),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        print("\nMLflow run completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Model URI: {model_info.model_uri}")

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
