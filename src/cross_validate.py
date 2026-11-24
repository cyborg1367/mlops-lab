
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


EXPERIMENT_NAME = "breast-cancer-classification"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def load_configuration():
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config["cv"], config["train"]


def load_development_data():
    df = pd.read_csv("data/processed/development.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def main():
    # 1. Load configuration
    cv_params, train_params = load_configuration()

    # 2. Load development data
    X, y = load_development_data()

    # 3. Configure Stratified K-Fold
    stratified_cv = StratifiedKFold(
        n_splits=cv_params["n_splits"],
        shuffle=True,
        random_state=cv_params["random_state"],
    )

    # 4. Define evaluation metrics
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            pos_label=1,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            pos_label=1,
            zero_division=0,
        ),
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    selection_metric = cv_params["selection_metric"]

    if selection_metric not in scoring:
        raise ValueError(
            f"Unknown selection metric: {selection_metric}"
        )

    # 5. Initialize model
    model = RandomForestClassifier(
        random_state=train_params["random_state"],
        n_jobs=1,
    )

    # 6. Configure Grid Search
    search = GridSearchCV(
        estimator=model,
        param_grid=cv_params["param_grid"],
        scoring=scoring,
        refit=selection_metric,
        cv=stratified_cv,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    # 7. Connect to MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 8. Run Cross-Validation
    print(
        f"Starting Stratified {cv_params['n_splits']}-Fold Cross-Validation..."
    )
    print(f"Development samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Selection metric: {selection_metric}")

    search.fit(X, y)

    print("\nCross-Validation completed.")
    print(f"Best parameters: {search.best_params_}")

    # 9. Extract all CV results
    results = pd.DataFrame(search.cv_results_)

    best_index = search.best_index_

    summary_metrics = {}

    for metric in scoring:
        mean_score = results.loc[
            best_index, f"mean_test_{metric}"
        ]

        std_score = results.loc[
            best_index, f"std_test_{metric}"
        ]

        summary_metrics[f"cv_{metric}_mean"] = float(mean_score)
        summary_metrics[f"cv_{metric}_std"] = float(std_score)

    # 10. Save outputs
    output_dir = Path("reports/cv")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "cv_results.csv"
    best_params_path = output_dir / "best_params.json"

    results.to_csv(results_path, index=False)

    best_params = {
        "model_params": search.best_params_,
        "selection_metric": selection_metric,
        "best_cv_score": float(search.best_score_),
        "cv_metrics": summary_metrics,
    }

    with open(best_params_path, "w") as file:
        json.dump(best_params, file, indent=4)

    # 11. Log experiment in MLflow
    with mlflow.start_run(
        run_name="random-forest-stratified-5fold-cv"
    ) as run:

        mlflow.log_params({
            "cv_n_splits": cv_params["n_splits"],
            "cv_random_state": cv_params["random_state"],
            "selection_metric": selection_metric,
            "n_candidates": len(results),
        })

        mlflow.log_params({
            f"best_{key}": str(value)
            for key, value in search.best_params_.items()
        })

        mlflow.log_metrics(summary_metrics)

        mlflow.set_tags({
            "model_type": "RandomForestClassifier",
            "dataset": "Breast Cancer Wisconsin",
            "validation_strategy": "StratifiedKFold",
            "pipeline_stage": "cross_validate",
            "positive_class": "malignant",
        })

        mlflow.log_artifact(str(results_path), artifact_path="cv")
        mlflow.log_artifact(str(best_params_path), artifact_path="cv")

        print(f"\nMLflow Run ID: {run.info.run_id}")

    # 12. Print summary
    print("\nBest model CV metrics:")

    for name, value in summary_metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nResults saved to: {results_path}")
    print(f"Best parameters saved to: {best_params_path}")


if __name__ == "__main__":
    main()
