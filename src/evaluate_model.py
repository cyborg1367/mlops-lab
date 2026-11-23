
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def main():
    # 1. Load test data
    test_path = Path("data/processed/test.csv")
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # 2. Load trained model
    model_path = Path("models/model.joblib")
    model = joblib.load(model_path)

    # 3. Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 4. Calculate evaluation metrics
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

    # 5. Calculate confusion matrix
    # Class 0 = Benign
    # Class 1 = Malignant (positive class)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    metrics["test_samples"] = int(len(y_test))

    # 6. Save metrics as JSON
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    # 7. Plot and save confusion matrix
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

    # 8. Print results
    print("Model evaluation completed.")
    print(f"Test samples: {len(y_test)}")
    print("\nEvaluation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {figure_path}")


if __name__ == "__main__":
    main()
