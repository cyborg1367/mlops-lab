import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def main():
    # 1. Load configuration
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    train_params = config["train"]

    # 2. Load best hyperparameters from Cross Validation
    best_params_path = Path("reports/cv/best_params.json")

    with open(best_params_path, "r") as file:
        cv_results = json.load(file)

    best_params = cv_results["model_params"]

    # 3. Load the full Development Set
    development_df = pd.read_csv(
        "data/processed/development.csv"
    )

    X_train = development_df.drop(columns=["target"])
    y_train = development_df["target"]

    # 4. Initialize the final model
    model = RandomForestClassifier(
        **best_params,
        random_state=train_params["random_state"],
        n_jobs=-1,
    )

    # 5. Train on the entire Development Set
    model.fit(X_train, y_train)

    # 6. Save the trained model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # 7. Report results
    print("Final model training completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Selected hyperparameters: {best_params}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
