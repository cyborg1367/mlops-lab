
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def main():
    # Load configuration
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = config["train"]

    # Load training data
    train_df = pd.read_csv("data/processed/train.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    # Train
    model.fit(X_train, y_train)

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    print("Model training completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Model saved to: {model_path}")
    print(f"Model parameters: {params}")


if __name__ == "__main__":
    main()
