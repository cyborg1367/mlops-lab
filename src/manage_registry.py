
import argparse

import mlflow
import mlflow.pyfunc

from mlflow import MlflowClient
import pandas as pd


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "breast-cancer-classifier"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manage MLflow model registry aliases."
    )

    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version to assign to the alias.",
    )

    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="Alias to assign. Default: champion",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Connect to MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    # 2. Validate that the requested version exists
    version = client.get_model_version(
        name=MODEL_NAME,
        version=str(args.version),
    )

    print(
        f"Found model version: {version.version}"
    )

    # 3. Assign alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=args.alias,
        version=str(args.version),
    )

    print(
        f"\nAlias '{args.alias}' assigned "
        f"to version {args.version}."
    )

    # 4. Verify alias
    selected_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=args.alias,
    )

    print("\nRegistry status:")
    print(f"Model: {selected_version.name}")
    print(f"Version: {selected_version.version}")
    print(f"Alias: {args.alias}")

    # 5. Load model through alias
    model_uri = (
        f"models:/{MODEL_NAME}@{args.alias}"
    )

    model = mlflow.pyfunc.load_model(model_uri)

    # 6. Test inference
    test_df = pd.read_csv(
        "data/processed/test.csv"
    )

    X_test = test_df.drop(columns=["target"])

    predictions = model.predict(
        X_test.head(5)
    )

    print("\nPredictions:")
    print(predictions)


if __name__ == "__main__":
    main()
