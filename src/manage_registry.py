
import argparse

import mlflow
import mlflow.pyfunc
import pandas as pd

from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "breast-cancer-classifier"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and promote an MLflow model version."
    )

    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Registered model version to promote.",
    )

    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="Alias to assign after successful validation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Connect to MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    # 2. Validate that the requested version exists and is READY
    version = client.get_model_version(
        name=MODEL_NAME,
        version=str(args.version),
    )

    if version.status != "READY":
        raise RuntimeError(
            f"Model version {args.version} is not READY."
        )

    print(f"Found model version: {version.version}")

    # 3. Load the candidate version directly
    model_uri = (
        f"models:/{MODEL_NAME}/{args.version}"
    )

    print(f"\nLoading candidate model: {model_uri}")

    candidate_model = mlflow.pyfunc.load_model(
        model_uri
    )

    # 4. Generate synthetic smoke-test input
    dataset = load_breast_cancer()

    input_df = pd.DataFrame(
        dataset.data[:5],
        columns=dataset.feature_names,
    )

    # 5. Run smoke test
    predictions = candidate_model.predict(input_df)

    if len(predictions) != len(input_df):
        raise RuntimeError(
            "Smoke test failed: unexpected prediction count."
        )

    if not set(predictions).issubset({0, 1}):
        raise RuntimeError(
            "Smoke test failed: invalid prediction values."
        )

    print("\nSmoke test passed.")
    print(f"Candidate version: {args.version}")
    print(f"Predictions: {predictions}")

    # 6. Promote candidate only after successful smoke test
    print(
        f"\nAssigning alias '{args.alias}' "
        f"to version {args.version}..."
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=args.alias,
        version=str(args.version),
    )

    # 7. Verify alias assignment
    selected_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=args.alias,
    )

    if selected_version.version != str(args.version):
        raise RuntimeError(
            "Alias verification failed: "
            f"expected version {args.version}, "
            f"got version {selected_version.version}."
        )

    print("\nRegistry status:")
    print(f"Model: {selected_version.name}")
    print(f"Version: {selected_version.version}")
    print(f"Alias: {args.alias}")

    print(
        f"\nAlias '{args.alias}' successfully points "
        f"to version {selected_version.version}."
    )


if __name__ == "__main__":
    main()
