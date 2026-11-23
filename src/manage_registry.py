import mlflow
import mlflow.pyfunc

from mlflow import MlflowClient
import pandas as pd


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "breast-cancer-classifier"
MODEL_ALIAS = "champion"
MODEL_VERSION = "1"


def main():
    # 1. Connect to MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    # 2. Assign champion alias to version 1
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=MODEL_VERSION,
    )

    print("Alias assigned successfully.")

    # 3. Retrieve model version using alias
    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    print(f"Model name: {model_version.name}")
    print(f"Model version: {model_version.version}")
    print(f"Model alias: {MODEL_ALIAS}")

    # 4. Load model using alias
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    model = mlflow.pyfunc.load_model(model_uri)

    # 5. Test inference
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=["target"])

    predictions = model.predict(X_test.head(5))

    print("\nPredictions:")
    print(predictions)


if __name__ == "__main__":
    main()