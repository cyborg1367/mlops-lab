
import os

from mlflow import MlflowClient


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "breast-cancer-classifier"
ALIAS = "champion"


def main():
    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    # 1. Check that the registered model exists.
    model = client.get_registered_model(MODEL_NAME)

    # 2. Preserve an existing alias.
    if ALIAS in model.aliases:
        version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=ALIAS,
        )

        print(
            f"Alias '{ALIAS}' already exists."
        )
        print(
            f"Keeping version {version.version}."
        )
        return

    # 3. Find the latest READY model version.
    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'"
    )

    ready_versions = [
        version
        for version in versions
        if version.status == "READY"
    ]

    if not ready_versions:
        raise RuntimeError(
            f"No READY versions found for {MODEL_NAME}."
        )

    latest_version = max(
        ready_versions,
        key=lambda version: int(version.version),
    )

    # 4. Initialize the alias.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=latest_version.version,
    )

    # 5. Verify the result.
    selected = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=ALIAS,
    )

    print("Registry initialization successful.")
    print(f"Model: {MODEL_NAME}")
    print(f"Alias: {ALIAS}")
    print(f"Version: {selected.version}")


if __name__ == "__main__":
    main()
