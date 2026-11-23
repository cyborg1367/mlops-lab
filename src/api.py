import math
import os

from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel, Field

# Configuration
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_NAME = "breast-cancer-classifier"
MODEL_ALIAS = "champion"


# Request / Response schemas
class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        description="Feature names and their numeric values."
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_label: str
    model_name: str
    model_version: str
    model_alias: str


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect to MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
    )

    # 2. Resolve the current alias
    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    # 3. Load the model through its alias
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    model = mlflow.pyfunc.load_model(model_uri)

    # 4. Read expected input columns from model signature
    signature = model.metadata.signature

    if signature is None or signature.inputs is None:
        raise RuntimeError(
            "The registered model does not have an input signature."
        )

    feature_names = signature.inputs.input_names()
    if not feature_names:
        raise RuntimeError(
            "The model input signature contains no features"
        )

    # 5. Store loaded resource in application state
    app.state.model = model
    app.state.model_name = MODEL_NAME
    app.state.model_version = str(model_version.version)
    app.state.model_alias = MODEL_ALIAS
    app.state.feature_names = feature_names

    print(
        f"Loaded {MODEL_NAME} version"
        f"{model_version.version} using alias '{MODEL_ALIAS}'"
    )

    yield


app = FastAPI(
    title="Breast Cancer Classification API",
    description="MLflow-backed model serving API",
    version="1.0.0",
    lifespan=lifespan,
)


# Health endpoint
@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model_name": request.app.state.model_name,
        "model_version": request.app.state.model_version,
        "model_alias": request.app.state.model_alias,
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    expected_features = request.app.state.feature_names
    received_features = payload.features

    # 1. Validate feature names
    missing_features = sorted(
        set(expected_features) - set(received_features)
    )

    extra_features = sorted(
        set(received_features) - set(expected_features)
    )

    if missing_features or extra_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid feature set.",
                "missing_features": missing_features,
                "extra_features": extra_features,
            },
        )

    # 2. Validate numeric values
    if not all(math.isfinite(value) for value in received_features.values()):
        raise HTTPException(
            status_code=422,
            detail="Feature values must all be finite.",
        )

    # 3. Build DataFrame in the exact expected column order
    input_df = pd.DataFrame([received_features], columns=expected_features)

    # 4. Run inference
    predictions = request.app.state.model.predict(input_df)
    prediction = int(predictions[0])

    # 5. Map numeric class to human-readable label
    class_label = {0: "benign", 1: "malignant"}

    return PredictionResponse(
        prediction=prediction,
        class_label=class_label[prediction],
        model_name=request.app.state.model_name,
        model_version=request.app.state.model_version,
        model_alias=request.app.state.model_alias,
    )
