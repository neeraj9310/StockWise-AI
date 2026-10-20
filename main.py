
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# 1. CONFIGURATION AND LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
FEATURES_PATH = BASE_DIR / "feature_columns.json"


# ==========================================
# 2. LOAD TRAINED MODEL AND FILES
# ==========================================

def load_artifacts():
    """Load the trained model, label encoder and feature list."""

    for path in [MODEL_PATH, ENCODER_PATH, FEATURES_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required deployment file not found: {path}"
            )

    trained_model = joblib.load(MODEL_PATH)
    trained_encoder = joblib.load(ENCODER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    if not isinstance(features, list) or not features:
        raise ValueError(
            "feature_columns.json must contain a non-empty list."
        )

    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature names found.")

    if not hasattr(trained_model, "predict"):
        raise TypeError("The loaded model does not support predict().")

    if not hasattr(trained_encoder, "inverse_transform"):
        raise TypeError("The label encoder is invalid.")

    # Check model feature names when available.
    model_features = getattr(
        trained_model, "feature_names_in_", None
    )

    if model_features is not None:
        if list(model_features) != features:
            raise ValueError(
                "Feature names/order in feature_columns.json "
                "do not match the trained model."
            )

    # Check feature count when available.
    expected_count = getattr(
        trained_model, "n_features_in_", None
    )

    if expected_count is not None and expected_count != len(features):
        raise ValueError(
            f"Model expects {expected_count} features, "
            f"but feature_columns.json contains {len(features)}."
        )

    logger.info("Model loaded: %s", type(trained_model).__name__)
    logger.info("Number of input features: %s", len(features))

    return trained_model, trained_encoder, features


model, label_encoder, feature_columns = load_artifacts()

class_names = label_encoder.classes_.tolist()


# ==========================================
# 3. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="TCS Stock Prediction API",
    description=(
        "Classifies supplied TCS stock-market features "
        "into BUY, HOLD, or SELL."
    ),
    version="1.0.0"
)

# Configure your frontend domain before production deployment.
# For local development, this permits requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==========================================
# 4. INPUT VALIDATION
# ==========================================

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Open: float = Field(gt=0, allow_inf_nan=False)
    High: float = Field(gt=0, allow_inf_nan=False)
    Low: float = Field(gt=0, allow_inf_nan=False)
    Close: float = Field(gt=0, allow_inf_nan=False)
    Volume: float = Field(ge=0, allow_inf_nan=False)

    Daily_Return: float = Field(allow_inf_nan=False)
    Open_Close_Diff: float = Field(allow_inf_nan=False)
    High_Low_Diff: float = Field(allow_inf_nan=False)

    MA_10: float = Field(gt=0, allow_inf_nan=False)
    MA_20: float = Field(gt=0, allow_inf_nan=False)
    MA_50: float = Field(gt=0, allow_inf_nan=False)

    RSI: float = Field(ge=0, le=100, allow_inf_nan=False)
    MACD: float = Field(allow_inf_nan=False)
    MACD_Signal: float = Field(allow_inf_nan=False)

    BB_Upper: float = Field(allow_inf_nan=False)
    BB_Middle: float = Field(allow_inf_nan=False)
    BB_Lower: float = Field(allow_inf_nan=False)

    Volume_Change: float = Field(allow_inf_nan=False)
    Volume_MA: float = Field(ge=0, allow_inf_nan=False)

    Close_Lag_1: float = Field(gt=0, allow_inf_nan=False)
    Close_Lag_5: float = Field(gt=0, allow_inf_nan=False)
    Close_Lag_10: float = Field(gt=0, allow_inf_nan=False)
    Close_Lag_20: float = Field(gt=0, allow_inf_nan=False)

    NIFTY_Close: float = Field(gt=0, allow_inf_nan=False)
    NIFTY_Return: float = Field(allow_inf_nan=False)
    NIFTY_Volatility: float = Field(ge=0, allow_inf_nan=False)

    CPI_Inflation: float = Field(allow_inf_nan=False)
    GDP_Growth: float = Field(allow_inf_nan=False)


# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "TCS Stock Prediction API is running",
        "model_type": type(model).__name__,
        "classes": [str(c) for c in class_names],
        "features_required": len(feature_columns),
        "docs": "/docs",
        "health": "/health"
    }


# ==========================================
# 6. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns)
    }


# ==========================================
# 7. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: PredictionInput):

    try:
        input_data = data.model_dump()

        # Ensure every model feature is supplied.
        missing_features = [
            col for col in feature_columns
            if col not in input_data
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing required features",
                    "features": missing_features
                }
            )

        # Keep exactly the feature order used during training.
        input_df = pd.DataFrame(
            [[input_data[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Prevent invalid numeric values.
        if not np.isfinite(input_df.to_numpy(dtype=float)).all():
            raise HTTPException(
                status_code=422,
                detail="All feature values must be finite numbers."
            )

        # Predict the encoded class.
        prediction = model.predict(input_df)[0]

        # Convert encoded prediction to its original label.
        predicted_label = label_encoder.inverse_transform(
            np.asarray([prediction])
        )[0]

        result = {
            "prediction": str(predicted_label),
            "model": type(model).__name__,
            "features_used": len(feature_columns),
            "note": (
                "This is a model classification, not a guarantee "
                "of future stock performance."
            )
        }

        # Include probabilities only when supported by the model.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            model_classes = getattr(
                model, "classes_", None
            )

            if model_classes is not None:
                result["probabilities"] = {
                    str(label_encoder.inverse_transform(
                        np.asarray([class_id])
                    )[0]): round(float(probability), 6)
                    for class_id, probability in zip(
                        model_classes, probabilities
                    )
                }

        return result

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs."
        ) from error
