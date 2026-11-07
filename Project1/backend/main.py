
import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =====================================
# 1. PROJECT PATH
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# =====================================
# 2. LOAD MODEL AND FILES
# =====================================

try:
    model = joblib.load(MODEL_DIR / "model.joblib")

    with open(MODEL_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    with open(MODEL_DIR / "target_labels.json", "r") as f:
        target_labels = json.load(f)

    with open(MODEL_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    sample_df = pd.read_csv(MODEL_DIR / "feature_examples.csv")

except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


# =====================================
# 3. FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="GitHub Repo Classifier API",
    description="Random Forest ML Prediction System for GitHub Repositories",
    version="1.0.0"
)

# CORS — allow all origins so the Streamlit frontend can reach this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# 4. REQUEST SCHEMA
# =====================================

class PredictionRequest(BaseModel):
    features: Dict[str, Any]


# =====================================
# 5. HOME API
# =====================================

@app.get("/")
def home():
    return {
        "message": "GitHub Repo Classifier API is running!",
        "model": "Random Forest Classifier",
        "status": "success",
        "docs": "/docs"
    }


# =====================================
# 6. HEALTH CHECK
# =====================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# =====================================
# 7. GET FEATURES
# =====================================

@app.get("/features")
def get_features():
    return {
        "total_features": len(feature_names),
        "features": feature_names
    }


# =====================================
# 8. GET SAMPLE DATA
# =====================================

@app.get("/sample")
def get_sample():
    if sample_df.empty:
        raise HTTPException(status_code=500, detail="Sample dataset is empty")

    sample = sample_df.iloc[0].to_dict()
    sample = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in sample.items()
    }
    return {"sample": sample}


# =====================================
# 9. MODEL METRICS
# =====================================

@app.get("/metrics")
def get_metrics():
    return metrics


# =====================================
# 10. PREDICTION API
# =====================================

@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        input_data = data.features

        missing_features = [
            col for col in feature_names if col not in input_data
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing features",
                    "missing": missing_features
                }
            )

        ordered_data = {col: input_data[col] for col in feature_names}
        input_df = pd.DataFrame([ordered_data], columns=feature_names)

        prediction = model.predict(input_df)[0]
        predicted_label = str(prediction)

        try:
            encoded_class = int(prediction)
            if (
                str(encoded_class) == str(prediction)
                and 0 <= encoded_class < len(target_labels)
            ):
                predicted_label = target_labels[encoded_class]
        except (ValueError, TypeError):
            pass

        probabilities = None
        if hasattr(model, "predict_proba"):
            probability_values = model.predict_proba(input_df)[0]
            probabilities = {}
            for class_value, probability in zip(model.classes_, probability_values):
                try:
                    class_index = int(class_value)
                    if (
                        str(class_index) == str(class_value)
                        and 0 <= class_index < len(target_labels)
                    ):
                        class_name = target_labels[class_index]
                    else:
                        class_name = str(class_value)
                except (ValueError, TypeError):
                    class_name = str(class_value)

                probabilities[class_name] = round(float(probability) * 100, 2)

        return {
            "status": "success",
            "prediction": str(prediction),
            "predicted_label": str(predicted_label),
            "probabilities": probabilities
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
