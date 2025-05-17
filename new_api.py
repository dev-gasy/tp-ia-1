#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple API for testing
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


# Define the models
class ClaimData(BaseModel):
    Age: Optional[int] = None
    Sexe: Optional[str] = None
    Code_Emploi: Optional[int] = None


class PredictionRequest(BaseModel):
    data: ClaimData


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: Optional[float] = None
    employee_assignment: Optional[str] = None


# Create a new FastAPI app
app = FastAPI(title="Test API")


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/model_info")
async def model_info():
    return {
        "model_type": "RandomForest",
        "classifier_type": "RandomForestClassifier",
        "description": "Classification model for predicting claim duration"
    }


@app.get("/model/info")
async def model_detail_info():
    return {
        "model_version": "1.0.0",
        "training_date": "2023-04-15",
        "model_type": "RandomForest",
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80
        }
    }


@app.get("/model/feature_importance")
async def feature_importance():
    return {
        "feature_importance": [
            {"feature": "Age", "importance": 0.2},
            {"feature": "Sexe", "importance": 0.05},
            {"feature": "Code_Emploi", "importance": 0.15},
            {"feature": "Salaire_Annuel", "importance": 0.1},
            {"feature": "Description_Invalidite", "importance": 0.5}
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return {
        "prediction": 1,
        "prediction_label": "Long (> 180 days)",
        "probability": 0.8,
        "employee_assignment": "Experienced employee"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
