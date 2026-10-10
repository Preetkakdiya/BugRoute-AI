"""
BugRoute AI - Production REST API Service
Exposes HTTP POST /predict endpoint for JIRA / GitHub Webhooks.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import BugRoutePredictor

app = FastAPI(
    title="BugRoute AI REST API",
    description="Automated Bug Report Component Routing & Severity Service",
    version="1.0.0"
)

predictor = BugRoutePredictor()


class BugReportRequest(BaseModel):
    title: str
    description: str
    model: str = "CodeBERT"


class PredictionResponse(BaseModel):
    model: str
    predicted_component: str
    confidence: float
    top_3: list
    important_words: list


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "BugRoute AI Engine"}


@app.post("/predict", response_model=PredictionResponse)
def predict_bug_route(request: BugReportRequest):
    if not request.title.strip() and not request.description.strip():
        raise HTTPException(status_code=400, detail="Title or description must be provided.")
    
    result = predictor.predict(request.title, request.description, request.model)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
