"""
api.py — FastAPI REST Backend
==============================
Exposes the Sentiment Analysis Engine as a production HTTP API.

Endpoints:
  POST /predict          → Single review prediction
  POST /predict/batch    → Batch prediction
  GET  /health           → Health check

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Example curl:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"review": "This movie was fantastic!"}'
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from src.predict import SentimentPredictor
from src.utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Analysis Engine API",
    description="NLP-powered movie review sentiment classifier using classical ML",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor once at startup
_predictor: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


# ──────────────────────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    review: str = Field(..., min_length=1, description="Movie review text")
    model: str  = Field("logistic_regression", description="Model to use for prediction")


class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of movie review texts")
    model:   str       = Field("logistic_regression")


class PredictionResponse(BaseModel):
    review:        str
    prediction:    str
    confidence:    float
    confidence_pct: str
    probabilities: dict = {}
    model:         str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        predictor = get_predictor()
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: ReviewRequest):
    """
    Predict sentiment for a single movie review.

    Returns prediction label, confidence score, and class probabilities.
    """
    try:
        predictor = get_predictor()
        result    = predictor.predict(request.review)

        if result.get("sentiment") in ("ERROR", "UNCERTAIN"):
            raise HTTPException(
                status_code=422,
                detail=result.get("error", "Prediction failed")
            )

        return PredictionResponse(
            review        = result["review"],
            prediction    = result["sentiment"].lower(),
            confidence    = result["confidence"],
            confidence_pct = result["confidence_pct"],
            probabilities = result.get("probabilities", {}),
            model         = result["model"],
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found. Train first: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchReviewRequest):
    """
    Predict sentiment for a batch of movie reviews.

    More efficient than calling /predict in a loop.
    """
    if not request.reviews:
        raise HTTPException(status_code=422, detail="reviews list cannot be empty")
    if len(request.reviews) > 500:
        raise HTTPException(status_code=422, detail="Maximum 500 reviews per batch")

    try:
        predictor = get_predictor()
        results   = predictor.predict_batch(request.reviews)

        return {
            "count": len(results),
            "predictions": [
                {
                    "review":     r["review"],
                    "prediction": r["sentiment"].lower(),
                    "confidence": r["confidence"],
                }
                for r in results
            ],
            "summary": {
                "positive": sum(1 for r in results if r.get("sentiment") == "POSITIVE"),
                "negative": sum(1 for r in results if r.get("sentiment") == "NEGATIVE"),
            }
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
