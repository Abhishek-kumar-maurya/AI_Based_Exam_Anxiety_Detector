"""
backend/main.py
───────────────
FastAPI application — exposes /predict for exam-anxiety classification.
The BERT model is loaded once at startup via the AnxietyPredictor utility.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.predictor import AnxietyPredictor


# ---------------------------------------------------------------------------
# Lifespan — load model once on startup, release on shutdown
# ---------------------------------------------------------------------------
predictor: Optional[AnxietyPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    model_dir = os.environ.get(
        "MODEL_DIR",
        os.path.join(os.path.dirname(__file__), "..", "model", "bert_anxiety_model"),
    )
    try:
        predictor = AnxietyPredictor(model_dir=model_dir)
        print(f"[OK] Model loaded from: {model_dir}")
    except Exception as exc:
        print(f"[ERROR] Model load failed: {exc}")
    yield
    predictor = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Exam Anxiety Detector API",
    description="Classifies exam anxiety levels (Low / Moderate / High) using fine-tuned BERT.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "I feel very nervous about my exams"}]
        }
    }


class PredictionResponse(BaseModel):
    anxiety_level: str
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Exam Anxiety Detector API — POST /predict to analyse text."}


@app.get("/health", tags=["Health"])
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        result = predictor.predict(text)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
