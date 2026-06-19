from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.predictor import AnxietyPredictor

# ---------------------------------------------------------------------------
# Application lifespan — loads model once on startup
# ---------------------------------------------------------------------------
predictor: AnxietyPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the BERT model when the server starts up."""
    global predictor
    model_dir = os.environ.get(
        "MODEL_DIR",
        os.path.join(os.path.dirname(__file__), '..', 'model', 'bert_anxiety_model')
    )
    try:
        predictor = AnxietyPredictor(model_dir=model_dir)
        print(f"✅ Model loaded from: {model_dir}")
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        print("   Hint: run `python training/train_model.py` first.")
    yield  # server is running
    predictor = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Exam Anxiety Detector API",
    description=(
        "REST API for detecting exam anxiety levels (Low / Moderate / High) "
        "using a fine-tuned BERT model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Allow the Streamlit frontend (and any other origin) to call this API.
# In production you can lock this down to the specific Render URL.
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "ok",
        "message": "Exam Anxiety Detector API is running. POST /predict to analyse text.",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Render uses this endpoint to verify the service is healthy."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_anxiety(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model first or check startup logs.",
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        result = predictor.predict(text)
        return PredictionResponse(
            anxiety_level=result["anxiety_level"],
            confidence=result["confidence"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")
