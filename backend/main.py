from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.predictor import AnxietyPredictor

app = FastAPI(
    title="Exam Anxiety Detector API",
    description="API for detecting exam anxiety levels using a fine-tuned BERT model.",
    version="1.0.0"
)

# Initialize predictor globally
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = AnxietyPredictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model. Did you run the training script first? Error: {e}")

class PredictionRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I feel very nervous about my exams"
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    anxiety_level: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Exam Anxiety Detector API. Use POST /predict to analyze text."}

@app.post("/predict", response_model=PredictionResponse)
def predict_anxiety(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
        
    try:
        result = predictor.predict(request.text)
        return PredictionResponse(
            anxiety_level=result["anxiety_level"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
