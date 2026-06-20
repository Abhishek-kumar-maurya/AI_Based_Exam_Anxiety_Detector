# AI Based Exam Anxiety Detector

An NLP full-stack app that classifies student text into **Low / Moderate / High** anxiety using a fine-tuned BERT model, a FastAPI backend, and a Streamlit frontend.

## Project Structure

```
AI_Based_Exam_Anxiety_Detector/
├── backend/
│   └── main.py                  # FastAPI — /predict endpoint
├── frontend/
│   └── app.py                   # Streamlit UI
├── utils/
│   └── predictor.py             # BERT inference wrapper (used by backend)
├── preprocessing/
│   └── preprocess.py            # Data loading & tokenization (training only)
├── training/
│   └── train_model.py           # Fine-tune BERT and save weights locally
├── scripts/
│   └── upload_model_to_hub.py   # Push trained weights to Hugging Face Hub
├── dataset/
│   └── anxiety_dataset.csv      # Training data (text + label)
├── model/
│   └── bert_anxiety_model/      # Saved weights (git-ignored, stored on HF Hub)
├── requirements.txt             # Runtime dependencies (Render)
├── requirements-dev.txt         # Training dependencies (local only)
├── render.yaml                  # Render deploy config
└── start.sh                     # Startup script (backend + frontend)
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Runtime only:
pip install -r requirements.txt

# For training too:
pip install -r requirements-dev.txt
```

### 2. Train the model

```bash
python training/train_model.py
```

Outputs accuracy / F1 metrics and saves the model to `model/bert_anxiety_model/`.

### 3. Run locally

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload

# Terminal 2 — frontend
streamlit run frontend/app.py
```

- API docs: <http://localhost:8000/docs>
- UI: <http://localhost:8501>

---

## Deploy to Render

### 1. Upload model to Hugging Face Hub

```bash
huggingface-cli login
python scripts/upload_model_to_hub.py --repo YOUR_USERNAME/exam-anxiety-bert
```

### 2. Set environment variables in the Render dashboard

| Variable | Value |
|---|---|
| `HF_MODEL_REPO` | `YOUR_USERNAME/exam-anxiety-bert` |
| `HUGGING_FACE_HUB_TOKEN` | *(only if repo is private)* |
| `API_URL` | `http://localhost:8000/predict` *(already default)* |

### 3. Push to Git and deploy

Render picks up `render.yaml` automatically. The free tier runs both backend and frontend in a single dyno via `start.sh`.
