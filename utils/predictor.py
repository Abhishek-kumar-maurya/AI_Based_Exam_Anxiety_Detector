"""
utils/predictor.py
──────────────────
Loads the fine-tuned BERT model and exposes a predict() method.

Model resolution order
1. Local directory  (MODEL_DIR env-var or the default model/ path)
2. Hugging Face Hub (HF_MODEL_REPO env-var)   — used on Render / CI

Set the following environment variables on your Render backend service:
  MODEL_DIR      – (optional) absolute path to model dir inside the container
  HF_MODEL_REPO  – HF repo id, e.g. "your-username/exam-anxiety-bert"
"""

import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


class AnxietyPredictor:
    """Wrapper around a fine-tuned BERT classifier for exam-anxiety detection."""

    INT_MAP = {
        0: "Low Anxiety",
        1: "Moderate Anxiety",
        2: "High Anxiety",
    }

    def __init__(self, model_dir: str | None = None):
        # ── Resolve model source ──────────────────────────────────────────────
        if model_dir is None:
            model_dir = os.environ.get(
                "MODEL_DIR",
                os.path.join(os.path.dirname(__file__), "..", "model", "bert_anxiety_model"),
            )

        hf_repo = os.environ.get("HF_MODEL_REPO", "")

        local_model_exists = os.path.isdir(model_dir) and any(
            fname.endswith((".safetensors", ".bin"))
            for fname in os.listdir(model_dir)
            if os.path.isdir(model_dir)
        )

        if local_model_exists:
            source = model_dir
            print(f"Loading model from local path: {source}")
        elif hf_repo:
            source = hf_repo
            print(f"Local model not found. Loading from Hugging Face Hub: {source}")
        else:
            raise FileNotFoundError(
                f"No model found at '{model_dir}' and HF_MODEL_REPO is not set.\n"
                "Either run `python training/train_model.py` or set HF_MODEL_REPO."
            )

        # ── Load tokenizer & model ────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(source)
        self.model = BertForSequenceClassification.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Return anxiety_level (str) and confidence (float) for the given text."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)

        return {
            "anxiety_level": self.INT_MAP.get(predicted_class.item(), "Unknown"),
            "confidence": round(confidence.item(), 4),
        }
