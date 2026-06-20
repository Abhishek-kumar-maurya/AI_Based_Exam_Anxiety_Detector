"""
training/train_model.py
────────────────────────
Fine-tunes BERT on the anxiety dataset and saves the model locally.

Run from the project root:
    pip install -r requirements-dev.txt
    python training/train_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.preprocess import TextPreprocessor, load_and_preprocess_data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AnxietyDataset(Dataset):
    def __init__(self, texts: list, labels: list, preprocessor: TextPreprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.preprocessor.preprocess(self.texts[idx])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model():
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "anxiety_dataset.csv")
    print("Loading data…")
    df = load_and_preprocess_data(dataset_path)

    texts, labels = df["text"].tolist(), df["target"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    preprocessor = TextPreprocessor()
    train_loader = DataLoader(AnxietyDataset(train_texts, train_labels, preprocessor), batch_size=4, shuffle=True)
    val_loader   = DataLoader(AnxietyDataset(val_texts,   val_labels,   preprocessor), batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initialising model…")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Training…")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("Evaluating…")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    print("\n── Evaluation Metrics ──────────────────────────────────────────────")
    print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(__file__), "..", "model", "bert_anxiety_model")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    preprocessor.tokenizer.save_pretrained(save_path)
    print(f"\n✅ Model saved to: {save_path}")
    print("Upload to HF Hub with:  python scripts/upload_model_to_hub.py --repo <id>")


if __name__ == "__main__":
    train_model()
