"""
preprocessing/preprocess.py
────────────────────────────
Data loading and tokenization utilities used during training only.
Not imported by the backend or frontend at runtime.
"""

import pandas as pd
from transformers import BertTokenizer

# Map raw dataset labels → anxiety levels → integer class ids
LABEL_MAP = {
    "Normal": "Low Anxiety",
    "Anxiety": "Moderate Anxiety",
    "Stress": "Moderate Anxiety",
    "Depression": "High Anxiety",
    "Suicidal": "High Anxiety",
}

INT_MAP = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV dataset and map labels to integer class ids."""
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["anxiety_level"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["anxiety_level"])
    df["target"] = df["anxiety_level"].map(INT_MAP)
    return df


class TextPreprocessor:
    """Tokenizes text using the BERT tokenizer for training."""

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def _encode(self, texts):
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def preprocess(self, text: str):
        """Tokenize a single string."""
        return self._encode(str(text))

    def preprocess_batch(self, texts: list):
        """Tokenize a list of strings."""
        return self._encode([str(t) for t in texts])
