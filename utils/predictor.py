"""
utils/predictor.py
──────────────────
Loads the fine-tuned BERT classifier and exposes a predict() method.

Prediction pipeline (4 layers):
  1. Relevance gate   — is the text even about exams / mental health?
  2. Keyword scoring  — count explicit anxiety/calm signals in the words
  3. BERT inference   — what does the model think?
  4. Final decision   — combine keyword score + BERT confidence into a
                        human-readable verdict + reason

Model resolution order:
  1. Local path  — MODEL_DIR env-var  (or default model/bert_anxiety_model/)
  2. HF Hub      — HF_MODEL_REPO env-var  (used on Render)
"""

import os
import re
from typing import Optional

# Suppress tokenizer fork warnings (safe for single-process inference)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Inference only — disable global gradient tracking
torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Keyword lexicons  (all lowercase)
# ---------------------------------------------------------------------------

# Words that signal the text is about exams / studies / mental health
TOPIC_KEYWORDS = {
    "exam", "exams", "test", "tests", "quiz", "quizzes", "finals", "midterm",
    "midterms", "assignment", "assignments", "homework", "study", "studying",
    "studies", "grade", "grades", "marks", "score", "scores", "result",
    "results", "paper", "subject", "class", "classes", "college", "school",
    "university", "course", "semester", "lecture", "anxiety", "anxious",
    "stress", "stressed", "nervous", "worried", "worry", "panic", "fear",
    "scared", "depressed", "depression", "pressure", "overwhelmed",
    "exhausted", "burnout", "hopeless", "mental", "health", "feeling",
    "sleep", "cry", "crying", "sad", "unhappy",
}

# Words that clearly signal anxiety / distress
HIGH_SIGNAL_WORDS = {
    "panic", "terrified", "overwhelming", "overwhelmed", "hopeless",
    "helpless", "desperate", "suicidal", "can't cope", "breaking down",
    "falling apart", "can't take it", "want to disappear", "give up",
    "dark thoughts", "dangerous thoughts", "no way out", "end it all",
}

MODERATE_SIGNAL_WORDS = {
    "stressed", "stress", "anxious", "anxiety", "nervous", "worried",
    "worry", "pressure", "scared", "fear", "afraid", "dread", "dreading",
    "racing heart", "shaky", "sleepless", "can't sleep", "exhausted",
    "burnout", "too much", "can't concentrate", "distracted", "crying",
    "cry", "sad", "depressed", "depression", "fail", "failing", "failure",
}

LOW_SIGNAL_WORDS = {
    "little nervous", "bit nervous", "slightly worried", "minor stress",
    "some pressure", "mild anxiety", "slightly anxious",
}

# Words that signal calm / no anxiety
CALM_WORDS = {
    "fine", "good", "great", "confident", "ready", "prepared", "calm",
    "relaxed", "comfortable", "happy", "excited", "enjoying", "refreshed",
    "well", "okay", "alright", "no problem", "no issue", "understand",
    "understood", "clear", "easy",
}

# Minimum confidence from BERT to trust its label
CONFIDENCE_THRESHOLD = 0.65

# Minimum topic-relevance score to even attempt anxiety detection
# (fraction of words that must be topic-related)
RELEVANCE_MIN_FRACTION = 0.07  # ~1 in 14 words


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tokenize_lower(text: str):
    """Return lowercase word tokens from text."""
    return re.findall(r"[a-z']+", text.lower())


def _relevance_score(words: list) -> float:
    """Fraction of words that are topic-related."""
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in TOPIC_KEYWORDS)
    # Also check bigrams (e.g. "can't cope")
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    hits += sum(1 for bg in bigrams if bg in TOPIC_KEYWORDS)
    return hits / len(words)


def _keyword_anxiety_score(words: list) -> tuple:
    """
    Returns (score, matched_signals) where score is 0–3:
      0  = calm / no anxiety signals
      1  = low anxiety signals
      2  = moderate anxiety signals
      3  = high anxiety signals
    """
    text_joined = " ".join(words)
    bigrams = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
    all_tokens = set(words) | bigrams

    high_hits   = [w for w in HIGH_SIGNAL_WORDS   if w in all_tokens or w in text_joined]
    mod_hits    = [w for w in MODERATE_SIGNAL_WORDS if w in all_tokens]
    low_hits    = [w for w in LOW_SIGNAL_WORDS    if w in text_joined]
    calm_hits   = [w for w in CALM_WORDS          if w in all_tokens]

    if high_hits:
        return 3, high_hits
    if len(mod_hits) >= 2:
        return 2, mod_hits
    if len(mod_hits) == 1:
        return 1, mod_hits
    if low_hits:
        return 1, low_hits
    if calm_hits:
        return 0, calm_hits
    return -1, []   # neutral / unknown


def _build_reason(
    level: str,
    keyword_score: int,
    keyword_hits: list,
    bert_label: str,
    bert_conf: float,
    is_relevant: bool,
) -> str:
    """Build a plain-English explanation of how the verdict was reached."""

    if not is_relevant:
        return (
            "Your message does not appear to be about exams, studies, or emotional "
            "wellbeing, so no anxiety was detected. If you are experiencing exam "
            "stress, try describing how you feel about your upcoming tests or studies."
        )

    if level == "No Anxiety":
        if keyword_score == 0:
            calm = ", ".join(keyword_hits[:3]) if keyword_hits else "calm language"
            return (
                f"Your message uses positive, calm language ({calm}). "
                "There are no signs of exam-related stress or anxiety. "
                "Keep up the good mindset!"
            )
        return (
            "The AI detected some exam-related words but no strong anxiety signals. "
            f"The model's confidence was low ({bert_conf:.0%}), which suggests your "
            "text does not clearly express distress. You seem to be doing okay!"
        )

    hits_str = ", ".join(f'"{h}"' for h in keyword_hits[:4])

    if level == "Low Anxiety":
        return (
            f"Mild anxiety signals were detected in your message (words like {hits_str}). "
            "This level of pre-exam nervousness is completely normal and actually helps "
            f"with focus. The AI model agreed with {bert_conf:.0%} confidence."
        )

    if level == "Moderate Anxiety":
        return (
            f"Several anxiety-related expressions were found in your message "
            f"(e.g. {hits_str}). Combined with the AI model's assessment "
            f"({bert_conf:.0%} confidence), this suggests a noticeable level of "
            "exam stress that deserves attention. Consider taking structured breaks "
            "and talking to someone you trust."
        )

    if level == "High Anxiety":
        return (
            f"Strong distress signals were detected in your message "
            f"(e.g. {hits_str}). The AI model flagged this with {bert_conf:.0%} confidence. "
            "High anxiety can seriously impact both your performance and wellbeing. "
            "Please reach out to a counselor, friend, or mental health resource — "
            "you do not have to handle this alone."
        )

    return "Analysis complete."


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class AnxietyPredictor:
    """4-layer anxiety detector: relevance → keywords → BERT → human verdict."""

    BERT_LABELS = {0: "Low Anxiety", 1: "Moderate Anxiety", 2: "High Anxiety"}

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = os.environ.get(
                "MODEL_DIR",
                os.path.join(os.path.dirname(__file__), "..", "model", "bert_anxiety_model"),
            )

        hf_repo = os.environ.get("HF_MODEL_REPO", "")
        local_ok = os.path.isdir(model_dir) and any(
            f.endswith((".safetensors", ".bin")) for f in os.listdir(model_dir)
        )

        if local_ok:
            source = model_dir
            print(f"Loading model from local path: {source}")
        elif hf_repo:
            source = hf_repo
            print(f"Loading model from Hugging Face Hub: {source}")
        else:
            raise FileNotFoundError(
                f"No model found at '{model_dir}' and HF_MODEL_REPO is not set.\n"
                "Train locally with:  python training/train_model.py\n"
                "Then upload with:    python scripts/upload_model_to_hub.py --repo <id>"
            )

        self.device = torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(source)
        self.model = BertForSequenceClassification.from_pretrained(
            source,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Run the full 4-layer pipeline and return:
          anxiety_level : str   — "No Anxiety" | "Low" | "Moderate" | "High"
          confidence    : float — 0.0–1.0
          reason        : str   — human-readable explanation
        """
        words = _tokenize_lower(text)

        # ── Layer 1: Relevance gate ────────────────────────────────────────
        rel_score = _relevance_score(words)
        is_relevant = rel_score >= RELEVANCE_MIN_FRACTION

        # ── Layer 2: Keyword scoring ───────────────────────────────────────
        kw_score, kw_hits = _keyword_anxiety_score(words)

        # ── Layer 3: BERT inference ────────────────────────────────────────
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask=attention_mask).logits
        probs  = F.softmax(logits.float(), dim=1)   # cast to float32 for softmax stability
        bert_conf, bert_class = torch.max(probs, dim=1)
        bert_conf  = round(bert_conf.item(), 4)
        bert_label = self.BERT_LABELS.get(bert_class.item(), "Low Anxiety")

        # ── Layer 4: Final decision ────────────────────────────────────────
        #
        # Rules (in priority order):
        #  a) Not relevant + no keyword hits  → No Anxiety
        #  b) Keyword says HIGH               → High Anxiety (regardless of BERT)
        #  c) BERT confident + keyword agrees → trust BERT label
        #  d) BERT confident + keyword says calm → downgrade to No Anxiety / Low
        #  e) Low BERT confidence             → trust keywords only
        #  f) Nothing found                   → No Anxiety

        if not is_relevant and kw_score <= 0:
            final_level = "No Anxiety"

        elif kw_score == 3:
            # Explicit high-distress words override everything
            final_level = "High Anxiety"

        elif bert_conf >= CONFIDENCE_THRESHOLD:
            if kw_score == 0 and bert_label != "Low Anxiety":
                # BERT sees anxiety but keywords say calm → downgrade
                final_level = "No Anxiety"
            elif kw_score == -1 and bert_label == "Low Anxiety":
                # Neutral text + BERT mild → treat as No Anxiety
                final_level = "No Anxiety"
            else:
                final_level = bert_label

        elif kw_score == 2:
            final_level = "Moderate Anxiety"
        elif kw_score == 1:
            final_level = "Low Anxiety"
        elif kw_score == 0:
            final_level = "No Anxiety"
        else:
            # kw_score == -1 (neutral) and BERT low confidence
            final_level = "No Anxiety"

        reason = _build_reason(
            level=final_level,
            keyword_score=kw_score,
            keyword_hits=kw_hits,
            bert_label=bert_label,
            bert_conf=bert_conf,
            is_relevant=is_relevant,
        )

        return {
            "anxiety_level": final_level,
            "confidence": bert_conf,
            "reason": reason,
        }
