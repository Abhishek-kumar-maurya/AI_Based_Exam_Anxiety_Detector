"""
frontend/app.py
───────────────
Streamlit UI — sends text to the FastAPI backend and displays the result.
Set the API_URL environment variable to point at your deployed backend.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

ANXIETY_META = {
    "No Anxiety": {
        "color": "#22c55e",
        "bg": "#f0fdf4",
        "badge": "NO ANXIETY",
        "emoji": "✅",
        "headline": "You're doing well!",
        "tips": [
            "Keep up your current routine — it's working.",
            "Stay hydrated and get enough sleep before the exam.",
            "A quick review of key topics will boost your confidence further.",
        ],
    },
    "Low Anxiety": {
        "color": "#3b82f6",
        "bg": "#eff6ff",
        "badge": "LOW ANXIETY",
        "emoji": "🔵",
        "headline": "A little nervous — that's normal.",
        "tips": [
            "Light nervousness before an exam actually sharpens focus.",
            "Take short 5-minute breaks every hour while studying.",
            "Practice a few past papers to build confidence.",
        ],
    },
    "Moderate Anxiety": {
        "color": "#f59e0b",
        "bg": "#fffbeb",
        "badge": "MODERATE ANXIETY",
        "emoji": "🟡",
        "headline": "You're feeling stressed — let's address it.",
        "tips": [
            "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
            "Break your study plan into smaller, daily goals.",
            "Talk to a friend or classmate — sharing helps more than you think.",
        ],
    },
    "High Anxiety": {
        "color": "#ef4444",
        "bg": "#fef2f2",
        "badge": "HIGH ANXIETY",
        "emoji": "🔴",
        "headline": "You're under serious pressure — please seek support.",
        "tips": [
            "Step away from studying right now and take a proper break.",
            "Talk to a counselor, teacher, or someone you fully trust.",
            "Contact a student mental health helpline — you deserve support.",
        ],
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        border-left: 5px solid;
    }
    .badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .headline {
        font-size: 22px;
        font-weight: 700;
        margin: 4px 0 12px 0;
    }
    .reason-box {
        background: rgba(0,0,0,0.04);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 15px;
        line-height: 1.6;
        margin-top: 10px;
    }
    .tip-item {
        padding: 6px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 AI Exam Anxiety Detector")
st.markdown(
    "Describe how you feel about your upcoming exams in your own words. "
    "The AI will logically analyse your message and explain its reasoning."
)
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("anxiety_form"):
    user_input = st.text_area(
        "How are you feeling right now?",
        height=160,
        placeholder=(
            "E.g. I have finals next week and I can't sleep. "
            "Every time I open my textbook I feel my heart racing...\n\n"
            "Write naturally — the more you share, the better the analysis."
        ),
    )
    submitted = st.form_submit_button("Analyse", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if submitted:
    text = user_input.strip()

    if not text:
        st.warning("Please write something before analysing.")
    elif len(text.split()) < 4:
        st.warning("Please write at least a sentence so the AI can analyse properly.")
    else:
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(API_URL, json={"text": text}, timeout=30)

                if resp.status_code == 200:
                    data       = resp.json()
                    level      = data.get("anxiety_level", "No Anxiety")
                    confidence = data.get("confidence", 0.0)
                    reason     = data.get("reason", "")
                    meta       = ANXIETY_META.get(level, ANXIETY_META["No Anxiety"])

                    # ── Result card ──────────────────────────────────────────
                    st.markdown(
                        f"""
                        <div class="result-card" style="background:{meta['bg']};border-color:{meta['color']}">
                            <span class="badge" style="background:{meta['color']};color:white">
                                {meta['emoji']} {meta['badge']}
                            </span>
                            <div class="headline" style="color:{meta['color']}">{meta['headline']}</div>
                            <div class="reason-box">{reason}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ── Confidence bar ───────────────────────────────────────
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption("Model confidence")
                        st.progress(confidence)
                    with col2:
                        st.metric(label="", value=f"{confidence:.0%}")

                    # ── Tips ─────────────────────────────────────────────────
                    st.markdown("#### 💡 What you can do")
                    for tip in meta["tips"]:
                        st.markdown(f"- {tip}")

                    # ── Crisis notice ─────────────────────────────────────────
                    if level == "High Anxiety":
                        st.error(
                            "**If you are having thoughts of harming yourself, please contact "
                            "a mental health helpline immediately.** "
                            "iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345"
                        )

                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the backend API.\n\nCurrent `API_URL`: `{API_URL}`\n\n"
                    "Make sure the FastAPI backend is running: `uvicorn backend.main:app`"
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out — the server may be cold-starting. Please try again.")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
