import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration — the backend URL is injected via an environment variable so
# it works both locally (localhost) and on Render (the deployed API URL).
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered",
)

# ── Anxiety level metadata ────────────────────────────────────────────────────
ANXIETY_MAPPING = {
    "Low Anxiety": {
        "color": "green",
        "emoji": "🟢 😊",
        "message": (
            "You seem to be handling things well. "
            "Keep up the good work and maintain a balanced study routine!"
        ),
        "tips": [
            "Take short breaks to rest your mind.",
            "Stay hydrated and maintain a healthy sleep schedule.",
            "Review your material confidently — you've got this!",
        ],
    },
    "Moderate Anxiety": {
        "color": "orange",
        "emoji": "🟡 😟",
        "message": (
            "You are experiencing some stress. It's completely normal before exams, "
            "but make sure to take care of yourself."
        ),
        "tips": [
            "Try the Pomodoro technique: study for 25 minutes, then take a 5-minute break.",
            "Practice deep breathing exercises when you feel overwhelmed.",
            "Reach out to a study buddy to review materials together.",
        ],
    },
    "High Anxiety": {
        "color": "red",
        "emoji": "🔴 😰",
        "message": (
            "Your text indicates a high level of anxiety or distress. "
            "Please remember that your mental health is more important than any exam."
        ),
        "tips": [
            "Stop studying for now and step away from your desk.",
            "Talk to a friend, family member, or counselor about how you are feeling.",
            "Consider reaching out to professional mental health resources or a student hotline.",
        ],
    },
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 AI Based Exam Anxiety Detector")
st.markdown(
    "Enter your thoughts or feelings about your upcoming exams, and our AI will "
    "analyse your anxiety level and provide personalised tips."
)

with st.form("anxiety_form"):
    user_input = st.text_area(
        "How are you feeling about your exams?",
        height=150,
        placeholder=(
            "E.g., I'm feeling really stressed because I have so much to study "
            "and so little time..."
        ),
    )
    submit_button = st.form_submit_button("Analyse My Anxiety")

if submit_button:
    if not user_input.strip():
        st.warning("Please enter some text to analyse.")
    else:
        with st.spinner("Analysing your text using BERT…"):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": user_input},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    level = data.get("anxiety_level", "Unknown")
                    confidence = data.get("confidence", 0.0)

                    if level in ANXIETY_MAPPING:
                        mapping = ANXIETY_MAPPING[level]
                        st.markdown("---")
                        st.subheader("Analysis Result")

                        st.markdown(
                            f"### {mapping['emoji']} "
                            f"<span style='color:{mapping['color']}'>"
                            f"**{level}**</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Confidence: {confidence:.2%}")

                        st.info(mapping["message"])

                        st.subheader("💡 Personalised Tips")
                        for tip in mapping["tips"]:
                            st.markdown(f"- {tip}")
                    else:
                        st.error(f"Unknown anxiety level returned: {level}")

                else:
                    st.error(f"API error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "⚠️ Could not reach the backend API. "
                    f"Make sure the FastAPI server is running and `API_URL` is set correctly.\n\n"
                    f"Current `API_URL`: `{API_URL}`"
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ The request timed out. The server may be cold-starting — please try again.")
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")
