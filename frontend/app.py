import streamlit as st
import requests

# Constants
API_URL = "http://localhost:8000/predict"

# Configuration
st.set_page_config(
    page_title="AI Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered"
)

# UI Elements Mapping
ANXIETY_MAPPING = {
    "Low Anxiety": {
        "color": "green",
        "emoji": "🟢 😊",
        "message": "You seem to be handling things well. Keep up the good work and maintain a balanced study routine!",
        "tips": [
            "Take short breaks to rest your mind.",
            "Stay hydrated and maintain a healthy sleep schedule.",
            "Review your material confidently—you've got this!"
        ]
    },
    "Moderate Anxiety": {
        "color": "orange",
        "emoji": "🟡 😟",
        "message": "You are experiencing some stress. It's completely normal before exams, but make sure to take care of yourself.",
        "tips": [
            "Try the Pomodoro technique: study for 25 minutes, then take a 5-minute break.",
            "Practice deep breathing exercises when you feel overwhelmed.",
            "Reach out to a study buddy to review materials together."
        ]
    },
    "High Anxiety": {
        "color": "red",
        "emoji": "🔴 😰",
        "message": "Your text indicates a high level of anxiety or distress. Please remember that your mental health is more important than any exam.",
        "tips": [
            "Stop studying for now and step away from your desk.",
            "Talk to a friend, family member, or counselor about how you are feeling.",
            "Consider reaching out to professional mental health resources or a student hotline if you feel overwhelmed."
        ]
    }
}

st.title("🧠 AI Based Exam Anxiety Detector")
st.markdown("Enter your thoughts or feelings about your upcoming exams, and our AI will analyze your anxiety level and provide personalized tips.")

with st.form("anxiety_form"):
    user_input = st.text_area("How are you feeling about your exams?", height=150, placeholder="E.g., I'm feeling really stressed because I have so much to study and so little time...")
    submit_button = st.form_submit_button("Analyze My Anxiety")

if submit_button:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing your text using BERT..."):
            try:
                response = requests.post(API_URL, json={"text": user_input})
                
                if response.status_code == 200:
                    data = response.json()
                    level = data.get("anxiety_level", "Unknown")
                    confidence = data.get("confidence", 0.0)
                    
                    if level in ANXIETY_MAPPING:
                        mapping = ANXIETY_MAPPING[level]
                        st.markdown("---")
                        st.subheader("Analysis Result")
                        
                        # Display level with color and emoji
                        st.markdown(f"### {mapping['emoji']} <span style='color:{mapping['color']}'>**{level}**</span>", unsafe_allow_html=True)
                        st.caption(f"Confidence: {confidence:.2f}")
                        
                        st.info(mapping['message'])
                        
                        st.subheader("💡 Personalized Tips")
                        for tip in mapping['tips']:
                            st.markdown(f"- {tip}")
                            
                    else:
                        st.error(f"Unknown anxiety level returned: {level}")
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend API. Please ensure the FastAPI server is running strongly on localhost:8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
