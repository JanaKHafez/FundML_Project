"""
Streamlit Frontend for Hate Speech Detection
Connects to Flask backend API for predictions and feedback submission.
Created with the help of Github Copilot.
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Backend API configuration
API_URL = "http://localhost:5000"

# Label mapping
target_labels = {
    1: "Non-Hate",
    0: "Hate"
}

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Session State for storing prediction results
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .hate-speech {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .non-hate-speech {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .feedback-section {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_prediction(text):
    """Get prediction from backend API."""
    print(f"Requesting prediction for text: {text}")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def submit_feedback(text, predicted_label, true_label):
    """Submit feedback to backend API."""
    print(f"Submitting feedback: text='{text}', predicted_label={predicted_label}, true_label={true_label}")
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "text": text,
                "predicted_label": predicted_label,
                "true_label": true_label
            },
            timeout=10
        )
        print(f"Feedback submission response: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return False

# Main UI
st.markdown('<div class="main-header">🛡️ Hate Speech Detection System</div>', unsafe_allow_html=True)

# Check API health
api_healthy = check_api_health()
if not api_healthy:
    st.error("⚠️ Backend API is not running! Please start the Flask server first.")
    st.code("python backend_api.py", language="bash")
    st.stop()

# Main content area
st.header("🔍 Analyze Text")
st.write("Enter text below to check if it contains hate speech.")

# Text input
text_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type or paste text here...",
    key="text_input"
)

col1, col2, col3 = st.columns([1, 1, 3])

def clear_state():
    st.session_state['prediction_result'] = None
    st.session_state["text_input"] = ""

with col1:
    analyze_button = st.button("🔎 Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_state
    )

if analyze_button and text_input:
    with st.spinner("Analyzing..."):
        result = get_prediction(text_input)
        st.session_state['prediction_result'] = result

elif analyze_button and not text_input:
    st.warning("⚠️ Please enter some text to analyze.")

if st.session_state['prediction_result']:
    result = st.session_state['prediction_result']

    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        label_int = result["prediction"]     # 0 or 1
        label_str = target_labels[label_int] # "Hate" or "Non-Hate"

        # Confidence
        confidence = result["confidence"] * 100

        # Pick CSS class and icon
        if label_int == 0:  # Hate
            prediction_class = "hate-speech"
            icon_label = f"⚠️ {target_labels[0]}"
        else:  # Non-Hate
            prediction_class = "non-hate-speech"
            icon_label = f"✅ {target_labels[1]}"

        st.markdown(
            f"""
            <div class="prediction-box {prediction_class}">
                <h3>{icon_label}</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hate Probability", f"{result['probability_hate']*100:.2f}%")
        with col2:
            st.metric("Non-Hate Probability", f"{result['probability_non_hate']*100:.2f}%")

        # Feedback section
        st.markdown("---")
        st.subheader("📝 Provide Feedback")

        fb_col1, fb_col2 = st.columns(2)

        # Prediction correct
        with fb_col1:
            if st.button("👍 Prediction is Correct", use_container_width=True):
                current_text = st.session_state.get('text_input', '')                
                if submit_feedback(current_text, label_int, label_int):
                    st.success("Thank you! Feedback recorded.")
                    st.session_state['prediction_result'] = None 
                    st.rerun()

        # Prediction wrong -> toggle correct label
        with fb_col2:
            if st.button("❌ Prediction is Incorrect", use_container_width=True):
                current_text = st.session_state.get('text_input', '')
                correct_label = 1 if label_int == 0 else 0                
                if submit_feedback(current_text, label_int, correct_label):
                    st.success("Thanks! Your correction is recorded.")
                    st.session_state['prediction_result'] = None
                    st.rerun()

        # Show cleaned text
        with st.expander("🔧 View Cleaned Text"):
            st.code(result.get('cleaned_text', 'N/A'))

# Examples section
st.markdown("---")
st.subheader("💡 Example Texts")
st.write("Click on an example to analyze it:")

examples = [
    "I love this community!",
    "Kill yourself.",
    "Great job on the project! Keep up the excellent work.",
    "I hate you.",
    "I hope you die."
]

def load_example(example_text):
    st.session_state['text_input'] = example_text

cols = st.columns(len(examples))
for i, (col, example) in enumerate(zip(cols, examples)):
    with col:
        st.button(
            f"Example {i+1}",
            use_container_width=True,
            on_click=load_example,
            args=(example,)
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🛡️ Hate Speech Detection System</p>
    </div>
    """,
    unsafe_allow_html=True
)