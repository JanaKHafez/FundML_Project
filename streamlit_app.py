"""
Streamlit Frontend for Hate Speech Detection
Connects to Flask backend API for predictions and feedback submission.
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Backend API configuration
API_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="wide"
)

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
        return response.status_code == 200
    except:
        return False

def get_stats():
    """Get feedback statistics from backend."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Main UI
st.markdown('<div class="main-header">🛡️ Hate Speech Detection System</div>', unsafe_allow_html=True)

# Check API health
api_healthy = check_api_health()
if not api_healthy:
    st.error("⚠️ Backend API is not running! Please start the Flask server first.")
    st.code("python backend_api.py", language="bash")
    st.stop()

# Sidebar for information and statistics
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application detects hate speech in text using a custom-trained neural network.
    
    **Features:**
    - Real-time hate speech detection
    - User feedback collection
    - Online learning with periodic model retraining
    """)
    
    st.header("📊 Statistics")
    stats = get_stats()
    if stats:
        st.metric("Total Feedback", stats.get('total_feedback', 0))
        st.metric("Pending for Training", stats.get('unused_feedback', 0))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hate", stats.get('hate_count', 0))
        with col2:
            st.metric("Non-Hate", stats.get('non_hate_count', 0))
    
    st.header("🔧 Model Info")
    st.write("""
    **Architecture:**
    - Input: TF-IDF vectors
    - Embedding: 512 units
    - Hidden 1: 128 units (ReLU)
    - Hidden 2: 64 units (ReLU)
    - Output: 1 unit (Sigmoid)
    
    **Training:**
    - Loss: Binary Cross-Entropy
    - Optimizer: Gradient Descent
    - Batch size: 128
    """)

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

# Analyze button
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    analyze_button = st.button("🔎 Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Process prediction
if analyze_button and text_input:
    with st.spinner("Analyzing..."):
        result = get_prediction(text_input)
    
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        # Display prediction result
        prediction_class = "hate-speech" if result['prediction'] == 1 else "non-hate-speech"
        label = result['label']
        confidence = result['confidence'] * 100
        
        st.markdown(
            f"""
            <div class="prediction-box {prediction_class}">
                <h3>{'⚠️ ' + label if result['prediction'] == 1 else '✅ ' + label}</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Hate Speech Probability",
                f"{result['probability_hate']*100:.2f}%"
            )
        with col2:
            st.metric(
                "Non-Hate Speech Probability",
                f"{result['probability_non_hate']*100:.2f}%"
            )
        
        # Feedback section
        st.markdown("---")
        st.subheader("📝 Provide Feedback")
        st.write("Help improve the model by confirming if the prediction is correct!")
        
        feedback_col1, feedback_col2 = st.columns(2)
        
        with feedback_col1:
            if st.button("✅ Prediction is Correct", use_container_width=True):
                if submit_feedback(text_input, result['prediction'], result['prediction']):
                    st.success("Thank you! Your feedback has been recorded.")
                    # Update stats
                    st.rerun()
                else:
                    st.error("Failed to submit feedback. Please try again.")
        
        with feedback_col2:
            if st.button("❌ Prediction is Incorrect", use_container_width=True):
                # Toggle the label for incorrect prediction
                correct_label = 0 if result['prediction'] == 1 else 1
                if submit_feedback(text_input, result['prediction'], correct_label):
                    st.success("Thank you! Your correction has been recorded and will be used to improve the model.")
                    st.rerun()
                else:
                    st.error("Failed to submit feedback. Please try again.")
        
        # Show cleaned text (expandable)
        with st.expander("🔧 View Preprocessed Text"):
            st.code(result.get('cleaned_text', 'N/A'))

elif analyze_button and not text_input:
    st.warning("⚠️ Please enter some text to analyze.")

# Examples section
st.markdown("---")
st.subheader("💡 Example Texts")
st.write("Click on an example to analyze it:")

examples = [
    "I love this community! Everyone is so kind and helpful.",
    "You are absolutely worthless and should disappear.",
    "Great job on the project! Keep up the excellent work.",
    "I hate people who don't agree with me.",
    "Let's work together to build a better future for everyone."
]

cols = st.columns(len(examples))
for i, (col, example) in enumerate(zip(cols, examples)):
    with col:
        if st.button(f"Example {i+1}", use_container_width=True):
            st.session_state.text_input = example
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🛡️ Hate Speech Detection System | Powered by Custom Neural Network</p>
        <p style='font-size: 0.8rem;'>Model automatically retrains every hour with new feedback data</p>
    </div>
    """,
    unsafe_allow_html=True
)