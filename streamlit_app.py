#template source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import streamlit as st

# Download NLTK resources if not already present
# do it just once
nltk.download('stopwords')

st.session_state.setdefault('stop_words', set(stopwords.words('english')))
stop_words = st.session_state['stop_words']
st.session_state.setdefault('stemmer', SnowballStemmer('english'))
stemmer = st.session_state['stemmer']

def clean_text(text):
    # Ensure it's a string
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove punctuation, numbers, special chars
    text = re.sub(r"[^a-z\s]", '', text)

    # Tokenize
    tokens = text.split()

    # Remove empty or stop words
    cleaned_tokens = []
    for w in tokens:
        if w and w not in stop_words:
            try:
                stemmed = stemmer.stem(w)
                cleaned_tokens.append(stemmed)
            except RecursionError:
                # If a weird token triggers recursion, skip it
                continue

    return " ".join(cleaned_tokens)


# Streamed response emulator
def response_generator(prompt):
    cleaned = clean_text(prompt)
    return cleaned

st.title("Hate Speech Detection")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter Text to Classify"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})