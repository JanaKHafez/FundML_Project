from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
import threading
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(ROOT)
OUTPUT_DIR = os.path.join(ROOT, "Output")
FEEDBACK_FILE = os.path.join(ROOT, "user_feedback.csv")
RETRAIN_INTERVAL = 3600  # Retrain every hour

# Global variables
model = None
tfidf_vectorizer = None
cleaning_components = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_meta = {}

class HateSpeechNN(nn.Module):
    def __init__(self, input_size, embedding_size=512, h1=128, h2=64):
        super().__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.hidden1 = nn.Linear(embedding_size, h1)
        self.hidden2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(self.embedding(x)))
        x = self.relu(self.hidden2(x))
        return self.output(x).squeeze(1)

def clean_text(text):        
    if cleaning_components is None:
        return text
    stemmer = cleaning_components['stemmer']
    stop_words = cleaning_components['stop_words']
    # Ensure it's a string
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove punctuation, numbers, special chars
    text = re.sub(r"[^a-z\s]", '', text)
    # Tokenize
    tokens = text.split()
    # Remove empty or stop words and apply stemming
    cleaned_tokens = []
    for w in tokens:
        if w and w not in stop_words:
            try:
                stemmed = stemmer.stem(w)
                cleaned_tokens.append(stemmed)
            except RecursionError:
                continue
    
    return " ".join(cleaned_tokens)

def load():
    global model, tfidf_vectorizer, cleaning_components, model_meta

    print(f"Loading model and artifacts on {device}...")

    try:
        # 1. Load TF-IDF vectorizer
        vectorizer_path = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # 2. Load PyTorch Model Checkpoint
        model_path = os.path.join(OUTPUT_DIR, 'hate_speech_model.pt')
        checkpoint = torch.load(model_path, map_location=device)

        # Extract metadata
        input_size = checkpoint.get('input_size', len(tfidf_vectorizer.vocabulary_))
        embedding_size = checkpoint.get('embedding_size', 512)
        h1 = checkpoint.get('hidden1_size', 128)
        h2 = checkpoint.get('hidden2_size', 64)
        model_meta['pos_weight'] = checkpoint.get('pos_weight', 1.0)

        # Initialize and load model
        model = HateSpeechNN(input_size=input_size, embedding_size=embedding_size, h1=h1, h2=h2)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except RuntimeError as e:
            print(f"Warning: {e}")
            
        model.to(device)
        model.eval()

        # 3. Load text processing components
        text_processor_path = os.path.join(OUTPUT_DIR, 'text_processor.pkl')
        with open(text_processor_path, 'rb') as f:
            text_processor = pickle.load(f)
            cleaning_components = {
                'stemmer': text_processor['stemmer'],
                'stop_words': text_processor['stop_words']
            }

        print("Model and artifacts loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict hate speech using PyTorch model."""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if model is None or tfidf_vectorizer is None:
            return jsonify({'error': 'Model or preprocessing artifacts not loaded on server'}), 503
        
        # 1. Clean text
        cleaned_text = clean_text(text)
        
        # 2. TF-IDF
        tfidf_vector = tfidf_vectorizer.transform([cleaned_text])
        
        # 3. Convert to tensor
        input_tensor = torch.tensor(tfidf_vector.toarray(), dtype=torch.float32).to(device)
        
        # 4. Predict
        with torch.no_grad():
            logits = model(input_tensor)
            prob_non_hate = float(torch.sigmoid(logits).item())

        prob_hate = 1.0 - prob_non_hate

        if prob_hate >= 0.5:
            prediction = 0
            label = "Hate"
            confidence = prob_hate
        else:
            prediction = 1
            label = "Non-Hate"
            confidence = prob_non_hate

        response = {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': int(prediction),
            'label': label,
            'confidence': float(confidence),
            'probability_hate': prob_hate,
            'probability_non_hate': prob_non_hate,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        text = data.get('text', '')
        true_label = data.get('true_label')
        
        if not text or true_label is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        cleaned_text = clean_text(text)
        
        
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'cleaned_text': cleaned_text,
            'used_for_training': False,
            'true_label': true_label
        }
        
        df_feedback = pd.DataFrame([feedback_entry])
        
        if os.path.exists(FEEDBACK_FILE):
            df_feedback.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            df_feedback.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)
        
        return jsonify({'status': 'success', 'message': 'Feedback received'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def periodic_retrain():
    while True:
        time.sleep(RETRAIN_INTERVAL)
        try:
            if os.path.exists(FEEDBACK_FILE):
                df = pd.read_csv(FEEDBACK_FILE)
                unused = df[df['used_for_training'] == False]
                
                if len(unused) >= 10:
                    print(f"\n[{datetime.now()}] Retraining with {len(unused)} new samples...")
                    retrain_model(unused)
                else:
                    print(f"Not enough feedback ({len(unused)})")
        except Exception as e:
            print(f"Error in retraining thread: {e}")

def retrain_model(new_data):
    global model
    
    try:
        # Prepare Data
        X_new_tfidf = tfidf_vectorizer.transform(new_data['cleaned_text'].values)
        X_tensor = torch.tensor(X_new_tfidf.toarray(), dtype=torch.float32).to(device)
        y_tensor = torch.tensor(new_data['true_label'].values, dtype=torch.float32).to(device)
        
        # Create Dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Setup Training
        model.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(model_meta['pos_weight']).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Low LR for fine-tuning
        
        # Training Loop (Fine-tune for 5 epochs)
        epochs = 5
        print(f"Fine-tuning on {len(new_data)} samples for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        print(f"Retraining finished. Final Loss: {total_loss / len(loader):.4f}")
        
        # Save Updated Model
        model.eval()
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_size": X_tensor.shape[1],
            "pos_weight": model_meta['pos_weight']
        }, os.path.join(OUTPUT_DIR, "hate_speech_model.pt"))
        
        # Update CSV
        df_all = pd.read_csv(FEEDBACK_FILE)
        df_all.loc[df_all['used_for_training'] == False, 'used_for_training'] = True
        df_all.to_csv(FEEDBACK_FILE, index=False)
        
        print("Model saved and feedback marked as used.")
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        model.eval()

if __name__ == '__main__':
    load()
    
    retrain_thread = threading.Thread(target=periodic_retrain, daemon=True)
    retrain_thread.start()
    
    print("Starting Flask API server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)