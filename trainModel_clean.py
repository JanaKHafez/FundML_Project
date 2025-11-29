"""
This file implements a clean version of the model training without additional optimizations.
"""

import os, re, pickle, time
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 512
H1, H2 = 128, 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    tokens = text.split()

    cleaned = []
    for w in tokens:
        if w and w not in stop_words:
            try:
                cleaned.append(stemmer.stem(w))
            except RecursionError:
                continue
    return " ".join(cleaned)

class HateSpeechNN(nn.Module):
    def __init__(self, input_size, embedding_size=EMBEDDING_SIZE, h1=H1, h2=H2, pos_weight=1.0):
        super().__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.hidden1 = nn.Linear(embedding_size, h1)
        self.hidden2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, 1)
        self.relu = nn.ReLU()

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
    def forward(self, x):
        x = self.relu(self.hidden1(self.embedding(x)))
        x = self.relu(self.hidden2(x))
        return self.output(x).squeeze(1)

def main():

    # Load & Clean Dataset
    cleaned_csv_path = os.path.join(OUTPUT_DIR, "cleaned_hate_speech_dataset.csv")

    if os.path.exists(cleaned_csv_path):
        df = pd.read_csv(cleaned_csv_path)
        print(f"Loaded cleaned dataset. Shape: {df.shape}")
    else:
        print("Downloading dataset from Kaggle...")
        dataset_dir = kagglehub.dataset_download("waalbannyantudre/hate-speech-detection-curated-dataset")
        csv_path = next(os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv"))

        df = pd.read_csv(csv_path)
        y_temp = pd.factorize(df["Label"])[0].astype(float)
        df = df[y_temp != 2].reset_index(drop=True)
        df["Label"] = pd.factorize(df["Label"])[0].astype(int)

        print("Cleaning text...")
        df["cleaned_content"] = df["Content"].astype(str).map(clean_text)
        df.to_csv(cleaned_csv_path, index=False)
        print("Saved cleaned dataset.")

    # TF-IDF
    tfidf_path = os.path.join(OUTPUT_DIR, "tfidf_matrix.npz")
    tfidf_pkl_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")

    if os.path.exists(tfidf_path) and os.path.exists(tfidf_pkl_path):
        X_tfidf = scipy.sparse.load_npz(tfidf_path)
        with open(tfidf_pkl_path, "rb") as f:
            tfidf = pickle.load(f)
        print("Loaded TF-IDF from cache.")
    else:
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(df["cleaned_content"])
        scipy.sparse.save_npz(tfidf_path, X_tfidf)
        with open(tfidf_pkl_path, "wb") as f:
            pickle.dump(tfidf, f)
        print("Created and saved TF-IDF.")

    y = df["Label"].values.astype(int)

    # Split
    X_train_sp, X_test_sp, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=SEED, stratify=y
    )

    num_neg, num_pos = np.bincount(y_train)
    pos_weight = float(num_neg) / max(1.0, float(num_pos))

    # Convert to Dense
    X_train = X_train_sp.toarray().astype(np.float32)
    X_test = X_test_sp.toarray().astype(np.float32)

    # Datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test.astype(np.float32))
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    input_size = X_tfidf.shape[1]
    model = HateSpeechNN(input_size=input_size, pos_weight=pos_weight).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = model.loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # Evaluate
    print("Evaluating...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
            y_pred.append(preds)
            y_true.append(yb.numpy().astype(int))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['Non-Hate', 'Hate']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "embedding_size": EMBEDDING_SIZE,
        "hidden1_size": H1,
        "hidden2_size": H2,
        "pos_weight": pos_weight
    }, os.path.join(OUTPUT_DIR, "hate_speech_model.pt"))

    with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n")

    with open(os.path.join(OUTPUT_DIR, "text_processor.pkl"), "wb") as f:
        pickle.dump(
            {"clean_text_function": clean_text, "stemmer": stemmer, "stop_words": stop_words},
            f
        )

    print("All artifacts saved. Done.")


if __name__ == "__main__":
    main()