# hate_speech_detection_pytorch.py
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
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ---------------- Config ----------------
ROOT = "/home/janhaf2n/fundML_Project/"
OUTPUT_DIR = os.path.join(ROOT, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 512
H1, H2 = 128, 64

# ---------------- Text Cleaning ----------------
nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

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

# ---------------- Data Loading ----------------
print("[1/5] Downloading dataset from Kaggle...")
dataset_dir = kagglehub.dataset_download("waalbannyantudre/hate-speech-detection-curated-dataset")
csv_path = next(os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv"))
df = pd.read_csv(csv_path)
print(f"Dataset loaded. Shape: {df.shape}")

y_temp = pd.factorize(df["Label"])[0].astype(float)
df = df[y_temp != 2].reset_index(drop=True)
df["Label"] = pd.factorize(df["Label"])[0].astype(int)

print("[2/5] Cleaning text...")
df["cleaned_content"] = df["Content"].astype(str).map(clean_text)
print("Text cleaning completed.")

df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_hate_speech_dataset.csv"), index=False)

# ---------------- TF-IDF ----------------
print("[3/5] Creating TF-IDF vectors...")
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df["cleaned_content"])
y = df["Label"].values.astype(int)
print(f"TF-IDF shape: {X_tfidf.shape}")
scipy.sparse.save_npz(os.path.join(OUTPUT_DIR, "tfidf_matrix.npz"), X_tfidf)
with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# ---------------- Train/Test Split ----------------
X_train_sp, X_test_sp, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=SEED, stratify=y
)
num_neg, num_pos = np.bincount(y_train)
pos_weight = float(num_neg) / max(1.0, float(num_pos))
print(f"[4/5] Train/Test split done. Train size: {X_train_sp.shape[0]}, Test size: {X_test_sp.shape[0]}")

# ---------------- PyTorch Model ----------------
class HateSpeechNN(nn.Module):
    def __init__(self, input_size, embedding_size=EMBEDDING_SIZE, h1=H1, h2=H2):
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

# ---------------- Dataset ----------------
try:
    X_train_dense = X_train_sp.toarray().astype(np.float32)
    X_test_dense = X_test_sp.toarray().astype(np.float32)
    train_dataset = TensorDataset(torch.from_numpy(X_train_dense), torch.from_numpy(y_train.astype(np.float32)))
    test_dataset = TensorDataset(torch.from_numpy(X_test_dense), torch.from_numpy(y_test.astype(np.float32)))
except MemoryError:
    class SparseDataset(Dataset):
        def __init__(self, sp_matrix, labels): self.X, self.y = sp_matrix, labels
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx].toarray().astype(np.float32).ravel(), np.float32(self.y[idx])
    train_dataset = SparseDataset(X_train_sp, y_train)
    test_dataset = SparseDataset(X_test_sp, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model, Loss, Optimizer ----------------
input_size = X_tfidf.shape[1]
model = HateSpeechNN(input_size=input_size).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------- Training & Evaluation ----------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_total = 0.0
    preds, labels = [], []

    for i, (xb, yb) in enumerate(loader, 1):
        # Move to device and ensure float32
        xb, yb = xb.to(device).float(), yb.to(device).float()

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * xb.size(0)

        # Detach before converting to numpy
        pred = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(int)
        preds.append(pred)
        labels.append(yb.detach().cpu().numpy().astype(int))

        if i % 10 == 0 or i == len(loader):
            print(f"  Batch {i}/{len(loader)} completed")

    avg_loss = loss_total / len(loader.dataset)
    accuracy = accuracy_score(np.concatenate(labels), np.concatenate(preds))
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader, 1):
            # Move to device and ensure float32
            xb, yb = xb.to(device).float(), yb.to(device).float()
            logits = model(xb)
            loss_total += criterion(logits, yb).item() * xb.size(0)

            # Detach before converting to numpy
            pred = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(int)
            preds.append(pred)
            labels.append(yb.detach().cpu().numpy().astype(int))

            if i % 10 == 0 or i == len(loader):
                print(f"  Eval batch {i}/{len(loader)} completed")

    avg_loss = loss_total / len(loader.dataset)
    labels_all = np.concatenate(labels)
    preds_all = np.concatenate(preds)
    accuracy = accuracy_score(labels_all, preds_all)
    return avg_loss, accuracy, labels_all, preds_all

history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
print("[5/5] Starting training...")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    print(f"Epoch {epoch}/{EPOCHS}...")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc, y_true_all, y_pred_all = evaluate(model, test_loader, criterion, DEVICE)
    history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
    print(f"Epoch {epoch} completed: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, time={time.time()-t0:.2f}s\n")

# ---------------- Metrics ----------------
accuracy = accuracy_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
print(f"Final Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print("Classification Report:\n", classification_report(y_true_all, y_pred_all, target_names=['Non-Hate','Hate']))
print("Confusion Matrix:\n", confusion_matrix(y_true_all, y_pred_all))

# ---------------- Save artifacts ----------------
torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": input_size,
    "embedding_size": EMBEDDING_SIZE,
    "hidden1_size": H1, "hidden2_size": H2,
    "pos_weight": pos_weight,
    "history": history
}, os.path.join(OUTPUT_DIR,"hate_speech_model.pt"))
with open(os.path.join(OUTPUT_DIR,"tfidf_vectorizer.pkl"),"wb") as f: pickle.dump(tfidf,f)
with open(os.path.join(OUTPUT_DIR,"model_metrics.txt"),"w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n")
with open(os.path.join(OUTPUT_DIR,'text_processor.pkl'), 'wb') as f:
    pickle.dump({'clean_text_function': clean_text, 'stemmer': stemmer, 'stop_words': stop_words}, f)
print("All artifacts saved. Done.")

