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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import multiprocessing

# ---------------- Config ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 512
H1, H2 = 128, 64

torch.set_float32_matmul_precision('high') 

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


# ---------------- SparseDataset (module-level for multiprocessing) ----------------
class SparseDataset(Dataset):
    """Dataset wrapper for scipy sparse matrices kept at module-level so it can be pickled
    by multiprocessing workers (required on Windows spawn start method).
    """
    def __init__(self, sp_matrix, labels):
        self.X = sp_matrix
        self.y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().astype(np.float32).ravel()
        label = np.float32(self.y[idx])
        return row, label

# ---------------- Main execution (wrapped) ----------------
def main():
    # ---------------- Data Loading ----------------
    print("[1/5] Loading cleaned dataset...")
    cleaned_csv_path = os.path.join(OUTPUT_DIR, "cleaned_hate_speech_dataset.csv")
    
    if os.path.exists(cleaned_csv_path):
        df = pd.read_csv(cleaned_csv_path)
        print(f"Dataset loaded from cache. Shape: {df.shape}")
    else:
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

        df.to_csv(cleaned_csv_path, index=False)

    # ---------------- TF-IDF ----------------
    print("[3/5] Creating TF-IDF vectors...")
    tfidf_path = os.path.join(OUTPUT_DIR, "tfidf_matrix.npz")
    tfidf_pkl_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    
    if os.path.exists(tfidf_path) and os.path.exists(tfidf_pkl_path):
        X_tfidf = scipy.sparse.load_npz(tfidf_path)
        with open(tfidf_pkl_path, "rb") as f:
            tfidf = pickle.load(f)
        print(f"TF-IDF loaded from cache. Shape: {X_tfidf.shape}")
    else:
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(df["cleaned_content"])
        scipy.sparse.save_npz(tfidf_path, X_tfidf)
        with open(tfidf_pkl_path, "wb") as f:
            pickle.dump(tfidf, f)
        print(f"TF-IDF created and saved. Shape: {X_tfidf.shape}")
    
    y = df["Label"].values.astype(int)

    # ---------------- Train/Test Split ----------------
    X_train_sp, X_test_sp, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=SEED, stratify=y
    )
    num_neg, num_pos = np.bincount(y_train)
    pos_weight = float(num_neg) / max(1.0, float(num_pos))
    print(f"[4/5] Train/Test split done. Train size: {X_train_sp.shape[0]}, Test size: {X_test_sp.shape[0]}")

    # ---------------- PyTorch Lightning Model ----------------
    class HateSpeechNN(pl.LightningModule):
        def __init__(self, input_size, embedding_size=EMBEDDING_SIZE, h1=H1, h2=H2, pos_weight=1.0, lr=LEARNING_RATE):
            super().__init__()
            self.save_hyperparameters()
            self.embedding = nn.Linear(input_size, embedding_size)
            self.hidden1 = nn.Linear(embedding_size, h1)
            self.hidden2 = nn.Linear(h1, h2)
            self.output = nn.Linear(h2, 1)
            self.relu = nn.ReLU()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            self.lr = lr
            
        def forward(self, x):
            x = self.relu(self.hidden1(self.embedding(x)))
            x = self.relu(self.hidden2(x))
            return self.output(x).squeeze(1)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc', acc, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
            return {'preds': preds, 'labels': y}
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ---------------- Dataset ----------------
    try:
        X_train_dense = X_train_sp.toarray().astype(np.float32)
        X_test_dense = X_test_sp.toarray().astype(np.float32)
        train_dataset = TensorDataset(torch.from_numpy(X_train_dense), torch.from_numpy(y_train.astype(np.float32)))
        test_dataset = TensorDataset(torch.from_numpy(X_test_dense), torch.from_numpy(y_test.astype(np.float32)))
    except MemoryError:
        train_dataset = SparseDataset(X_train_sp, y_train)
        test_dataset = SparseDataset(X_test_sp, y_test)

    env_workers = os.environ.get("NUM_WORKERS")
    if env_workers is not None:
        try:
            workers = int(env_workers)
        except ValueError:
            workers = max(0, (os.cpu_count() or 1) - 1)
    else:
        cpu = os.cpu_count() or 1
        workers = max(0, min(cpu - 1, 19))

    is_sparse = scipy.sparse.issparse(X_train_sp)
    if os.name == 'nt' and is_sparse:
        print("Detected sparse dataset on Windows")
        workers = 0

    input_size = X_tfidf.shape[1]
    bytes_per_sample = input_size * 4
    est_batch_bytes = bytes_per_sample * BATCH_SIZE
    max_batch_bytes = 2 * 1024 ** 3
    batch_size = BATCH_SIZE
    if est_batch_bytes > max_batch_bytes:
        safe_batch = max(1, int(max_batch_bytes // bytes_per_sample))
        print(f"Estimated dense batch size ({est_batch_bytes/(1024**3):.2f} GiB) is too large. Reducing batch size from {BATCH_SIZE} to {safe_batch}.")
        batch_size = safe_batch

    print(f"DataLoader workers selected: {workers}, batch_size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=(workers>0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=(workers>0))

    # ---------------- Model & Trainer ----------------
    input_size = X_tfidf.shape[1]
    model = HateSpeechNN(input_size=input_size, pos_weight=pos_weight, lr=LEARNING_RATE)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )

    # ---------------- Training ----------------
    print("[5/5] Starting training with PyTorch Lightning...")
    trainer.fit(model, train_loader, test_loader)

    # ---------------- Evaluation ----------------
    print("Evaluating model...")
    model.eval()
    model = model.to(DEVICE)
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            logits = model(xb)
            preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
            y_pred_all.append(preds)
            y_true_all.append(yb.cpu().numpy().astype(int))
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

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
        "pos_weight": pos_weight
    }, os.path.join(OUTPUT_DIR,"hate_speech_model.pt"))
    with open(os.path.join(OUTPUT_DIR,"tfidf_vectorizer.pkl"),"wb") as f: pickle.dump(tfidf,f)
    with open(os.path.join(OUTPUT_DIR,"model_metrics.txt"),"w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n")
    with open(os.path.join(OUTPUT_DIR,'text_processor.pkl'), 'wb') as f:
        pickle.dump({'clean_text_function': clean_text, 'stemmer': stemmer, 'stop_words': stop_words}, f)
    print("All artifacts saved. Done.")


if __name__ == '__main__':
    if os.name == 'nt':
        multiprocessing.freeze_support()
    main()

