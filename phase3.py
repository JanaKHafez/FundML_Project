import os
# Use all available CPUs for BLAS/OpenMP and related libs before importing heavy libs
cpu_count = str(os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = cpu_count
os.environ["OPENBLAS_NUM_THREADS"] = cpu_count
os.environ["MKL_NUM_THREADS"] = cpu_count
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_count
os.environ["NUMEXPR_NUM_THREADS"] = cpu_count

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import os

DATA_DIR = "/home/janhaf2n/SiemensLLMs/Temp/"

# Cleaned dataset
df_cleaned = pd.read_csv(os.path.join(DATA_DIR, "cleaned_hate_speech_dataset.csv"))
texts = df_cleaned["cleaned_content"].astype(str)
y = df_cleaned["Label"].astype(int)

# TF-IDF
X_tfidf = scipy.sparse.load_npz(os.path.join(DATA_DIR, "tfidf_matrix.npz"))
# BoW
X_bow = scipy.sparse.load_npz(os.path.join(DATA_DIR, "bow_matrix.npz"))

print("TF-IDF shape:", X_tfidf.shape)
print("BoW shape:", X_bow.shape)

# Try to use GPU if available, otherwise CPU. Also set PyTorch thread limits if available.
# device = "cpu"
# try:
#     import torch
#     if torch.cuda.is_available():
#         device = "cuda"
#     # limit PyTorch intra-op threads to avoid oversubscription when we parallelize training
#     try:
#         torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
#         torch.set_num_interop_threads(1)
#     except Exception:
#         pass
# except Exception:
#     device = "cpu"

# embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
# # Use a reasonable batch size and allow sentence-transformers to use multiple workers if supported
# num_workers = max(1, int(os.cpu_count() or 1) // 2)
# try:
#     X_embed = embedder.encode(texts.tolist(), show_progress_bar=True, batch_size=64, convert_to_numpy=True, device=device, num_workers=num_workers)
# except TypeError:
#     # fall back if this version doesn't support some args
#     X_embed = embedder.encode(texts.tolist(), show_progress_bar=True)

# # Save embeddings for reuse
# embed_path = os.path.join(DATA_DIR, "text_embeddings.npy")
# np.save(embed_path, X_embed)
# print(f"Embeddings saved to: {embed_path}")
# print("Embeddings shape:", X_embed.shape)

X_embed = np.load(os.path.join(DATA_DIR, "text_embeddings.npy"))
print("Loaded embeddings shape:", X_embed.shape)

X_embed = X_embed.astype(np.float32)
X_tfidf = X_tfidf.astype(np.float32)
X_bow = X_bow.astype(np.float32)

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
X_train_bow, X_test_bow, _, _ = train_test_split(
    X_bow, y, test_size=0.2, random_state=42, stratify=y)
X_train_embed, X_test_embed, _, _ = train_test_split(
    X_embed, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1), #
    "Linear Regression": LinearRegression(), #
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1), #
    "Perceptron": Perceptron(max_iter=1000, class_weight="balanced"), #
    "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=20),
    "Decision Tree (ID3)": DecisionTreeClassifier(criterion="entropy", random_state=42), #
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "Naive Bayes (Multinomial)": MultinomialNB(), #
    "Support Vector Machine": LinearSVC(class_weight="balanced"), #
}

feature_sets = {
    "TF-IDF": (X_train_tfidf, X_test_tfidf),
    "BoW": (X_train_bow, X_test_bow),
    "Embeddings": (X_train_embed, X_test_embed)
}

def evaluate_predictions(y_true, y_pred, model_name, feature_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"[{model_name} on {feature_name}] Accuracy: {acc:.4f}, F1: {f1:.4f}")
    # save results: model, feature, accuracy, precision, recall, f1
    return {
        "Model": model_name,
        "Features": feature_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

results = []

# Prepare tasks to run in parallel (one task per model-feature pair)
tasks = []
for feature_name, (Xtr, Xte) in feature_sets.items():
    print(f"\n=== Preparing tasks for {feature_name} features ===")
    for model_name, model in models.items():
        # Use GaussianNB for continuous embeddings (MultinomialNB is for count data)
        task_model = model
        task_model_name = model_name
        if feature_name == "Embeddings" and model_name == "Naive Bayes (Multinomial)":
            task_model = GaussianNB()
            task_model_name = "Naive Bayes (Gaussian)"
        tasks.append((feature_name, task_model_name, task_model, Xtr, Xte))

def _fit_and_eval(feature_name, model_name, model, Xtr, Xte, y_train, y_test):
    """Clone estimator, fit and evaluate inside a threadpool-limited context to avoid oversubscription."""
    try:
        with threadpool_limits(limits=1):
            est = clone(model)
            est.fit(Xtr, y_train)
            y_pred = est.predict(Xte)
            if model_name == "Linear Regression":
                y_pred = (y_pred > 0.5).astype(int)
        print(f"[{model_name} on {feature_name}] done")
        return evaluate_predictions(y_test, y_pred, model_name, feature_name)
    except Exception as e:
        print(f"[{model_name} on {feature_name}] failed: {e}")
        return None

# Run tasks in parallel using all available CPUs (joblib handles process pool)
parallel_results = Parallel(n_jobs=-1, prefer="processes")(delayed(_fit_and_eval)(fn, mn, m, Xtr, Xte, y_train, y_test) for (fn, mn, m, Xtr, Xte) in tasks)

for r in parallel_results:
    if r is not None:
        results.append(r)

results_df = pd.DataFrame(results)
results_df.sort_values(by=["Features", "F1"], ascending=[True, False], inplace=True)

out_path = os.path.join(DATA_DIR, "phase3_model_comparison.csv")
results_df.to_csv(out_path, index=False)

print("\n===== FINAL RESULTS =====")
print(results_df)
print(f"\nResults saved to: {out_path}")

# load the results
loaded_results = pd.read_csv(out_path)
print("\n===== LOADED RESULTS =====")
print(loaded_results)

#analyze results
print("\n===== ANALYSIS =====")
#show f1 scores by model and feature set
for feature_name in feature_sets.keys():    
    feature_results = loaded_results[loaded_results["Features"] == feature_name]
    print(f"\n--- F1 Scores for {feature_name} ---")
    for _, row in feature_results.iterrows():
        print(f"{row['Model']}: F1 = {row['F1']:.4f}")
    #show average f1 score for the feature set
    avg_f1 = feature_results["F1"].mean()
    print(f"Average F1 for {feature_name}: {avg_f1:.4f}")
    #show top 3 models for the feature set
    top3 = feature_results.nlargest(3, "F1")
    print("Top 3 Models:")
    for _, row in top3.iterrows():
        print(f"{row['Model']}: F1 = {row['F1']:.4f}")

#show average f1 score by model across feature sets
print("\n--- Average F1 Scores by Model ---")
for model_name in loaded_results["Model"].unique():
    model_results = loaded_results[loaded_results["Model"] == model_name]
    avg_f1 = model_results["F1"].mean()
    print(f"{model_name}: Average F1 = {avg_f1:.4f}")

#identify best feature set
best_feature_set = loaded_results.groupby("Features")["F1"].mean().idxmax()
print(f"\nBest Feature Set Overall: {best_feature_set}")

#identify top 3 models and their feature sets overall
top3_overall = loaded_results.nlargest(3, "F1")
print("\nTop 3 Models Overall:")
for _, row in top3_overall.iterrows():
    print(f"{row['Model']} with {row['Features']} features, F1 = {row['F1']:.4f}")

#Graphical analysis
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(data=loaded_results, x="Model", y="F1", hue="Features")
plt.title("F1 Scores by Model and Feature Set")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()