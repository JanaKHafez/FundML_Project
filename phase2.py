# %% [markdown]
# # Import

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("waalbannyantudre/hate-speech-detection-curated-dataset")

print("Path to dataset files:", path)

# %%
import pandas as pd
import os

for file in os.listdir(path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(path, file)
        break

# Load the dataset
df = pd.read_csv(dataset_path)

# Show the first few rows
print("Dataset shape:", df.shape)
print("Head: ")
print(df.head())

# %% [markdown]
# # Clean and Preprocess Text

# %%
import numpy as np

y = pd.factorize(df["Label"])[0].astype(float)
print(y.shape)

# Print all unique values in y
print("Unique values in y:", np.unique(y))

# Print number of entries with each unique value
print("Counts of each unique value in y:", np.bincount(y.astype(int)))

# Print all entries from df["content"] where y = 2
print("Entries where y=2:")
print(df["Content"][y == 2])

# %%
# Remove all entries where y = 2, as that isn't a valid label
df = df[y != 2]
print("Dataset shape:", df.shape)
print("Head: ")
print(df.head())

# %%
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

print("Cleaning text")
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

df["cleaned_content"] = df["Content"].apply(clean_text)
print("Text cleaning completed")
print("Shape: ", df.shape)
print("Head: ")
print(df.head())

# %% [markdown]
# # TF-IDF

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_content"])

# Only convert the first 5 rows to dense for display
sample_df = pd.DataFrame(
    X_tfidf[:5].toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("TF-IDF encoding completed")
print("Shape: ", X_tfidf.shape)
print("Sample head:")
print(sample_df.head())

# %% [markdown]
# # Bag-of-Words

# %%
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(df["cleaned_content"])

# Only convert the first 5 rows to dense for display
sample_bow_df = pd.DataFrame(
    X_bow[:5].toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("Bag-of-Words encoding completed")
print("Shape: ", X_bow.shape)
print("Sample head:")
print(sample_bow_df.head())

# %% [markdown]
# # Correlation

# %%
from scipy.stats import pearsonr
import pandas as pd

import numpy as np
import pandas as pd
from scipy import sparse

def correlations(X_sparse, y, feature_names, batch_size=1000):
    """
    Memory-safe sparse Pearson correlation for very large matrices.
    """
    if not sparse.isspmatrix_csr(X_sparse):
        X_sparse = X_sparse.tocsr()
    
    # Convert y to numeric if needed
    if not np.issubdtype(np.asarray(y).dtype, np.number):
        y = pd.factorize(y)[0]
    y = np.asarray(y, dtype=float).ravel()
    n = X_sparse.shape[0]

    y_mean = y.mean()
    y_std = y.std()
    y_centered = y - y_mean

    corrs = np.zeros(X_sparse.shape[1], dtype=np.float32)

    for start in range(0, X_sparse.shape[1], batch_size):
        end = min(start + batch_size, X_sparse.shape[1])
        X_batch = X_sparse[:, start:end]

        # Compute feature means and variances efficiently
        means = np.array(X_batch.mean(axis=0)).ravel()
        sumsq = np.array(X_batch.multiply(X_batch).mean(axis=0)).ravel()
        vars_ = sumsq - means**2
        stds = np.sqrt(np.maximum(vars_, 1e-12))

        # Compute covariances without centering the whole matrix
        cov = (X_batch.T.dot(y_centered) - n * means * y_mean) / (n - 1)

        # Correlation
        corrs[start:end] = cov / (stds * y_std)

    return pd.Series(corrs, index=feature_names).replace([np.inf, -np.inf], 0).fillna(0).sort_values(ascending=False)

y = pd.to_numeric(df["Label"], errors="coerce")

bow_feature_names = bow_vectorizer.get_feature_names_out() 
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

corr_bow = correlations(X_bow, y, bow_feature_names).sort_values(ascending=False)
corr_tfidf = correlations(X_tfidf, y, tfidf_feature_names).sort_values(ascending=False)

print_top_n = 20
print(f"Top {print_top_n} most correlated (BoW):")
print(corr_bow.head(print_top_n))
print(f"{print_top_n} least correlated (BoW):")
print(corr_bow.tail(print_top_n))

print(f"Top {print_top_n} most correlated (TF-IDF):")
print(corr_tfidf.head(print_top_n))
print(f"{print_top_n} least correlated (TF-IDF):")
print(corr_tfidf.tail(print_top_n))

# %% [markdown]
# # Dataset Analysis

# %%
import matplotlib.pyplot as plt
from collections import Counter

print("Dataset Overview:")
print(df.info(), "\n")
print(df.describe(include='all'))
print("\nClass distribution:")
print(df['Label'].value_counts(normalize=True) * 100)

print("Missing Values:")
print(df.isnull().sum())

plt.figure(figsize=(5,4))
df['Label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Label Distribution (0=Non-Hate, 1=Hate)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

df['word_count'] = df['cleaned_content'].apply(lambda x: len(str(x).split()))
print("Word Count Statistics:")
print(df['word_count'].describe())

plt.figure(figsize=(6,4))
plt.hist(df['word_count'], bins=30, color='purple', alpha=0.7)
plt.title("Distribution of Text Lengths (Word Count)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

all_words = " ".join(df["cleaned_content"]).split()
freq = Counter(all_words)
print("Top 50 Most Frequent Words:")
print(freq.most_common(50))

print("Unique and Duplicate Text Entries:")
print("Unique text entries:", df['Content'].nunique())
print("Duplicate entries:", df.duplicated(subset='Content').sum())

# Plot top correlated features for BoW
plt.figure(figsize=(10,5))
corr_bow.head(50).plot(kind='bar', color='teal')
plt.title("Top 50 Most Correlated Words (Bag-of-Words)")
plt.ylabel("Pearson Correlation with Label")
plt.show()

# Plot top correlated features for TF-IDF
plt.figure(figsize=(10,5))
corr_tfidf.head(50).plot(kind='bar', color='orange')
plt.title("Top 50 Most Correlated Words (TF-IDF)")
plt.ylabel("Pearson Correlation with Label")
plt.show()

# %% [markdown]
# # Save

# %%
import scipy.sparse

# Save cleaned DataFrame
df_cleaned = df.copy()
df_cleaned["cleaned_content"] = df["cleaned_content"]

ROOT = "/home/janhaf2n/fundML_Project/"

output_cleaned_path = os.path.join(ROOT, "cleaned_hate_speech_dataset.csv")
df_cleaned.to_csv(output_cleaned_path, index=False)
print(f"Cleaned dataset saved to: {output_cleaned_path}")

# Save TF-IDF sparse matrix and feature names
output_tfidf_matrix = os.path.join(ROOT, "tfidf_matrix.npz")
output_tfidf_features = os.path.join(ROOT, "tfidf_features.txt")
scipy.sparse.save_npz(output_tfidf_matrix, X_tfidf)
with open(output_tfidf_features, "w", encoding="utf-8") as f:
    for feat in tfidf_vectorizer.get_feature_names_out():
        f.write(f"{feat}\n")
print(f"TF-IDF matrix saved to: {output_tfidf_matrix}")
print(f"TF-IDF feature names saved to: {output_tfidf_features}")

# Save BoW sparse matrix and feature names
output_bow_matrix = os.path.join(ROOT, "bow_matrix.npz")
output_bow_features = os.path.join(ROOT, "bow_features.txt")
scipy.sparse.save_npz(output_bow_matrix, X_bow)
with open(output_bow_features, "w", encoding="utf-8") as f:
    for feat in bow_vectorizer.get_feature_names_out():
        f.write(f"{feat}\n")
print(f"BoW matrix saved to: {output_bow_matrix}")
print(f"BoW feature names saved to: {output_bow_features}")


