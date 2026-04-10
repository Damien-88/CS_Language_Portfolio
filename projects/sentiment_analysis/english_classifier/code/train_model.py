# ENGLISH SENTIMENT ANALYSIS MODEL TRAINING

# IMPORTS
# Data Handling
import pandas as pd

# Path Handling
from pathlib import Path

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Model Persistence
import pickle

# Preprocessing Pipeline
from preprocess_en import clean_text

# FILE PATHS
BASE_DIR = Path(__file__).resolve().parents[1] # Base directory
DATA_DIR = BASE_DIR / "data" # Data folder
DATA_FILE = DATA_DIR / "train_en.csv" # Input data
MODEL_FILE = DATA_DIR / "sentiment_model.pkl" # Output model
VECTORIZER_FILE = DATA_DIR / "vectorizer_en.pkl" # Output vectorizer


# LOAD DATA
df = pd.read_csv(DATA_FILE) # Load the dataset
df = df[["text", "sentiment"]] # Keep only relevant columns

# Map labels to numeric values
label_map = {
    "negative": 0,
    "neutral" : 1,
    "positive": 2
}
df["sentiment"] = df["sentiment"].map(label_map)

# Drop invalid rows
df = df.dropna(subset = ["text", "sentiment"])

# Convert to integer
df["sentiment"] = df["sentiment"].astype(int)

# Debug checks
print("Dataset size:", df.shape)
print("Class distribution:\n", df["sentiment"].value_counts())

# PREPROCESS TEXT
df["cleaned_text"] = df["text"].apply(clean_text) # Clean the text data

# SPLIT DATA
X = df["cleaned_text"] # Features
y = df["sentiment"] # Target variable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.2, # 20% for testing
    random_state = 42, # Reproducibility
    stratify = y # Ensure balanced class distribution in splits
)

# VECTORIZE TEXT TF-IDF
vectorizer = TfidfVectorizer(
    max_features = 5000, # Limit vocabulary size or efficienct
    ngram_range = (1, 2) # Unigrams and Bigrams to impprove accuracy
)

# Fit vectorizer on training data and transform train and test sets
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# TRAIN MODEL
# Logistic regression for strong baseline
model = LogisticRegression(max_iter = 1000, multi_class = "auto") # Increase max iterations for convergence
model.fit(X_train_vec, y_train) # Train model on vectorized training data

# EVALUATE MODEL
y_pred = model.predict(X_test_vec) # Predict on test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

print("MODEL PERFORMANCE")
print(f"Accuracy: {accuracy:.4f}\n") # Output accuracy
print(classification_report(y_test, y_pred)) # Output detailed classification report

# SAVE MODEL AND VECTORIZER
with open(MODEL_FILE, "wb") as mf:
    pickle.dump(model, mf) # Save trained model to file

with open(VECTORIZER_FILE, "wb") as vf:
    pickle.dump(vectorizer, vf) # Save vectorizer to file

# VERIFY COMPLETION
print("\nTraining Complete")
print(f"Model saved to {MODEL_FILE}")
print(f"Vectorizer saved to {VECTORIZER_FILE}")