# IMPORTS

# Pathlib is used for safe, cross-platform file path handling
from pathlib import Path

# Pandas is used for loading and handling CSV datasets
import pandas as pd

# Pickle is used to save and load Python objects (models + vectorizers)
import pickle

# Regular expressions are used for text cleaning
import re

# Unicode normalization ensures consistent text representation
import unicodedata

# NLTK stopwords provide a list of common words to remove (e.g., "the", "is")
import nltk
from nltk.corpus import stopwords


# OPTIONAL NLP ENGINE (spaCy)

# Try to load spaCy for better tokenization and lemmatization
# If spaCy is not available, fallback methods will be used
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")  # English language model
except Exception:
    nlp = None  # spaCy not available


# STOPWORDS INITIALIZATION

# Load English stopwords from NLTK
STOPWORDS = set(stopwords.words("english"))


# DATA HANDLING FUNCTIONS

def load_csv(file_path):
    """
    Load a CSV dataset into a pandas DataFrame.

    Expected format:
    - text column (input sentence)
    - sentiment column (label)
    """

    # Convert string path into Path object
    path = Path(file_path)

    # Ensure file exists before loading
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read CSV file into DataFrame
    return pd.read_csv(path)


def save_csv(df, file_path):
    """
    Save a pandas DataFrame to a CSV file.
    """

    # Convert file path into Path object
    path = Path(file_path)

    # Write DataFrame to CSV without index column
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def get_text_and_labels(df, text_col="text", label_col="sentiment"):
    """
    Split dataset into features (X) and labels (y).

    X = input text
    y = sentiment labels
    """

    # Extract text column as array
    X = df[text_col].values

    # Extract sentiment labels as array
    y = df[label_col].values

    return X, y


# TEXT PREPROCESSING FUNCTIONS

def clean_text(text):
    """
    Clean raw input text by:
    - Lowercasing
    - Removing URLs and mentions
    - Removing punctuation
    - Normalizing Unicode
    """

    # Normalize text to standard Unicode form
    text = unicodedata.normalize("NFKC", text)

    # Convert text to lowercase for consistency
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)

    # Remove @mentions (e.g., Twitter handles)
    text = re.sub(r"@\\w+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """
    Convert text into tokens (words).

    Uses spaCy if available, otherwise simple split.
    """

    # Use spaCy tokenizer if available (more accurate)
    if nlp:
        doc = nlp(text)
        return [token.text for token in doc]

    # Fallback: simple whitespace splitting
    return text.split()


def remove_stopwords(tokens):
    """
    Remove common English stopwords that do not add meaning.
    """

    return [token for token in tokens if token not in STOPWORDS]


def lemmatize_tokens(tokens):
    """
    Reduce words to their base form (lemmatization).

    Example:
    - "running" → "run"
    - "better" → "good" (spaCy dependent)
    """

    # Use spaCy lemmatizer if available
    if nlp:
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]

    # If spaCy is not available, return tokens unchanged
    return tokens


def preprocess_text(text):
    """
    Full preprocessing pipeline:
    clean → tokenize → remove stopwords → lemmatize

    Output is a single processed string ready for TF-IDF.
    """

    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    # Return final cleaned string for vectorizer input
    return " ".join(tokens)


# MODEL PERSISTENCE FUNCTIONS

def save_model(model, file_path):
    """
    Save trained ML model to disk using pickle.
    """

    # Open file in binary write mode
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def load_model(file_path):
    """
    Load trained ML model from disk.
    """

    # Open file in binary read mode
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_vectorizer(vectorizer, file_path):
    """
    Save TF-IDF vectorizer to disk.

    IMPORTANT:
    Must reuse same vocabulary during prediction.
    """

    with open(file_path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(file_path):
    """
    Load TF-IDF vectorizer from disk.
    """

    with open(file_path, "rb") as f:
        return pickle.load(f)