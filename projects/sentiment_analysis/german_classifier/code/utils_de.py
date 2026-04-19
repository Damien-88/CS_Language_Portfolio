# Utility functions for the German sentiment analysis workflow.

from pathlib import Path  # Cross-platform path handling.
import pickle  # Model/vectorizer serialization.
import re  # Regex-based text cleanup.
import unicodedata  # Unicode normalization utilities.

import pandas as pd  # DataFrame loading and saving.
from nltk.corpus import stopwords  # German stopword corpus.
from nltk import word_tokenize  # NLTK tokenizer fallback.

# Try to enable spaCy for better tokenization and lemmatization quality.
try:
    import spacy  # Import spaCy only if it exists in this environment.
    nlp = spacy.load("de_core_news_sm")  # Load German model.
except (ImportError, OSError):  # Handle missing package or missing model files.
    nlp = None  # Use fallback logic when spaCy is unavailable.


# Keep negation terms so sentiment polarity is not lost during stopword removal.
NEGATION_WORDS = {
    "nicht",  # Core negation particle.
    "kein",  # Negation determiner (masculine/neuter base).
    "keine",  # Negation determiner form.
    "keinen",  # Negation determiner form.
    "keinem",  # Negation determiner form.
    "keiner",  # Negation determiner form.
    "keines",  # Negation determiner form.
    "nie",  # Negation adverb meaning "never".
    "niemals",  # Stronger negation adverb meaning "never".
    "weder",  # Negation conjunction used in pair constructions.
    "noch",  # Paired conjunction used with "weder".
}

# Load German stopwords if NLTK resources are installed.
try:
    base_stopwords = set(stopwords.words("german"))  # Convert to set for O(1) membership tests.
except LookupError:  # NLTK corpus not downloaded in current runtime.
    base_stopwords = set()  # Safe fallback to keep preprocessing runnable.

STOPWORDS = base_stopwords - NEGATION_WORDS  # Keep negation cues for sentiment tasks.


def load_csv(file_path):  # Load CSV into a DataFrame.
    """Load CSV data from disk into a pandas DataFrame."""
    path = Path(file_path)  # Normalize to a Path object.
    if not path.exists():  # Validate that input file exists.
        raise FileNotFoundError(f"File not found: {file_path}")  # Fail fast with explicit message.
    return pd.read_csv(path, encoding="utf-8")  # Read CSV using UTF-8.


def save_csv(df, file_path):  # Save DataFrame to CSV.
    """Save a pandas DataFrame to CSV using UTF-8 encoding."""
    path = Path(file_path)  # Normalize to a Path object.
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.
    df.to_csv(path, index=False, encoding="utf-8")  # Write CSV without index column.


def get_text_and_labels(df, text_col="text", label_col="sentiment"):  # Split frame into features and labels.
    """Return text and label arrays from the given DataFrame."""
    missing = [col for col in (text_col, label_col) if col not in df.columns]  # Detect absent required columns.
    if missing:  # Guard against schema mismatch.
        raise KeyError(f"Missing required columns: {missing}")  # Raise clear schema error.

    return df[text_col].values, df[label_col].values  # Return NumPy-backed arrays.


def clean_text(text):  # Normalize and clean German text.
    """Normalize and clean German text for downstream vectorization."""
    text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize unicode and lowercase.
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs.
    text = re.sub(r"@\w+", "", text)  # Remove @mentions.
    text = re.sub(r"[^a-z0-9\säöüß']", " ", text)  # Keep German letters, digits, spaces, apostrophes.
    text = re.sub(r"\s+", " ", text).strip()  # Collapse repeated whitespace.
    return text  # Return cleaned string.


def tokenize_text(text):  # Convert cleaned text to tokens.
    """Tokenize text with spaCy when available, else NLTK/split fallback."""
    if nlp:  # Prefer spaCy tokenizer quality if available.
        return [token.text for token in nlp(text)]  # Extract token text values.

    try:
        return word_tokenize(text, language="german")  # Use NLTK tokenizer with German rules.
    except LookupError:  # Punkt resource may be missing in some environments.
        return text.split()  # Fallback tokenizer based on whitespace.


def remove_stopwords(tokens):  # Remove low-information tokens.
    """Remove stopwords while preserving sentiment-bearing negations."""
    return [token for token in tokens if token not in STOPWORDS]  # Keep tokens not in stopword set.


def lemmatize_tokens(tokens):  # Reduce words to lemma forms.
    """Lemmatize tokens with spaCy; return raw tokens if unavailable."""
    if nlp:  # Lemmatization requires spaCy model.
        doc = nlp(" ".join(tokens))  # Rebuild text and process with spaCy pipeline.
        return [token.lemma_ for token in doc]  # Return lemma for each token.
    return tokens  # Keep original tokens when lemmatizer is unavailable.


def preprocess_text(text):  # Full preprocessing pipeline.
    """Run full German preprocessing pipeline and return a single string."""
    cleaned = clean_text(text)  # Cleanup and normalize input.
    tokens = tokenize_text(cleaned)  # Tokenize normalized text.
    tokens = remove_stopwords(tokens)  # Drop stopwords except key negations.
    lemmas = lemmatize_tokens(tokens)  # Lemmatize tokens where possible.
    return " ".join(lemmas)  # Return vectorizer-ready sentence string.


def save_model(model, file_path):  # Persist model artifact.
    """Persist a trained model to disk."""
    path = Path(file_path)  # Normalize path input.
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination directory exists.
    with path.open("wb") as f:  # Open target file in binary write mode.
        pickle.dump(model, f)  # Serialize model object with pickle.


def load_model(file_path):  # Load persisted model artifact.
    """Load a trained model from disk."""
    path = Path(file_path)  # Normalize path input.
    with path.open("rb") as f:  # Open target file in binary read mode.
        return pickle.load(f)  # Deserialize and return model.


def save_vectorizer(vectorizer, file_path):  # Persist vectorizer artifact.
    """Persist a fitted TF-IDF vectorizer to disk."""
    path = Path(file_path)  # Normalize path input.
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination directory exists.
    with path.open("wb") as f:  # Open target file in binary write mode.
        pickle.dump(vectorizer, f)  # Serialize vectorizer with pickle.


def load_vectorizer(file_path):  # Load persisted vectorizer artifact.
    """Load a fitted TF-IDF vectorizer from disk."""
    path = Path(file_path)  # Normalize path input.
    with path.open("rb") as f:  # Open target file in binary read mode.
        return pickle.load(f)  # Deserialize and return vectorizer.