# Utility functions for the Russian sentiment analysis workflow.

from pathlib import Path  # Cross-platform path handling.
import pickle  # Model/vectorizer serialization.
import re  # Regex-based text cleanup.
import unicodedata  # Unicode normalization utilities.

import pandas as pd  # DataFrame loading and saving.
from nltk.corpus import stopwords  # Russian stopword corpus.
from nltk import word_tokenize  # NLTK tokenizer fallback.

# Try to enable spaCy for better tokenization and lemmatization quality.
try:
    import spacy  # Import spaCy only if it exists in this environment.
    nlp = spacy.load("ru_core_news_sm")  # Load Russian model.
except (ImportError, OSError):  # Handle missing package or missing model files.
    nlp = None  # Use fallback logic when spaCy is unavailable.


# Keep negation terms so sentiment polarity is not lost during stopword removal.
NEGATION_WORDS = {
    "не",  # Core negation particle.
    "нет",  # Direct negation meaning "no".
    "ни",  # Particle for negative constructions.
    "никогда",  # Negation adverb meaning "never".
    "никто",  # Negative pronoun meaning "nobody".
    "ничто",  # Negative pronoun meaning "nothing".
    "нигде",  # Negative adverb meaning "nowhere".
    "никуда",  # Negative adverb meaning "to nowhere".
    "никак",  # Negative adverb meaning "in no way".
    "нисколько",  # Negative quantifier meaning "not at all".
}

# Load Russian stopwords if NLTK resources are installed.
try:
    base_stopwords = set(stopwords.words("russian"))  # O(1) membership checks.
except LookupError:  # NLTK corpus not downloaded in current runtime.
    base_stopwords = set()  # Safe fallback to keep preprocessing runnable.

STOPWORDS = base_stopwords - NEGATION_WORDS  # Keep negation cues for sentiment tasks.

# Precompile regex patterns once to avoid recompiling on each call.
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_TEXT_RE = re.compile(r"[^a-zа-яё0-9\s']")
WS_RE = re.compile(r"\s+")


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


def clean_text(text):  # Normalize and clean Russian text.
    """Normalize and clean Russian text for downstream vectorization."""
    text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize unicode and lowercase.
    text = URL_RE.sub("", text)  # Remove URLs.
    text = MENTION_RE.sub("", text)  # Remove @mentions.
    text = NON_TEXT_RE.sub(" ", text)  # Keep Russian/Latin letters, digits, spaces, apostrophes.
    text = WS_RE.sub(" ", text).strip()  # Collapse repeated whitespace.
    return text  # Return cleaned string.


def tokenize_text(text):  # Convert cleaned text to tokens.
    """Tokenize text with spaCy when available, else NLTK/split fallback."""
    if nlp:  # Prefer spaCy tokenizer quality if available.
        return [token.text for token in nlp(text)]  # Extract token text values.

    try:
        return word_tokenize(text, language="russian")  # Use NLTK tokenizer with Russian rules.
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
    """Run full Russian preprocessing pipeline and return a single string."""
    cleaned = clean_text(text)  # Cleanup and normalize input.

    # Fast path: run spaCy once and do filtering + lemmatization in one pass.
    if nlp:
        lemmas = [
            token.lemma_
            for token in nlp(cleaned)
            if not token.is_space and token.text not in STOPWORDS
        ]
        return " ".join(lemmas)

    tokens = tokenize_text(cleaned)  # Tokenize normalized text.
    tokens = remove_stopwords(tokens)  # Drop stopwords except key negations.
    return " ".join(tokens)  # Return vectorizer-ready sentence string.


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