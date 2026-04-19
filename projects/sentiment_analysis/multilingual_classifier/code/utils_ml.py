"""
Backup utility module for the multilingual sentiment analysis classifier.

Provides helper classes/functions for data loading, saving, model persistence,
and multilingual text preprocessing for English, German, and Russian.

Note: This module is not part of the active classifier runtime path.
      The classifier uses preprocess_ml.py, train_model_ml.py,
      and predict_model_ml.py directly.
"""

# IMPORTS

# Pathlib is used for safe, cross-platform file path handling.
from pathlib import Path
# Pickle is used to save and load Python objects (models + vectorizers).
import pickle
# Regular expressions are used for text cleaning and language-specific filtering.
import re
# Unicode normalization ensures consistent text representation.
import unicodedata
# Pandas is used for loading and handling CSV datasets.
import pandas as pd
# NLTK stopwords provide language-specific low-information tokens to remove.
from nltk.corpus import stopwords
# NLTK tokenizer fallback when spaCy is unavailable.
from nltk import word_tokenize
# WordNet lemmatizer fallback for English when spaCy is unavailable.
from nltk.stem import WordNetLemmatizer


class DatasetUtils:
    """Data loading/saving helpers for CSV-based sentiment datasets."""

    @staticmethod
    def load_csv(file_path):
        """
        Load a CSV dataset into a pandas DataFrame.

        Expected format:
        - text column (input sentence)
        - sentiment column (label)
        """
        path = Path(file_path)  # Convert string path into Path object.
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")  # Fail early on missing data path.
        return pd.read_csv(path)  # Read CSV file into DataFrame.

    @staticmethod
    def save_csv(df, file_path):
        """Save a pandas DataFrame to a CSV file."""
        path = Path(file_path)  # Convert file path into Path object.
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination directory exists.
        df.to_csv(path, index=False, encoding="utf-8")  # Write DataFrame without index column.

    @staticmethod
    def get_text_and_labels(df, text_col="text", label_col="sentiment"):
        """Split dataset into features (X) and labels (y)."""
        X = df[text_col].values  # Extract text column as array.
        y = df[label_col].values  # Extract sentiment labels as array.
        return X, y


class ModelPersistenceUtils:
    """Pickle-based model/vectorizer persistence helpers."""

    @staticmethod
    def save_model(model, file_path):
        """Save trained ML model to disk using pickle."""
        with open(file_path, "wb") as f:  # Open file in binary write mode.
            pickle.dump(model, f)

    @staticmethod
    def load_model(file_path):
        """Load trained ML model from disk."""
        with open(file_path, "rb") as f:  # Open file in binary read mode.
            return pickle.load(f)

    @staticmethod
    def save_vectorizer(vectorizer, file_path):
        """
        Save TF-IDF vectorizer to disk.

        IMPORTANT:
        Must reuse same vocabulary during prediction.
        """
        with open(file_path, "wb") as f:
            pickle.dump(vectorizer, f)

    @staticmethod
    def load_vectorizer(file_path):
        """Load TF-IDF vectorizer from disk."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


class MultilingualTextProcessor:
    """Language-aware text preprocessor for English, German, and Russian."""

    # Per-language settings for stopwords, negations, and allowed characters.
    LANGUAGE_CONFIG = {
        "english": {
            "spacy_model": "en_core_web_sm",
            "nltk_language": "english",
            "stopwords_corpus": "english",
            "negation_words": {
                "no", "not", "nor", "never", "none", "nobody", "nothing",
                "neither", "nowhere", "hardly", "scarcely", "barely",
            },
            "char_pattern": r"[^a-z0-9\s']",
        },
        "german": {
            "spacy_model": "de_core_news_sm",
            "nltk_language": "german",
            "stopwords_corpus": "german",
            "negation_words": {
                "nicht", "kein", "keine", "keinen", "keinem", "keiner",
                "keines", "nie", "niemals", "weder", "noch",
            },
            "char_pattern": r"[^a-z0-9\säöüß']",
        },
        "russian": {
            "spacy_model": "ru_core_news_sm",
            "nltk_language": "russian",
            "stopwords_corpus": "russian",
            "negation_words": {
                "не", "нет", "ни", "никогда", "никто", "ничто",
                "нигде", "никуда", "никак", "нисколько",
            },
            "char_pattern": r"[^a-zа-яё0-9\s']",
        },
    }

    def __init__(self, language="english"):
        if language not in self.LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language '{language}'. Choose from: {list(self.LANGUAGE_CONFIG)}")

        self.language = language
        config = self.LANGUAGE_CONFIG[language]

        self.nltk_language = config["nltk_language"]  # NLTK tokenizer language key.
        self.url_re = re.compile(r"http\S+|www\.\S+")  # URL removal pattern.
        self.mention_re = re.compile(r"@\w+")  # Mention removal pattern.
        self.non_text_re = re.compile(config["char_pattern"])  # Language-aware character filter.
        self.ws_re = re.compile(r"\s+")  # Whitespace normalization pattern.

        # Try to load spaCy model for this language.
        try:
            import spacy
            self.nlp = spacy.load(config["spacy_model"])  # Language-specific spaCy pipeline.
        except Exception:
            self.nlp = None  # spaCy not available.

        # Load language stopwords and preserve negations.
        try:
            base_stopwords = set(stopwords.words(config["stopwords_corpus"]))
        except LookupError:
            base_stopwords = set()
        self.stopwords = base_stopwords - config["negation_words"]

        # English-only lemmatizer fallback for non-spaCy environments.
        self.lemmatizer = WordNetLemmatizer() if language == "english" else None

    def clean_text(self, text):
        """
        Clean raw input text by:
        - Lowercasing
        - Removing URLs and mentions
        - Removing punctuation
        - Normalizing Unicode
        """
        text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize Unicode and lowercase.
        text = self.url_re.sub("", text)  # Remove URLs.
        text = self.mention_re.sub("", text)  # Remove @mentions.
        text = self.non_text_re.sub(" ", text)  # Remove non-language characters.
        text = self.ws_re.sub(" ", text).strip()  # Collapse and trim whitespace.
        return text

    def tokenize_text(self, text):
        """Convert text into tokens (words)."""
        if self.nlp:
            return [token.text for token in self.nlp(text)]  # Use spaCy tokenizer when available.

        # Fallback: NLTK tokenization, then whitespace split if punkt is unavailable.
        try:
            return word_tokenize(text, language=self.nltk_language)
        except LookupError:
            return text.split()

    def remove_stopwords(self, tokens):
        """Remove language stopwords while preserving sentiment-relevant negations."""
        return [token for token in tokens if token not in self.stopwords]

    def lemmatize_tokens(self, tokens):
        """Reduce words to base form with spaCy (or English NLTK fallback)."""
        if self.nlp:
            doc = self.nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]

        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens  # For non-English without spaCy, keep tokens unchanged.

    def preprocess_text(self, text):
        """
        Full preprocessing pipeline:
        clean -> tokenize -> remove stopwords -> lemmatize

        Output is a single processed string ready for TF-IDF.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_text(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        return " ".join(tokens)  # Return final cleaned string for vectorizer input.


# Shared processor cache so models/resources are loaded once per language.
PROCESSOR_CACHE = {}


# BACKWARD-COMPATIBLE MODULE-LEVEL WRAPPERS

def load_csv(file_path):
    return DatasetUtils.load_csv(file_path)


def save_csv(df, file_path):
    DatasetUtils.save_csv(df, file_path)


def get_text_and_labels(df, text_col="text", label_col="sentiment"):
    return DatasetUtils.get_text_and_labels(df, text_col=text_col, label_col=label_col)


def preprocess_text(text, language="english"):
    # Reuse cached processor for efficiency across repeated calls.
    if language not in PROCESSOR_CACHE:
        PROCESSOR_CACHE[language] = MultilingualTextProcessor(language=language)
    return PROCESSOR_CACHE[language].preprocess_text(text)


def save_model(model, file_path):
    ModelPersistenceUtils.save_model(model, file_path)


def load_model(file_path):
    return ModelPersistenceUtils.load_model(file_path)


def save_vectorizer(vectorizer, file_path):
    ModelPersistenceUtils.save_vectorizer(vectorizer, file_path)


def load_vectorizer(file_path):
    return ModelPersistenceUtils.load_vectorizer(file_path)