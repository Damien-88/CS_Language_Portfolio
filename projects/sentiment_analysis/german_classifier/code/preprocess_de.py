# German text preprocessing module for sentiment analysis.

import re  # Pattern-based text cleanup.
import unicodedata  # Unicode normalization helpers.

from nltk.corpus import stopwords  # German stopword list support.
from nltk import word_tokenize  # Fallback tokenizer.

# Try to load spaCy for better tokenization and lemmatization.
try:
    import spacy  # Import spaCy only if available.
    nlp = spacy.load("de_core_news_sm")  # Load small German language model.
except Exception:  # Handle missing package/model.
    nlp = None  # Indicates spaCy is unavailable.


# Keep sentiment-relevant negation words when removing stopwords.
NEGATION_WORDS = {
    "nicht",  # Core negation particle.
    "kein",  # Negation determiner (base form).
    "keine",  # Negation determiner variant.
    "keinen",  # Negation determiner variant.
    "keinem",  # Negation determiner variant.
    "keiner",  # Negation determiner variant.
    "keines",  # Negation determiner variant.
    "nie",  # Negation adverb meaning never.
    "niemals",  # Stronger negation adverb meaning never.
    "weder",  # First half of weder...noch construction.
    "noch"  # Second half of weder...noch construction.
}

try:
    base_stopwords = set(stopwords.words("german"))  # Faster membership checks.
except LookupError:
    base_stopwords = set()  # Safe fallback for runtime portability.

STOPWORDS = base_stopwords - NEGATION_WORDS  # Preserve negation polarity cues.


def clean_text(text):  # Clean and normalize raw input text.
    text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize and lowercase.
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs.
    text = re.sub(r"@\w+", "", text)  # Remove @mentions.
    text = re.sub(r"[^a-z0-9\säöüß']", " ", text)  # Keep letters/digits/whitespace/apostrophes.
    text = re.sub(r"\s+", " ", text).strip()  # Collapse and trim whitespace.
    return text  # Return cleaned text string.


def tokenize_text(text):  # Tokenize cleaned text.
    if nlp:
        return [t.text for t in nlp(text)]  # Use spaCy tokens when available.

    try:
        return word_tokenize(text, language = "german")  # Use NLTK punkt tokenizer.
    except LookupError:
        return text.split()  # Fallback if punkt resource is missing.


def remove_stopwords(tokens):  # Remove low-information tokens.
    return [token for token in tokens if token not in STOPWORDS]  # Keep sentiment-carrying terms.


def lemmatize_tokens(tokens):  # Reduce inflected forms to base forms.
    if nlp:
        doc = nlp(" ".join(tokens))  # Create a spaCy doc from tokens.
        return [token.lemma_ for token in doc]  # Return lemmas.
    return tokens  # Without spaCy, keep tokens unchanged.


def preprocess_text(text):  # Full pipeline for training and inference.
    cleaned = clean_text(text)  # Cleanup and normalization.
    tokens = tokenize_text(cleaned)  # Tokenization.
    tokens = remove_stopwords(tokens)  # Stopword filtering.
    lemmas = lemmatize_tokens(tokens)  # Lemmatization.
    return " ".join(lemmas)  # Return vectorizer-ready string.
