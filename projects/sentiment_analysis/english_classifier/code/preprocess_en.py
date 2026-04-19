# English text preprocessing module for sentiment analysis.

import re  # Pattern-based text cleanup.
import unicodedata  # Unicode normalization helpers.

from nltk.corpus import stopwords  # English stopword list support.
from nltk.stem import WordNetLemmatizer  # Fallback lemmatizer.
from nltk import word_tokenize  # Fallback tokenizer.

# Try to load spaCy for better tokenization and lemmatization.
try:
    import spacy  # Import spaCy only if available.
    nlp = spacy.load("en_core_web_sm")  # Load small English language model.
except Exception:  # Handle missing package/model.
    nlp = None  # Indicates spaCy is unavailable.


# Keep sentiment-relevant negation words when removing stopwords.
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely"}

try:
    base_stopwords = set(stopwords.words("english"))  # Faster membership checks.
except LookupError:
    base_stopwords = set()  # Safe fallback for runtime portability.

STOPWORDS = base_stopwords - NEGATION_WORDS  # Preserve negation polarity cues.
lemmatizer = WordNetLemmatizer()

def clean_text(text):  # Clean and normalize raw input text.
    text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize and lowercase.
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs.
    text = re.sub(r"@\w+", "", text)  # Remove @mentions.
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # Keep letters/digits/whitespace/apostrophes.
    text = re.sub(r"\s+", " ", text).strip()  # Collapse and trim whitespace.
    return text  # Return cleaned text string.

def tokenize_text(text):  # Tokenize cleaned text.
    if nlp:
        return [t.text for t in nlp(text)]  # Use spaCy tokens when available.

    try:
        return word_tokenize(text)  # Use NLTK punkt tokenizer.
    except LookupError:
        return text.split()  # Fallback if punkt resource is missing.

def remove_stopwords(tokens):  # Remove low-information tokens.
    return [token for token in tokens if token not in STOPWORDS]  # Keep sentiment-carrying terms.

def lemmatize_tokens(tokens):  # Reduce inflected forms to base forms.
    if nlp:
        doc = nlp(" ".join(tokens))  # Create a spaCy doc from tokens.
        return [token.lemma_ for token in doc]  # Return lemmas.
    return [lemmatizer.lemmatize(token) for token in tokens]  # NLTK fallback for English.

def preprocess_text(text):  # Full pipeline for training and inference.
    cleaned = clean_text(text)  # Cleanup and normalization.
    tokens = tokenize_text(cleaned)  # Tokenization.
    tokens = remove_stopwords(tokens)  # Stopword filtering.
    lemmas = lemmatize_tokens(tokens)  # Lemmatization.
    return " ".join(lemmas)  # Return vectorizer-ready string.