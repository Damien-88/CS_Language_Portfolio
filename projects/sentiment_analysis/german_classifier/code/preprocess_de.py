# German text preprocessing for sentiment analysis

# Import Python regex utilities for pattern-based text cleanup.
import re
# Import Unicode helpers so text is normalized consistently.
import unicodedata

# Import NLTK's German stopword list.
from nltk.corpus import stopwords
# Import NLTK tokenizer used when spaCy is unavailable.
from nltk import word_tokenize

# Try to use spaCy for higher-quality tokenization and lemmatization.
try:
    # Import spaCy only if it exists in the runtime environment.
    import spacy
    # Load the small German spaCy pipeline.
    nlp = spacy.load("de_core_news_sm")
# If spaCy or its model is unavailable, use fallback logic.
except Exception:
    # Mark spaCy pipeline as unavailable.
    nlp = None


# Keep sentiment-relevant negations when removing stopwords.
_NEGATION_WORDS = {
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

# Attempt to load German stopwords from local NLTK resources.
try:
    # Convert to a set for fast O(1) membership checks.
    _base_stopwords = set(stopwords.words("german"))
# Handle environments where NLTK corpora are not downloaded.
except LookupError:
    # Fallback if NLTK stopwords are unavailable in the runtime environment.
    # Use an empty set so preprocessing still runs without crashing.
    _base_stopwords = set()

# Remove negation terms from stopwords so polarity cues are retained.
STOPWORDS = _base_stopwords - _NEGATION_WORDS


def clean_text(text):  # Normalize and clean raw German input text.
    # Normalize Unicode forms, cast to string, and lowercase for consistency.
    text = unicodedata.normalize("NFKC", str(text)).lower()
    # Remove URL patterns like http://..., https://..., and www....
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove social mentions like @username.
    text = re.sub(r"@\w+", "", text)
    # Keep apostrophes and German letters for better sentiment context.
    # Replace disallowed characters with spaces.
    text = re.sub(r"[^a-z0-9\säöüß']", " ", text)
    # Collapse repeated whitespace and trim leading/trailing spaces.
    text = re.sub(r"\s+", " ", text).strip()
    # Return cleaned text ready for tokenization.
    return text


def tokenize_text(text):  # Split cleaned text into tokens.
    # Prefer spaCy tokenizer when a model is available.
    if nlp:
        # Return raw token strings from spaCy token objects.
        return [t.text for t in nlp(text)]

    # Attempt NLTK tokenization as secondary strategy.
    try:
        # Use NLTK tokenizer configured for German language rules.
        return word_tokenize(text, language = "german")
    # Handle missing punkt tokenizer resources.
    except LookupError:
        # Fallback if punkt tokenizer resource is missing.
        # Use whitespace tokenization as final fallback.
        return text.split()


def remove_stopwords(tokens):  # Drop low-information tokens.
    # Keep only tokens that are not part of the stopword set.
    return [token for token in tokens if token not in STOPWORDS]


def lemmatize_tokens(tokens):  # Convert tokens to lemma/base forms.
    # Use spaCy lemmatizer when available.
    if nlp:
        # Recreate text from tokens so spaCy can process it.
        doc = nlp(" ".join(tokens))
        # Return lemma string for each token.
        return [token.lemma_ for token in doc]
    # If no lemmatizer is available, return original tokens unchanged.
    return tokens


def preprocess_text(text):  # Run the full preprocessing pipeline.
    # Step 1: normalize and clean raw input text.
    cleaned = clean_text(text)
    # Step 2: tokenize the cleaned text.
    tokens = tokenize_text(cleaned)
    # Step 3: remove stopwords while preserving key negations.
    tokens = remove_stopwords(tokens)
    # Step 4: lemmatize tokens for normalization of inflections.
    lemmas = lemmatize_tokens(tokens)
    # Return a vectorizer-ready string.
    return " ".join(lemmas)
