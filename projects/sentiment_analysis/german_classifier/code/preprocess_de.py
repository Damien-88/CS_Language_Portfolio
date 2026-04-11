# German text preprocessing for sentiment analysis

import re
import unicodedata

from nltk.corpus import stopwords
from nltk import word_tokenize

try:
    import spacy
    nlp = spacy.load("de_core_news_sm")
except Exception:
    nlp = None


# Keep sentiment-relevant negations when removing stopwords.
_NEGATION_WORDS = {
    "nicht",
    "kein",
    "keine",
    "keinen",
    "keinem",
    "keiner",
    "keines",
    "nie",
    "niemals",
    "weder",
    "noch"
}

try:
    _base_stopwords = set(stopwords.words("german"))
except LookupError:
    # Fallback if NLTK stopwords are unavailable in the runtime environment.
    _base_stopwords = set()

STOPWORDS = _base_stopwords - _NEGATION_WORDS


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    # Keep apostrophes and German letters for better sentiment context.
    text = re.sub(r"[^a-z0-9\säöüß']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    if nlp:
        return [t.text for t in nlp(text)]

    try:
        return word_tokenize(text, language = "german")
    except LookupError:
        # Fallback if punkt tokenizer resource is missing.
        return text.split()


def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]


def lemmatize_tokens(tokens):
    if nlp:
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    return tokens


def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(tokens)
    return " ".join(lemmas)
