# Russian text preprocessing module for sentiment analysis.

import re  # Pattern-based text cleanup.
import unicodedata  # Unicode normalization helpers.

from nltk.corpus import stopwords  # Russian stopword list support.
from nltk import word_tokenize  # Fallback tokenizer.

# Try to load spaCy for better tokenization and lemmatization.
try:
    import spacy  # Import spaCy only if available.
    nlp = spacy.load("ru_core_news_sm")  # Load small Russian language model.
except (ImportError, OSError):  # Handle missing package/model.
    nlp = None  # Indicates spaCy is unavailable.


# Keep sentiment-relevant negation words when removing stopwords.
NEGATION_WORDS = {
    "не",  # Basic negation particle.
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

# Try loading Russian stopwords from NLTK corpus data.
try:
    base_stopwords = set(stopwords.words("russian"))  # Faster membership checks.
except LookupError:  # If corpus data is missing.
    base_stopwords = set()  # Safe fallback for runtime portability.

STOPWORDS = base_stopwords - NEGATION_WORDS  # Preserve negation polarity cues.


def clean_text(text):  # Clean and normalize raw input text.
    text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize and lowercase.
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs.
    text = re.sub(r"@\w+", "", text)  # Remove @mentions.
    text = re.sub(r"[^a-zа-яё0-9\s']", " ", text)  # Keep letters/digits/whitespace/apostrophes.
    text = re.sub(r"\s+", " ", text).strip()  # Collapse and trim whitespace.
    return text  # Return cleaned text string.


def tokenize_text(text):  # Tokenize cleaned text.
    if nlp:
        return [t.text for t in nlp(text)]  # Use spaCy tokens when available.

    try:
        return word_tokenize(text, language = "russian")  # Use NLTK punkt tokenizer.
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


if __name__ == "__main__":  # Small CLI demo for manual checks.
    import argparse  # Imported only in CLI execution path.

    parser = argparse.ArgumentParser(description = "Russian Text Preprocessor")  # Build parser.
    parser.add_argument(
        "--text",  # CLI flag name.
        default = "Мне очень нравится этот продукт, он работает идеально!",  # Sample fallback sentence.
        help = "input text",  # Help text shown in usage.
    )
    args = parser.parse_args()  # Parse provided command-line args.

    print(preprocess_text(args.text))  # Print preprocessed output.