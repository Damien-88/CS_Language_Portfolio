# Multilingual text preprocessing module for sentiment analysis (English, German, Russian).

import re  # Pattern-based text cleanup.
import unicodedata  # Unicode normalization helpers.

from nltk.corpus import stopwords  # Language stopword lists.
from nltk.stem import WordNetLemmatizer  # English lemmatizer fallback.
from nltk import word_tokenize  # Fallback tokenizer.


class MultilingualPreprocessor:
    """
    Language-aware text preprocessor supporting English, German, and Russian.

    Loads the appropriate spaCy model, stopword list, negation words, and
    character filter for the chosen language. spaCy is used when available;
    NLTK provides fallbacks for tokenization and lemmatization.

    Usage:
        preprocessor = MultilingualPreprocessor("german")
        preprocessor.preprocess("Ich liebe dieses Produkt!")
    """

    # Per-language configuration
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

    def __init__(self, language: str = "english"):
        if language not in self.LANGUAGE_CONFIG:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Choose from: {list(self.LANGUAGE_CONFIG)}"
            )

        self.language = language
        config = self.LANGUAGE_CONFIG[language]

        self.nltk_language = config["nltk_language"]
        self.char_re = re.compile(config["char_pattern"])

        # Load spaCy model; fall back gracefully if unavailable.
        try:
            import spacy
            self.nlp = spacy.load(config["spacy_model"])
        except Exception:
            self.nlp = None

        # Load stopwords and subtract negations so polarity cues are preserved.
        try:
            base_sw = set(stopwords.words(config["stopwords_corpus"]))
        except LookupError:
            base_sw = set()
        self.stopwords = base_sw - config["negation_words"]

        # English-only NLTK lemmatizer fallback.
        self.lemmatizer = WordNetLemmatizer() if language == "english" else None

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:  # Clean and normalize raw input text.
        text = unicodedata.normalize("NFKC", str(text)).lower()  # Normalize and lowercase.
        text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs.
        text = re.sub(r"@\w+", "", text)  # Remove @mentions.
        text = self.char_re.sub(" ", text)  # Keep language-specific letters and symbols.
        text = re.sub(r"\s+", " ", text).strip()  # Collapse and trim whitespace.
        return text  # Return cleaned text string.

    def tokenize(self, text: str) -> list:  # Tokenize cleaned text.
        if self.nlp:
            return [t.text for t in self.nlp(text)]  # Use spaCy tokens when available.
        try:
            return word_tokenize(text, language = self.nltk_language)  # Use NLTK punkt tokenizer.
        except LookupError:
            return text.split()  # Fallback if punkt resource is missing.

    def remove_stopwords(self, tokens: list) -> list:  # Remove low-information tokens.
        return [t for t in tokens if t not in self.stopwords]  # Keep sentiment-carrying terms.

    def lemmatize(self, tokens: list) -> list:  # Reduce inflected forms to base forms.
        if self.nlp:
            doc = self.nlp(" ".join(tokens))  # Create a spaCy doc from tokens.
            return [token.lemma_ for token in doc]  # Return lemmas.
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]  # NLTK fallback for English.
        return tokens  # Without spaCy, keep tokens unchanged.

    def preprocess(self, text: str) -> str:  # Full pipeline for training and inference.
        cleaned = self.clean_text(text)  # Cleanup and normalization.
        tokens = self.tokenize(cleaned)  # Tokenization.
        tokens = self.remove_stopwords(tokens)  # Stopword filtering.
        lemmas = self.lemmatize(tokens)  # Lemmatization.
        return " ".join(lemmas)  # Return vectorizer-ready string.


# ------------------------------------------------------------------
# Module-level convenience function
# Instances are cached so spaCy models are only loaded once per language.
# ------------------------------------------------------------------

cache: dict = {}


def preprocess_text(text: str, language: str = "english") -> str:
    """
    Preprocess text for the given language using a cached preprocessor instance.

    Args:
        text:     Raw input string.
        language: One of "english", "german", "russian".

    Returns:
        Cleaned, tokenized, stopword-filtered, lemmatized string.
    """
    if language not in cache:
        cache[language] = MultilingualPreprocessor(language)
    return cache[language].preprocess(text)