# English text preprocessing for sentiment analysis

import re # Regular expressions for text cleaning
import unicodedata # Unicode normalization

from nltk.corpus import stopwords # Stopword list for English
from nltk.stem import WordNetLemmatizer # Lemmatizer for reducing words to their base form
from nltk import word_tokenize # Tokenizer for splitting text into words

# Attempt to load spaCy for improved tokenization and lemmatization.
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
# If spaCy is not available, fall back to NLTK-based processing.
except Exception:
    nlp = None


# Keep key sentiment negations even when removing stopwords.
_NEGATION_WORDS = {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely"}

try:
    _base_stopwords = set(stopwords.words("english"))
except LookupError:
    # Fallback if NLTK stopwords are not available in the environment.
    _base_stopwords = set()

STOPWORDS = _base_stopwords - _NEGATION_WORDS
lemmatizer = WordNetLemmatizer()

# Main preprocessing function that applies all steps in sequence.
def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower() # Normalize unicode and convert to lowercase
    text = re.sub(r"http\S+|www\.\S+", "", text) # Remove URLs
    text = re.sub(r"@\w+", "", text) # Remove mentions
    # Keep apostrophes to better preserve words like "don't".
    text = re.sub(r"[^a-z0-9\s']", " ", text) # Remove special characters
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    
    return text

# TOKENIZATION, STOPWORD REMOVAL, AND LEMMATIZATION
def tokenize_text(text):
    # Use spaCy tokenizer if available for better handling of contractions and punctuation.
    if nlp:
        return [t.text for t in nlp(text)]

    # Fallback to NLTK word_tokenize, which may require downloading the 'punkt' resource.
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback if punkt tokenizer resource is missing.
        return text.split()

# Remove stopwords but keep negations.
def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]

# Lemmatize tokens using spaCy if available, otherwise use NLTK's WordNetLemmatizer.
def lemmatize_tokens(tokens):
    if nlp:
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    return [lemmatizer.lemmatize(token) for token in tokens]

# Main preprocessing function that applies all steps in sequence.
def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(tokens)

    return " ".join(lemmas)