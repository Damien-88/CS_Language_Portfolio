"""Russian-aware tokenization helpers and lexical token filters."""

import re

try:
    from .utils import normalize_text
except ImportError:
    from utils import normalize_text

# Regular expression patterns for tokenization and word validation.
TOKEN_PATTERN = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*|\d+|[^\w\s]", re.UNICODE)
WORD_PATTERN = re.compile(r"^[А-Яа-яЁё-]+$", re.UNICODE)

def get_token_type(token):
    # Classify token as "word", "number", or "punctuation" based on patterns.
    if WORD_PATTERN.match(token):
        return "word"
    elif token.isdigit():
        return "number"
    else:
        return "punctuation"

def tokenize(text, keep_punctuation = False):
    # Split text into tokens and optionally keep punctuation tokens.
    normalized = normalize_text(text, lowercase = False)
    raw_tokens = TOKEN_PATTERN.findall(normalized)

    # If keeping punctuation, return dicts with token and type; otherwise, filter to words only.
    if keep_punctuation:
        return [
            {"token": t, "type": get_token_type(t)}
            for t in raw_tokens
        ]
    
    return [t for t in raw_tokens if is_word(t)]


def is_word(token):
    # Return True when token matches Cyrillic word pattern.
    if isinstance(token, dict):
        token = token.get("token", "")
    if not isinstance(token, str):
        return False
    return bool(WORD_PATTERN.match(token))


def filter_words(tokens):
    # Filter any iterable of tokens down to lexical words only.
    return [token for token in tokens if is_word(token)]