"""Core analysis pipeline for single-token, sentence, and batch processing."""

import argparse # For CLI argument parsing
import json # For pretty-printing analysis results in CLI mode
from pathlib import Path # For handling file paths in CLI mode
from typing import Iterable # For type hinting batch input as iterable of strings

# Import feature extraction and tokenization with package/script fallback.
try:
    from .feature_extractor import analyze_token, analyze_tokens
    from .tokenizer import tokenize
except ImportError:
    from feature_extractor import analyze_token, analyze_tokens
    from tokenizer import tokenize


def analyze_word(word):
    """Analyze one token and return normalized morphological features."""
    return analyze_token(word)


def analyze_sentence(text):
    """Tokenize a sentence and analyze only lexical tokens."""
    return analyze_tokens(tokenize(text, keep_punctuation = False))


def analyze_batch(texts):
    """Analyze multiple sentences in the same shape as input order."""
    return [analyze_sentence(text) for text in texts]


def main() -> None:
    """CLI entry point for ad-hoc analysis from terminal input."""
    parser = argparse.ArgumentParser(description = "Russian morphology analyzer")
    parser.add_argument("--word", help = "Analyze a single word")
    parser.add_argument("--text", help = "Analyze a full sentence")
    parser.add_argument("--file", type = Path, help = "Analyze each line in a UTF-8 text file")
    args = parser.parse_args()

    if args.word:
        print(json.dumps(analyze_word(args.word), ensure_ascii = False, indent = 2))
        return

    if args.text:
        print(json.dumps(analyze_sentence(args.text), ensure_ascii = False, indent = 2))
        return

    if args.file:
        # Ignore blank lines so each analyzed element is a real sentence/token sequence.
        lines = [line.strip() for line in args.file.read_text(encoding = "utf-8").splitlines() if line.strip()]
        print(json.dumps(analyze_batch(lines), ensure_ascii = False, indent = 2))
        return

    parser.error("Provide --word, --text, or --file.")


if __name__ == "__main__":
    main()