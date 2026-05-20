"""Package exports for the Russian morphology analyzer API."""

from .analyzer import analyze_batch, analyze_sentence, analyze_word

__all__ = ["analyze_word", "analyze_sentence", "analyze_batch"]