"""Public API re-export for the Russian morphology analyzer package."""

from code.analyzer import analyze_batch, analyze_sentence, analyze_word

__all__ = ["analyze_word", "analyze_sentence", "analyze_batch"]