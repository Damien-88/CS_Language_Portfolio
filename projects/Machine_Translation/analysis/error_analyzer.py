"""Main error analysis engine."""

from typing import List, Optional, Tuple
import re
from difflib import SequenceMatcher

from analysis.error_types import (
    ErrorCategory,
    ErrorSpan,
    SentenceLinguisticAnalysis,
    BatchLinguisticAnalysis,
)
from analysis.morphological_checker import MorphologicalComparator


class LexicalAnalyzer:
    """Analyze lexical choices and vocabulary mismatches."""

    # Common content words that should be preserved
    CRITICAL_CONTENT_WORDS = {"not", "no", "never", "always", "only"}

    def detect_oov_untranslated(
        self, source_tokens: List[str], target_tokens: List[str], alignment_ratio: float = 0.5
    ) -> List[Tuple[str, str]]:
        """
        Detect out-of-vocabulary (OOV) or untranslated words.

        Returns: List of (source_word, target_word)
        """
        oov_pairs = []

        # Heuristic: if source has word that looks like it wasn't translated
        for src_tok in source_tokens:
            # Check if source word appears verbatim in target (bad sign for translation)
            if src_tok.lower() in [t.lower() for t in target_tokens]:
                oov_pairs.append((src_tok, src_tok))

        return oov_pairs

    def detect_negation_loss(self, source: str, target: str) -> bool:
        """Detect if negation was lost in translation."""
        source_has_negation = any(
            word in source.lower() for word in ["not", "no", "none", "never", "kein", "не"]
        )
        target_has_negation = any(
            word in target.lower() for word in ["not", "no", "none", "never", "kein", "не"]
        )

        return source_has_negation and not target_has_negation


class WordOrderAnalyzer:
    """Analyze word order changes and syntactic reordering."""

    def compute_word_order_similarity(
        self, source_tokens: List[str], target_tokens: List[str]
    ) -> float:
        """
        Compute similarity of word order using sequence matching.

        Returns: score between 0 and 1 (1 = identical order)
        """
        # Simple: check if tokens appear in similar order
        src_positions = {tok: i for i, tok in enumerate(source_tokens)}
        tgt_positions = {tok: i for i, tok in enumerate(target_tokens)}

        # Find common tokens
        common_tokens = set(source_tokens) & set(target_tokens)

        if not common_tokens:
            return 0.0

        # Check if common tokens maintain relative order
        src_order = sorted([src_positions[tok] for tok in common_tokens])
        tgt_order = sorted([tgt_positions[tok] for tok in common_tokens])

        # Use sequence matcher
        matcher = SequenceMatcher(None, src_order, tgt_order)
        return matcher.ratio()

    def detect_svo_changes(self, source_tokens: List[str], target_tokens: List[str]) -> str:
        """
        Detect subject-verb-object (SVO) vs. SOV changes.

        Returns: description of word order change
        """
        # Simplified: detect if subject/object positions swapped
        # This would require POS tagging in production
        return "potential_word_order_shift"


class SemanticAnalyzer:
    """Analyze semantic drift and meaning preservation."""

    # Antonym pairs (simplified)
    ANTONYM_PAIRS = [
        ("good", "bad"),
        ("yes", "no"),
        ("high", "low"),
        ("big", "small"),
        ("fast", "slow"),
        ("hot", "cold"),
    ]

    def compute_semantic_similarity(self, source: str, target: str) -> float:
        """
        Compute semantic similarity between source and target.

        Simplified: character-level n-gram overlap.
        Returns: score between 0 and 1
        """
        def get_ngrams(text: str, n: int = 3) -> set:
            text = text.lower()
            return {text[i : i + n] for i in range(len(text) - n + 1)}

        src_ngrams = get_ngrams(source)
        tgt_ngrams = get_ngrams(target)

        if not src_ngrams or not tgt_ngrams:
            return 0.0

        intersection = len(src_ngrams & tgt_ngrams)
        union = len(src_ngrams | tgt_ngrams)

        return intersection / union if union > 0 else 0.0

    def detect_negation_flip(self, source: str, target: str) -> bool:
        """Detect if meaning was flipped (negation/antonym change)."""
        source_has_negation = any(
            word in source.lower() for word in ["not", "no", "none", "never", "kein", "nicht", "не"]
        )
        target_has_negation = any(
            word in target.lower() for word in ["not", "no", "none", "never", "kein", "nicht", "не"]
        )

        # If one has negation and other doesn't, meaning flipped
        return source_has_negation != target_has_negation


class GermanCompoundAnalyzer:
    """Analyze German compound-related issues."""

    def detect_compound_breakdown(
        self, source_tokens: List[str], target_tokens: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Detect potential compound breakdown issues.

        Returns: List of (source_compound, target_tokens)
        """
        breakdowns = []

        # Heuristic: long German words might be compounds
        for src_tok in source_tokens:
            if len(src_tok) > 8 and src_tok.lower().isalpha():  # Likely compound
                # Check if it was broken into multiple words in target
                # This is simplified; ideally use actual compound decomposer
                breakdowns.append((src_tok, " ".join(target_tokens)))

        return breakdowns


class TranslationErrorAnalyzer:
    """Main error analyzer combining all sub-analyzers."""

    def __init__(self):
        self.lexical = LexicalAnalyzer()
        self.word_order = WordOrderAnalyzer()
        self.semantic = SemanticAnalyzer()
        self.morphological = MorphologicalComparator()
        self.compound = GermanCompoundAnalyzer()

    def analyze(
        self,
        source: str,
        target: str,
        reference: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> SentenceLinguisticAnalysis:
        """
        Perform complete linguistic analysis.

        Args:
            source: Source text
            target: Translated text
            reference: Optional reference translation
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            SentenceLinguisticAnalysis with detected errors
        """
        analysis = SentenceLinguisticAnalysis(
            source_text=source,
            target_text=target,
            reference_text=reference,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Tokenize simply (space-based)
        source_tokens = source.split()
        target_tokens = target.split()

        # Run all analyzers
        self._detect_lexical_errors(analysis, source, target, source_tokens, target_tokens)
        self._detect_morphological_errors(analysis, source_tokens, target_tokens, source_lang, target_lang)
        self._detect_word_order_errors(analysis, source_tokens, target_tokens)
        self._detect_semantic_errors(analysis, source, target)
        self._detect_compound_errors(analysis, source_tokens, target_tokens, source_lang, target_lang)

        # Compute overall quality
        analysis.overall_quality = self._rate_overall_quality(analysis)

        # Summarize error categories
        analysis.error_categories = {}
        for error in analysis.errors:
            cat = error.category.value
            analysis.error_categories[cat] = analysis.error_categories.get(cat, 0) + 1

        return analysis

    def _detect_lexical_errors(
        self,
        analysis: SentenceLinguisticAnalysis,
        source: str,
        target: str,
        source_tokens: List[str],
        target_tokens: List[str],
    ) -> None:
        """Detect lexical mismatches."""
        # OOV/untranslated words
        oov_pairs = self.lexical.detect_oov_untranslated(source_tokens, target_tokens)
        for src, tgt in oov_pairs:
            analysis.errors.append(
                ErrorSpan(
                    category=ErrorCategory.OOV_UNTRANSLATED,
                    source_span=src,
                    target_span=tgt,
                    explanation=f"Word '{src}' appears untranslated or OOV in target",
                    severity="high",
                    confidence=0.8,
                )
            )

        # Negation loss
        if self.lexical.detect_negation_loss(source, target):
            analysis.errors.append(
                ErrorSpan(
                    category=ErrorCategory.LEXICAL_MISMATCH,
                    source_span=source,
                    target_span=target,
                    explanation="Negation lost in translation",
                    severity="high",
                    confidence=0.75,
                )
            )

    def _detect_morphological_errors(
        self,
        analysis: SentenceLinguisticAnalysis,
        source_tokens: List[str],
        target_tokens: List[str],
        source_lang: str,
        target_lang: str,
    ) -> None:
        """Detect morphological errors."""
        losses = self.morphological.detect_morphological_loss(
            source_tokens, target_tokens, source_lang, target_lang
        )

        for src_tok, tgt_tok, feature_loss in losses:
            analysis.errors.append(
                ErrorSpan(
                    category=ErrorCategory.MORPHOLOGICAL_LOSS,
                    source_span=src_tok,
                    target_span=tgt_tok,
                    explanation=f"Morphological feature lost: {feature_loss}",
                    severity="medium",
                    confidence=0.6,
                )
            )

        # Agreement errors
        if target_lang in ["de", "ru"]:
            agreement_errors = self.morphological.detect_agreement_errors(target_tokens, target_lang)
            for idx1, idx2, error_type in agreement_errors:
                if idx1 < len(target_tokens) and idx2 < len(target_tokens):
                    analysis.errors.append(
                        ErrorSpan(
                            category=ErrorCategory.AGREEMENT_ERROR,
                            source_span=f"{source_tokens[idx1]} {source_tokens[idx2]}" if idx2 < len(source_tokens) else "",
                            target_span=f"{target_tokens[idx1]} {target_tokens[idx2]}",
                            explanation=f"{error_type} in translation",
                            severity="medium",
                            confidence=0.65,
                        )
                    )

    def _detect_word_order_errors(
        self,
        analysis: SentenceLinguisticAnalysis,
        source_tokens: List[str],
        target_tokens: List[str],
    ) -> None:
        """Detect word order changes."""
        similarity = self.word_order.compute_word_order_similarity(source_tokens, target_tokens)

        if similarity < 0.6:  # Significant reordering
            analysis.errors.append(
                ErrorSpan(
                    category=ErrorCategory.WORD_ORDER_SHIFT,
                    source_span=" ".join(source_tokens),
                    target_span=" ".join(target_tokens),
                    explanation=f"Word order significantly reordered (similarity: {similarity:.2f})",
                    severity="low",
                    confidence=0.7,
                )
            )

    def _detect_semantic_errors(
        self,
        analysis: SentenceLinguisticAnalysis,
        source: str,
        target: str,
    ) -> None:
        """Detect semantic drift and meaning changes."""
        # Negation flip
        if self.semantic.detect_negation_flip(source, target):
            analysis.errors.append(
                ErrorSpan(
                    category=ErrorCategory.SEMANTIC_DRIFT,
                    source_span=source,
                    target_span=target,
                    explanation="Semantic meaning flipped (negation changed)",
                    severity="high",
                    confidence=0.8,
                )
            )

        # General semantic similarity
        similarity = self.semantic.compute_semantic_similarity(source, target)
        if similarity < 0.3:  # Low semantic overlap
            analysis.errors.append(
                ErrorSpan(
                    category=ErrorCategory.SEMANTIC_DRIFT,
                    source_span=source,
                    target_span=target,
                    explanation=f"Low semantic similarity ({similarity:.2f})",
                    severity="medium",
                    confidence=0.65,
                )
            )

    def _detect_compound_errors(
        self,
        analysis: SentenceLinguisticAnalysis,
        source_tokens: List[str],
        target_tokens: List[str],
        source_lang: str,
        target_lang: str,
    ) -> None:
        """Detect compound-related issues."""
        if source_lang.lower() == "de" or target_lang.lower() == "de":
            breakdowns = self.compound.detect_compound_breakdown(source_tokens, target_tokens)

            for src_compound, tgt_expansion in breakdowns[:3]:  # Limit to top 3
                analysis.errors.append(
                    ErrorSpan(
                        category=ErrorCategory.COMPOUND_BREAKDOWN,
                        source_span=src_compound,
                        target_span=tgt_expansion,
                        explanation=f"Compound '{src_compound}' may have decomposed incorrectly",
                        severity="low",
                        confidence=0.5,
                    )
                )

    def _rate_overall_quality(self, analysis: SentenceLinguisticAnalysis) -> str:
        """Rate overall translation quality based on errors."""
        if not analysis.errors:
            return "excellent"

        high_severity = sum(1 for e in analysis.errors if e.severity == "high")
        medium_severity = sum(1 for e in analysis.errors if e.severity == "medium")

        if high_severity >= 2:
            return "poor"
        elif high_severity >= 1 or medium_severity >= 3:
            return "fair"
        elif medium_severity >= 1:
            return "good"
        else:
            return "excellent"

    def analyze_batch(
        self,
        sources: List[str],
        targets: List[str],
        references: Optional[List[str]] = None,
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> BatchLinguisticAnalysis:
        """
        Analyze multiple sentence pairs.

        Args:
            sources: List of source texts
            targets: List of translated texts
            references: Optional list of reference translations
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            BatchLinguisticAnalysis with all results
        """
        batch = BatchLinguisticAnalysis(source_lang=source_lang, target_lang=target_lang)

        for i, (src, tgt) in enumerate(zip(sources, targets)):
            ref = references[i] if references and i < len(references) else None
            analysis = self.analyze(src, tgt, ref, source_lang, target_lang)
            batch.analyses.append(analysis)

        return batch