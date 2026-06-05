"""Morphological analysis for detecting grammatical errors."""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class MorphologicalFeatures:
    """Morphological features of a word."""
    word: str
    pos: str = ""  # Part of speech
    case: Optional[str] = None  # nom, acc, gen, dat
    gender: Optional[str] = None  # m, f, n
    number: Optional[str] = None  # sg, pl
    tense: Optional[str] = None  # past, present, future
    aspect: Optional[str] = None  # perfective, imperfective
    mood: Optional[str] = None  # indicative, subjunctive, conditional
    voice: Optional[str] = None  # active, passive
    person: Optional[str] = None  # 1, 2, 3
    language: str = "unknown"


class GermanMorphologyAnalyzer:
    """Analyze German morphological features."""

    # German articles and their features
    DEFINITE_ARTICLES = {
        "der": {"gender": "m", "case": "nom"},
        "den": {"gender": "m", "case": "acc"},
        "des": {"gender": "m", "case": "gen"},
        "dem": {"gender": "m", "case": "dat"},
        "die": {"gender": "f", "case": "nom"},
        "der": {"gender": "f", "case": "gen"},
        "der": {"gender": "f", "case": "dat"},
        "das": {"gender": "n", "case": "nom"},
        "das": {"gender": "n", "case": "acc"},
        "des": {"gender": "n", "case": "gen"},
        "dem": {"gender": "n", "case": "dat"},
    }

    def extract_gender_from_article(self, word: str) -> Optional[str]:
        """Infer gender from German article."""
        articles = {"der": "m", "die": "f", "das": "n"}
        return articles.get(word.lower())

    def has_capitalization(self, word: str) -> bool:
        """Check if word is capitalized (German nouns)."""
        return word[0].isupper() if word else False

    def detect_case_marker(self, word: str) -> Optional[str]:
        """Detect German case endings in adjectives/articles."""
        # Simplified case detection
        if word.endswith(("en", "em", "es", "er")):
            return "oblique"  # Accusative, dative, genitive
        if word.endswith(("e", "er", "es")):
            return "agreement_marker"
        return None

    def analyze_word(self, word: str) -> MorphologicalFeatures:
        """Analyze a German word."""
        features = MorphologicalFeatures(word=word, language="de")

        # Simple heuristics
        features.gender = self.extract_gender_from_article(word)

        if self.has_capitalization(word):
            features.pos = "NOUN"

        return features


class RussianMorphologyAnalyzer:
    """Analyze Russian morphological features."""

    # Russian case endings for nouns (simplified)
    CASE_ENDINGS = {
        "nominative": ["а", "о", "е", "и", "ы"],
        "accusative": ["у", "ю", "а", "о"],
        "genitive": ["ы", "и", "а", "я"],
        "dative": ["е", "и", "ю"],
        "instrumental": ["ом", "ем", "ой", "ей"],
        "locative": ["е", "и"],
    }

    def detect_case_ending(self, word: str) -> Optional[str]:
        """Detect Russian case from word ending."""
        word_lower = word.lower()
        for case, endings in self.CASE_ENDINGS.items():
            for ending in endings:
                if word_lower.endswith(ending):
                    return case
        return None

    def has_stress_mark(self, word: str) -> bool:
        """Check for Russian stress marks (usually ударение)."""
        return "́" in word or "̀" in word

    def analyze_word(self, word: str) -> MorphologicalFeatures:
        """Analyze a Russian word."""
        features = MorphologicalFeatures(word=word, language="ru")
        features.case = self.detect_case_ending(word)
        return features


class EnglishMorphologyAnalyzer:
    """Analyze English morphological features (simpler)."""

    def analyze_word(self, word: str) -> MorphologicalFeatures:
        """Analyze an English word."""
        features = MorphologicalFeatures(word=word, language="en")

        # Simple heuristics
        if word.lower() in ["a", "the", "an"]:
            features.pos = "DET"
        elif word.lower() in ["is", "are", "was", "were"]:
            features.pos = "VERB"

        return features


class MorphologicalComparator:
    """Compare morphological features between source and target."""

    ANALYZER_MAP = {
        "de": GermanMorphologyAnalyzer(),
        "ru": RussianMorphologyAnalyzer(),
        "en": EnglishMorphologyAnalyzer(),
    }

    def get_analyzer(self, lang: str) -> Optional:
        """Get analyzer for language."""
        return self.ANALYZER_MAP.get(lang.lower())

    def detect_morphological_loss(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[Tuple[str, str, str]]:
        """
        Detect morphological features lost in translation.

        Returns: List of (source_word, target_word, lost_feature)
        """
        losses = []

        src_analyzer = self.get_analyzer(source_lang)
        tgt_analyzer = self.get_analyzer(target_lang)

        if not src_analyzer or not tgt_analyzer:
            return losses

        # Simple alignment: assume words in same position correspond
        for src_tok, tgt_tok in zip(source_tokens, target_tokens):
            src_features = src_analyzer.analyze_word(src_tok)
            tgt_features = tgt_analyzer.analyze_word(tgt_tok)

            # Detect loss of case/gender/number
            if src_features.case and not tgt_features.case:
                losses.append((src_tok, tgt_tok, f"case:{src_features.case}"))
            if src_features.gender and not tgt_features.gender:
                losses.append((src_tok, tgt_tok, f"gender:{src_features.gender}"))

        return losses

    def detect_agreement_errors(
        self,
        tokens: List[str],
        lang: str,
    ) -> List[Tuple[int, int, str]]:
        """
        Detect agreement errors (subject-verb, noun-adj).

        Returns: List of (token_idx1, token_idx2, error_type)
        """
        errors = []
        analyzer = self.get_analyzer(lang)

        if not analyzer:
            return errors

        # Simplified: check adjacent words for agreement
        for i in range(len(tokens) - 1):
            tok1 = tokens[i]
            tok2 = tokens[i + 1]

            feat1 = analyzer.analyze_word(tok1)
            feat2 = analyzer.analyze_word(tok2)

            # Look for case mismatches (simplified heuristic)
            if feat1.case and feat2.case and feat1.case != feat2.case:
                # Might be agreement error
                if feat1.pos in ["NOUN", "ADJ"] and feat2.pos in ["NOUN", "ADJ"]:
                    errors.append((i, i + 1, "case_mismatch"))

        return errors
