"""German compound analysis module - Integration point for GermanCompoundDecomposer."""

from typing import Optional, List, Tuple, Any
from dataclasses import dataclass, field
import json

from analysis.error_types import ErrorSpan, ErrorCategory


@dataclass
class CompoundAnalysis:
    """Analysis of a single compound word."""
    original: str
    decomposition: List[str]
    is_valid: bool = True
    confidence: float = 1.0
    morphological_info: dict = field(default_factory=dict)


@dataclass
class CompoundTranslationAnalysis:
    """Analysis of compound handling in a translation."""
    source_compound: str
    source_decomposition: List[str]
    target_text: str
    target_compounds: List[str]
    target_decompositions: List[List[str]]
    compound_preserved: bool
    alignment_score: float
    notes: str = ""


class CompoundProcessor:
    """
    Integration layer for German compound analysis.

    This class wraps an external GermanCompoundDecomposer and integrates it
    into the translation error analysis pipeline.
    """

    def __init__(self, decomposer: Optional[Any] = None):
        """
        Initialize compound processor.

        Args:
            decomposer: Instance of GermanCompoundDecomposer (can be injected later)
        """
        self.decomposer = decomposer

    def set_decomposer(self, decomposer: Any) -> None:
        """Set or replace the decomposer."""
        self.decomposer = decomposer

    def decompose_text(self, text: str) -> List[CompoundAnalysis]:
        """
        Decompose all compounds in German text.

        Args:
            text: German text

        Returns:
            List of CompoundAnalysis for each word
        """
        results = []

        if not self.decomposer:
            return results

        tokens = text.split()

        for token in tokens:
            try:
                # Try to decompose using the provided decomposer
                decomposition = self.decomposer.decompose(token)

                # Create analysis object
                is_compound = len(decomposition) > 1
                analysis = CompoundAnalysis(
                    original=token,
                    decomposition=decomposition,
                    is_valid=True,
                    confidence=0.9 if is_compound else 1.0,
                )
                results.append(analysis)

            except Exception as e:
                # If decomposer fails, mark as non-decomposable
                results.append(
                    CompoundAnalysis(
                        original=token,
                        decomposition=[token],
                        is_valid=False,
                        confidence=0.0,
                        morphological_info={"error": str(e)},
                    )
                )

        return results

    def analyze_compound_preservation(
        self, source_de: str, target_de: str
    ) -> CompoundTranslationAnalysis:
        """
        Analyze whether compounds were preserved through translation.

        Args:
            source_de: Source German text
            target_de: Target German text (from translation)

        Returns:
            CompoundTranslationAnalysis showing preservation
        """
        # Decompose source
        source_compounds = self.decompose_text(source_de)
        source_compounds_only = [c for c in source_compounds if len(c.decomposition) > 1]

        if not source_compounds_only:
            return CompoundTranslationAnalysis(
                source_compound="",
                source_decomposition=[],
                target_text=target_de,
                target_compounds=[],
                target_decompositions=[],
                compound_preserved=True,
                alignment_score=1.0,
                notes="No compounds in source",
            )

        # Focus on first compound for analysis
        main_compound = source_compounds_only[0]

        # Decompose target
        target_compounds = self.decompose_text(target_de)

        # Check if target has complex words
        target_complex = [c for c in target_compounds if len(c.decomposition) > 1]

        # Simple preservation check: are there compounds in target?
        preserved = len(target_complex) > 0

        # Alignment score: how similar are decompositions?
        alignment = self._compute_alignment_score(
            main_compound.decomposition, [c.decomposition for c in target_complex]
        )

        return CompoundTranslationAnalysis(
            source_compound=main_compound.original,
            source_decomposition=main_compound.decomposition,
            target_text=target_de,
            target_compounds=[c.original for c in target_complex],
            target_decompositions=[c.decomposition for c in target_complex],
            compound_preserved=preserved,
            alignment_score=alignment,
            notes=f"Found {len(target_complex)} compounds in target",
        )

    def _compute_alignment_score(
        self, source_decomposition: List[str], target_decompositions: List[List[str]]
    ) -> float:
        """
        Compute how well target decompositions align with source.

        Simple heuristic: check semantic overlap of components.

        Args:
            source_decomposition: Components of source compound
            target_decompositions: List of target decompositions

        Returns:
            Score between 0 and 1 (1 = perfect alignment)
        """
        if not target_decompositions:
            return 0.0

        best_score = 0.0

        for target_decomp in target_decompositions:
            # Simple: compare component count and string similarity
            if len(source_decomposition) == len(target_decomp):
                # Check character overlap
                source_str = "".join(source_decomposition)
                target_str = "".join(target_decomp)

                # Compute Jaccard similarity
                source_chars = set(source_str.lower())
                target_chars = set(target_str.lower())

                if source_chars or target_chars:
                    intersection = len(source_chars & target_chars)
                    union = len(source_chars | target_chars)
                    score = intersection / union
                    best_score = max(best_score, score)

        return best_score

    def create_error_span(
        self, source_de: str, target_de: str, reference_de: Optional[str] = None
    ) -> Optional[ErrorSpan]:
        """
        Create an ErrorSpan if compound issues are detected.

        Args:
            source_de: Source German text
            target_de: Translated German text
            reference_de: Optional reference German text

        Returns:
            ErrorSpan if issues found, None otherwise
        """
        analysis = self.analyze_compound_preservation(source_de, target_de)

        if analysis.alignment_score < 0.5 or not analysis.compound_preserved:
            return ErrorSpan(
                category=ErrorCategory.COMPOUND_BREAKDOWN,
                source_span=analysis.source_compound,
                target_span=" ".join(analysis.target_compounds) or target_de,
                reference_span=reference_de,
                explanation=(
                    f"Compound '{analysis.source_compound}' decomposed as "
                    f"{analysis.source_decomposition}. Target alignment score: "
                    f"{analysis.alignment_score:.2f}. {analysis.notes}"
                ),
                severity="medium" if analysis.alignment_score > 0.3 else "high",
                confidence=0.7,
            )

        return None

    def to_dict(self, analysis: CompoundTranslationAnalysis) -> dict:
        """Convert analysis to dictionary."""
        return {
            "source_compound": analysis.source_compound,
            "source_decomposition": analysis.source_decomposition,
            "target_text": analysis.target_text,
            "target_compounds": analysis.target_compounds,
            "target_decompositions": analysis.target_decompositions,
            "compound_preserved": analysis.compound_preserved,
            "alignment_score": analysis.alignment_score,
            "notes": analysis.notes,
        }

    def to_json(self, analysis: CompoundTranslationAnalysis, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(analysis), indent=indent, ensure_ascii=False)