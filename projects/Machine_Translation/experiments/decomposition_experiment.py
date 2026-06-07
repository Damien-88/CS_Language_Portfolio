"""Experiments comparing morphological preprocessing effects on translation."""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import json

from config import TranslationConfig, PreprocessingConfig
from pipeline.translator_pipeline import TranslatorPipeline
from models.base import TranslationResult
from evaluation.metrics import BLEUMetric


@dataclass
class PreprocessingComparison:
    """Results comparing preprocessed vs. raw translation."""
    source_text: str
    raw_translation: TranslationResult
    preprocessed_translation: Optional[TranslationResult]
    bleu_raw: Optional[float] = None
    bleu_preprocessed: Optional[float] = None
    improvement: Optional[float] = None  # percentage change
    reference: Optional[str] = None
    decompositions: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source_text,
            "raw_translation": self.raw_translation.text,
            "preprocessed_translation": (
                self.preprocessed_translation.text
                if self.preprocessed_translation
                else None
            ),
            "bleu_raw": self.bleu_raw,
            "bleu_preprocessed": self.bleu_preprocessed,
            "improvement_percentage": self.improvement,
            "reference": self.reference,
            "decompositions": self.decompositions,
            "preprocessing_metadata": (
                self.raw_translation.metadata if self.raw_translation else {}
            ),
        }


class MorphologicalPreprocessingExperiment:
    """
    Experiment to evaluate the effect of morphological preprocessing.

    Compares translation quality with and without compound decomposition.
    """

    def __init__(self, decomposer: Optional[object] = None):
        """
        Initialize experiment.

        Args:
            decomposer: Optional external GermanCompoundDecomposer
        """
        self.decomposer = decomposer
        self.bleu_metric = BLEUMetric(max_n=4)

        # Create pipelines with and without preprocessing
        self.config_raw = TranslationConfig.default()
        self.config_raw.preprocessing = PreprocessingConfig(
            enable_morphological=False
        )

        self.config_preprocessed = TranslationConfig.default()
        self.config_preprocessed.preprocessing = PreprocessingConfig(
            enable_morphological=True
        )

        self.pipeline_raw = TranslatorPipeline(config=self.config_raw)
        self.pipeline_preprocessed = TranslatorPipeline(
            config=self.config_preprocessed,
            decomposer=decomposer,
        )

    def compare_single(
        self,
        source: str,
        reference: str,
        source_lang: str = "de",
        target_lang: str = "en",
    ) -> PreprocessingComparison:
        """
        Compare translation with and without preprocessing.

        Args:
            source: Source text
            reference: Reference translation
            source_lang: Source language
            target_lang: Target language

        Returns:
            PreprocessingComparison result
        """
        # Raw translation (no preprocessing)
        raw_result = self.pipeline_raw.translate(source, source_lang, target_lang)

        # Preprocessed translation
        preprocessed_result = None
        if self.decomposer:
            preprocessed_result = self.pipeline_preprocessed.translate(
                source, source_lang, target_lang
            )

        # Compute BLEU scores
        bleu_raw = self.bleu_metric.score(raw_result.text, [reference]).score
        bleu_preprocessed = None
        improvement = None

        if preprocessed_result:
            bleu_preprocessed = self.bleu_metric.score(
                preprocessed_result.text, [reference]
            ).score
            # Calculate improvement as percentage change
            if bleu_raw > 0:
                improvement = ((bleu_preprocessed - bleu_raw) / bleu_raw) * 100
            else:
                improvement = (bleu_preprocessed - bleu_raw) * 100

        # Extract decompositions from metadata
        decompositions = None
        if preprocessed_result and "decompositions" in preprocessed_result.metadata:
            decompositions = preprocessed_result.metadata["decompositions"]

        return PreprocessingComparison(
            source_text=source,
            raw_translation=raw_result,
            preprocessed_translation=preprocessed_result,
            bleu_raw=bleu_raw,
            bleu_preprocessed=bleu_preprocessed,
            improvement=improvement,
            reference=reference,
            decompositions=decompositions,
        )

    def compare_batch(
        self,
        sources: List[str],
        references: List[str],
        source_lang: str = "de",
        target_lang: str = "en",
    ) -> List[PreprocessingComparison]:
        """
        Compare translation for multiple sentences.

        Args:
            sources: List of source texts
            references: List of reference translations
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of PreprocessingComparison results
        """
        results = []
        for source, reference in zip(sources, references):
            comparison = self.compare_single(source, reference, source_lang, target_lang)
            results.append(comparison)
        return results

    def print_comparison_report(
        self,
        comparisons: List[PreprocessingComparison],
        show_details: bool = True,
    ) -> None:
        """
        Print formatted comparison report.

        Args:
            comparisons: List of PreprocessingComparison results
            show_details: Whether to show detailed per-sentence info
        """
        print()
        print("=" * 90)
        print("MORPHOLOGICAL PREPROCESSING EXPERIMENT REPORT")
        print("=" * 90)
        print()

        # Summary statistics
        bleu_raw_scores = [c.bleu_raw for c in comparisons if c.bleu_raw is not None]
        bleu_preprocessed_scores = [
            c.bleu_preprocessed for c in comparisons if c.bleu_preprocessed is not None
        ]
        improvements = [c.improvement for c in comparisons if c.improvement is not None]

        avg_bleu_raw = sum(bleu_raw_scores) / len(bleu_raw_scores) if bleu_raw_scores else 0
        avg_bleu_preprocessed = (
            sum(bleu_preprocessed_scores) / len(bleu_preprocessed_scores)
            if bleu_preprocessed_scores
            else 0
        )
        avg_improvement = (
            sum(improvements) / len(improvements) if improvements else 0
        )

        print(f"Sentences analyzed: {len(comparisons)}")
        print()

        print("AGGREGATE RESULTS")
        print("-" * 90)
        print(f"Avg BLEU (Raw):         {avg_bleu_raw:.4f}")
        print(f"Avg BLEU (Preprocessed): {avg_bleu_preprocessed:.4f}")
        print(f"Avg Improvement:        {avg_improvement:+.2f}%")
        print()

        # Improvement distribution
        improved = sum(1 for imp in improvements if imp > 0)
        same = sum(1 for imp in improvements if imp == 0)
        worse = sum(1 for imp in improvements if imp < 0)

        print("IMPACT DISTRIBUTION")
        print("-" * 90)
        print(f"Improved: {improved:3d} sentences ({improved/len(improvements)*100:5.1f}%)")
        print(f"Same:     {same:3d} sentences ({same/len(improvements)*100:5.1f}%)")
        print(f"Worse:    {worse:3d} sentences ({worse/len(improvements)*100:5.1f}%)")
        print()

        if show_details:
            print("DETAILED RESULTS")
            print("-" * 90)
            for i, comp in enumerate(comparisons, 1):
                print(f"[{i}] Source: {comp.source_text[:60]}")
                print(
                    f"    Raw BLEU:        {comp.bleu_raw:.4f} | "
                    f"Preprocessed BLEU: {comp.bleu_preprocessed:.4f} | "
                    f"Change: {comp.improvement:+.2f}%"
                )
                print(f"    Raw Translation:         {comp.raw_translation.text[:70]}")
                if comp.preprocessed_translation:
                    print(
                        f"    Preprocessed Translation: "
                        f"{comp.preprocessed_translation.text[:70]}"
                    )
                if comp.decompositions:
                    decomps_str = ", ".join(
                        [f"{k}→{v}" for k, v in list(comp.decompositions.items())[:3]]
                    )
                    print(f"    Decompositions: {decomps_str}")
                print()

        print("=" * 90)

    def export_results(
        self, comparisons: List[PreprocessingComparison], output_file: str
    ) -> None:
        """Export results to JSON."""
        results_dict = {
            "experiment": "morphological_preprocessing",
            "sentence_count": len(comparisons),
            "comparisons": [c.to_dict() for c in comparisons],
            "summary": {
                "avg_bleu_raw": sum(c.bleu_raw for c in comparisons if c.bleu_raw) / len([c for c in comparisons if c.bleu_raw]),
                "avg_bleu_preprocessed": sum(c.bleu_preprocessed for c in comparisons if c.bleu_preprocessed) / len([c for c in comparisons if c.bleu_preprocessed]),
                "improvement_count": sum(1 for c in comparisons if c.improvement and c.improvement > 0),
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"Results exported to: {output_file}")
