"""Evaluation pipeline orchestration."""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import json

from evaluation.metrics import BLEUMetric, ChrFMetric, BLEUScore, ChrFScore
from evaluation.linguistic_report import LinguisticReportGenerator, LinguisticEvaluationReport
from analysis.error_analyzer import TranslationErrorAnalyzer
from analysis.error_types import BatchLinguisticAnalysis


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    # Automatic metrics
    bleu: BLEUScore
    chrf: Optional[ChrFScore] = None

    # Linguistic analysis
    error_analysis: Optional[BatchLinguisticAnalysis] = None
    linguistic_report: Optional[LinguisticEvaluationReport] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "metrics": {
                "bleu": {
                    "score": self.bleu.score,
                    "precisions": self.bleu.precisions,
                    "bp": self.bleu.bp,
                    "ratio": self.bleu.ratio,
                },
            },
        }

        if self.chrf:
            result["metrics"]["chrf"] = {
                "score": self.chrf.score,
                "precision": self.chrf.precision,
                "recall": self.chrf.recall,
            }

        if self.linguistic_report:
            result["linguistic_analysis"] = self.linguistic_report.to_dict()

        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class EvaluationPipeline:
    """
    Complete evaluation pipeline combining metrics and linguistic analysis.
    """

    def __init__(
        self,
        bleu_max_n: int = 4,
        include_chrf: bool = True,
        include_linguistic: bool = True,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            bleu_max_n: Maximum n-gram for BLEU (default 4)
            include_chrf: Whether to compute chrF scores
            include_linguistic: Whether to perform linguistic analysis
        """
        self.bleu_metric = BLEUMetric(max_n=bleu_max_n)
        self.chrf_metric = ChrFMetric() if include_chrf else None
        self.include_linguistic = include_linguistic
        self.error_analyzer = (
            TranslationErrorAnalyzer() if include_linguistic else None
        )

    def evaluate(
        self,
        hypotheses: List[str],
        references_list: List[List[str]],
        source_lang: str = "en",
        target_lang: str = "en",
        sources: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Run complete evaluation.

        Args:
            hypotheses: List of translated texts
            references_list: List of reference sets (one per hypothesis)
            source_lang: Source language code
            target_lang: Target language code
            sources: Optional source texts (for linguistic analysis)

        Returns:
            EvaluationResult with all metrics and analysis
        """
        # Compute BLEU
        bleu = self.bleu_metric.corpus_score(hypotheses, references_list)

        # Compute chrF
        chrf = None
        if self.chrf_metric:
            flat_refs = [refs[0] for refs in references_list]  # Use first reference
            chrf = self.chrf_metric.corpus_score(hypotheses, flat_refs)

        # Linguistic analysis
        error_analysis = None
        linguistic_report = None

        if self.include_linguistic and sources:
            # Run error analysis
            error_analysis = self.error_analyzer.analyze_batch(
                sources=sources,
                targets=hypotheses,
                references=[refs[0] for refs in references_list],
                source_lang=source_lang,
                target_lang=target_lang,
            )

            # Generate linguistic report
            report_gen = LinguisticReportGenerator((source_lang, target_lang))
            linguistic_report = report_gen.generate(error_analysis)

        return EvaluationResult(
            bleu=bleu,
            chrf=chrf,
            error_analysis=error_analysis,
            linguistic_report=linguistic_report,
        )

    def print_report(self, result: EvaluationResult) -> None:
        """Print evaluation report to console."""
        print()
        print("=" * 80)
        print("TRANSLATION EVALUATION REPORT")
        print("=" * 80)
        print()

        # Metrics
        print("AUTOMATIC METRICS")
        print("-" * 80)
        print(f"BLEU:  {result.bleu}")
        if result.chrf:
            print(f"chrF:  {result.chrf}")
        print()

        # Linguistic analysis
        if result.linguistic_report:
            print("LINGUISTIC ANALYSIS")
            print("-" * 80)
            print(result.linguistic_report.linguistic_summary)
            print()

            if result.linguistic_report.recommendations:
                print("RECOMMENDATIONS FOR IMPROVEMENT")
                print("-" * 80)
                for i, rec in enumerate(result.linguistic_report.recommendations, 1):
                    print(f"{i}. {rec}")
                print()

        print("=" * 80)

    def print_detailed_report(self, result: EvaluationResult, save_path: Optional[str] = None) -> None:
        """Print detailed report with all information."""
        self.print_report(result)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(result.to_json())
            print(f"\nDetailed report saved to: {save_path}")


class SentenceEvaluator:
    """Evaluate individual sentences."""

    def __init__(self):
        self.bleu_metric = BLEUMetric(max_n=4)
        self.chrf_metric = ChrFMetric()
        self.error_analyzer = TranslationErrorAnalyzer()

    def evaluate_sentence(
        self,
        source: str,
        hypothesis: str,
        references: List[str],
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> dict:
        """
        Evaluate a single sentence.

        Args:
            source: Source text
            hypothesis: Translated text
            references: List of reference translations
            source_lang: Source language
            target_lang: Target language

        Returns:
            Dictionary with metrics and analysis
        """
        # Compute metrics
        bleu = self.bleu_metric.score(hypothesis, references)
        chrf = self.chrf_metric.score(hypothesis, references[0])

        # Linguistic analysis
        error_analysis = self.error_analyzer.analyze(
            source=source,
            target=hypothesis,
            reference=references[0] if references else None,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        return {
            "hypothesis": hypothesis,
            "references": references,
            "metrics": {
                "bleu": {
                    "score": bleu.score,
                    "precisions": bleu.precisions,
                },
                "chrf": {
                    "score": chrf.score,
                },
            },
            "linguistic_analysis": error_analysis.to_dict(),
            "overall_quality": error_analysis.overall_quality,
        }
