"""Linguistic evaluation report generation."""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json

from analysis.error_types import SentenceLinguisticAnalysis, BatchLinguisticAnalysis, ErrorCategory


@dataclass
class ErrorPattern:
    """A recurring error pattern."""
    pattern: str
    count: int
    examples: List[Tuple[str, str]] = field(default_factory=list)  # (source, target)
    confidence: float = 0.7


@dataclass
class LinguisticEvaluationReport:
    """Comprehensive linguistic evaluation report."""

    # Metadata
    source_lang: str
    target_lang: str
    sentence_count: int

    # Error summaries
    total_errors: int
    error_distribution: Dict[str, int]

    # Quality metrics
    quality_distribution: Dict[str, int]
    avg_errors_per_sentence: float

    # Linguistic patterns
    morphological_patterns: List[ErrorPattern] = field(default_factory=list)
    compound_patterns: List[ErrorPattern] = field(default_factory=list)
    agreement_patterns: List[ErrorPattern] = field(default_factory=list)
    word_order_patterns: List[ErrorPattern] = field(default_factory=list)
    semantic_patterns: List[ErrorPattern] = field(default_factory=list)

    # Summary notes
    linguistic_summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metadata": {
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "sentence_count": self.sentence_count,
            },
            "error_summary": {
                "total_errors": self.total_errors,
                "avg_errors_per_sentence": self.avg_errors_per_sentence,
                "distribution": self.error_distribution,
                "quality_distribution": self.quality_distribution,
            },
            "linguistic_patterns": {
                "morphological": [
                    {
                        "pattern": p.pattern,
                        "count": p.count,
                        "confidence": p.confidence,
                        "examples": p.examples[:3],
                    }
                    for p in self.morphological_patterns[:5]
                ],
                "compounds": [
                    {
                        "pattern": p.pattern,
                        "count": p.count,
                        "confidence": p.confidence,
                        "examples": p.examples[:3],
                    }
                    for p in self.compound_patterns[:5]
                ],
                "agreement": [
                    {
                        "pattern": p.pattern,
                        "count": p.count,
                        "confidence": p.confidence,
                        "examples": p.examples[:3],
                    }
                    for p in self.agreement_patterns[:5]
                ],
                "word_order": [
                    {
                        "pattern": p.pattern,
                        "count": p.count,
                        "confidence": p.confidence,
                        "examples": p.examples[:3],
                    }
                    for p in self.word_order_patterns[:5]
                ],
                "semantic": [
                    {
                        "pattern": p.pattern,
                        "count": p.count,
                        "confidence": p.confidence,
                        "examples": p.examples[:3],
                    }
                    for p in self.semantic_patterns[:5]
                ],
            },
            "summary": {
                "linguistic_summary": self.linguistic_summary,
                "recommendations": self.recommendations,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class LinguisticReportGenerator:
    """Generate comprehensive linguistic evaluation reports."""

    def __init__(self, lang_pair: Tuple[str, str] = ("en", "de")):
        """
        Initialize report generator.

        Args:
            lang_pair: (source_lang, target_lang)
        """
        self.source_lang, self.target_lang = lang_pair

    def generate(
        self, batch_analysis: BatchLinguisticAnalysis
    ) -> LinguisticEvaluationReport:
        """
        Generate complete evaluation report.

        Args:
            batch_analysis: BatchLinguisticAnalysis from error analyzer

        Returns:
            LinguisticEvaluationReport
        """
        report = LinguisticEvaluationReport(
            source_lang=batch_analysis.source_lang,
            target_lang=batch_analysis.target_lang,
            sentence_count=len(batch_analysis.analyses),
        )

        # Compute error statistics
        error_dist = batch_analysis.error_distribution()
        report.error_distribution = error_dist
        report.total_errors = sum(error_dist.values())
        report.avg_errors_per_sentence = (
            report.total_errors / report.sentence_count
            if report.sentence_count > 0 else 0.0
        )

        # Quality distribution
        report.quality_distribution = batch_analysis.quality_distribution()

        # Extract linguistic patterns
        self._extract_patterns(batch_analysis, report)

        # Generate summary
        report.linguistic_summary = self._generate_summary(report)
        report.recommendations = self._generate_recommendations(report)

        return report

    def _extract_patterns(
        self,
        batch_analysis: BatchLinguisticAnalysis,
        report: LinguisticEvaluationReport,
    ) -> None:
        """Extract recurring error patterns from analyses."""

        # Pattern accumulators
        morpho_patterns = Counter()
        compound_patterns = Counter()
        agreement_patterns = Counter()
        word_order_patterns = Counter()
        semantic_patterns = Counter()

        # Examples
        morpho_examples = []
        compound_examples = []
        agreement_examples = []
        word_order_examples = []
        semantic_examples = []

        for analysis in batch_analysis.analyses:
            for error in analysis.errors:
                category = error.category.value

                if category == "morphological_loss":
                    pattern = f"Feature loss: {error.explanation}"
                    morpho_patterns[pattern] += 1
                    morpho_examples.append(
                        (analysis.source_text, analysis.target_text)
                    )

                elif category == "compound_breakdown":
                    pattern = f"Compound: {error.source_span}"
                    compound_patterns[pattern] += 1
                    compound_examples.append(
                        (analysis.source_text, analysis.target_text)
                    )

                elif category == "agreement_error":
                    pattern = f"Agreement error: {error.explanation}"
                    agreement_patterns[pattern] += 1
                    agreement_examples.append(
                        (analysis.source_text, analysis.target_text)
                    )

                elif category == "word_order_shift":
                    pattern = "Word order reordering"
                    word_order_patterns[pattern] += 1
                    word_order_examples.append(
                        (analysis.source_text, analysis.target_text)
                    )

                elif category == "semantic_drift":
                    pattern = f"Semantic issue: {error.explanation}"
                    semantic_patterns[pattern] += 1
                    semantic_examples.append(
                        (analysis.source_text, analysis.target_text)
                    )

        # Convert to ErrorPattern objects (top 5 per category)
        report.morphological_patterns = [
            ErrorPattern(
                pattern=pattern,
                count=count,
                examples=morpho_examples[:3],
                confidence=min(count / report.sentence_count, 1.0),
            )
            for pattern, count in morpho_patterns.most_common(5)
        ]

        report.compound_patterns = [
            ErrorPattern(
                pattern=pattern,
                count=count,
                examples=compound_examples[:3],
                confidence=min(count / report.sentence_count, 1.0),
            )
            for pattern, count in compound_patterns.most_common(5)
        ]

        report.agreement_patterns = [
            ErrorPattern(
                pattern=pattern,
                count=count,
                examples=agreement_examples[:3],
                confidence=min(count / report.sentence_count, 1.0),
            )
            for pattern, count in agreement_patterns.most_common(5)
        ]

        report.word_order_patterns = [
            ErrorPattern(
                pattern=pattern,
                count=count,
                examples=word_order_examples[:3],
                confidence=min(count / report.sentence_count, 1.0),
            )
            for pattern, count in word_order_patterns.most_common(5)
        ]

        report.semantic_patterns = [
            ErrorPattern(
                pattern=pattern,
                count=count,
                examples=semantic_examples[:3],
                confidence=min(count / report.sentence_count, 1.0),
            )
            for pattern, count in semantic_patterns.most_common(5)
        ]

    def _generate_summary(self, report: LinguisticEvaluationReport) -> str:
        """Generate human-readable linguistic summary."""
        lines = []

        # Header
        lines.append(
            f"Translation Quality Summary ({report.source_lang.upper()} → {report.target_lang.upper()})"
        )
        lines.append("-" * 70)

        # Quality overview
        quality_dist = report.quality_distribution
        total = sum(quality_dist.values())
        excellent_pct = (quality_dist.get("excellent", 0) / total * 100) if total > 0 else 0
        good_pct = (quality_dist.get("good", 0) / total * 100) if total > 0 else 0
        fair_pct = (quality_dist.get("fair", 0) / total * 100) if total > 0 else 0
        poor_pct = (quality_dist.get("poor", 0) / total * 100) if total > 0 else 0

        lines.append(
            f"\nOverall Quality Distribution (n={total} sentences):"
        )
        lines.append(f"  Excellent: {quality_dist.get('excellent', 0):3d} ({excellent_pct:5.1f}%)")
        lines.append(f"  Good:      {quality_dist.get('good', 0):3d} ({good_pct:5.1f}%)")
        lines.append(f"  Fair:      {quality_dist.get('fair', 0):3d} ({fair_pct:5.1f}%)")
        lines.append(f"  Poor:      {quality_dist.get('poor', 0):3d} ({poor_pct:5.1f}%)")

        # Error statistics
        lines.append(f"\nError Statistics:")
        lines.append(f"  Total errors: {report.total_errors}")
        lines.append(f"  Avg errors per sentence: {report.avg_errors_per_sentence:.2f}")

        if report.error_distribution:
            lines.append(f"\n  Error type distribution:")
            for error_type, count in sorted(
                report.error_distribution.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (count / report.total_errors * 100) if report.total_errors > 0 else 0
                lines.append(f"    {error_type:25s}: {count:3d} ({pct:5.1f}%)")

        # Linguistic patterns
        lines.append(f"\nKey Linguistic Patterns:")

        if report.morphological_patterns:
            lines.append(f"\n  Morphological Issues (German/Russian case, gender, number):")
            for pattern in report.morphological_patterns[:3]:
                lines.append(
                    f"    • {pattern.pattern} ({pattern.count}x, confidence: {pattern.confidence:.2f})"
                )

        if report.compound_patterns and report.target_lang.lower() == "de":
            lines.append(f"\n  Compound Handling Issues (German):")
            for pattern in report.compound_patterns[:3]:
                lines.append(
                    f"    • {pattern.pattern} ({pattern.count}x, confidence: {pattern.confidence:.2f})"
                )

        if report.agreement_patterns:
            lines.append(f"\n  Agreement Errors (subject-verb, noun-adjective):")
            for pattern in report.agreement_patterns[:3]:
                lines.append(
                    f"    • {pattern.pattern} ({pattern.count}x, confidence: {pattern.confidence:.2f})"
                )

        if report.word_order_patterns:
            lines.append(f"\n  Word Order Patterns:")
            for pattern in report.word_order_patterns[:3]:
                lines.append(
                    f"    • {pattern.pattern} ({pattern.count}x, confidence: {pattern.confidence:.2f})"
                )

        if report.semantic_patterns:
            lines.append(f"\n  Semantic Issues:")
            for pattern in report.semantic_patterns[:3]:
                lines.append(
                    f"    • {pattern.pattern} ({pattern.count}x, confidence: {pattern.confidence:.2f})"
                )

        return "\n".join(lines)

    def _generate_recommendations(self, report: LinguisticEvaluationReport) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Morphological errors
        if report.morphological_patterns:
            top_morph = report.morphological_patterns[0]
            if top_morph.count > report.sentence_count * 0.1:
                recommendations.append(
                    f"Morphological feature loss is frequent ({top_morph.count}x). "
                    f"Consider fine-tuning on morphologically-rich data or adding "
                    f"morphological loss penalties during training."
                )

        # Compound errors
        if report.compound_patterns:
            top_compound = report.compound_patterns[0]
            if top_compound.count > report.sentence_count * 0.05:
                recommendations.append(
                    f"German compounds are problematic ({top_compound.count}x). "
                    f"Integrate the GermanCompoundDecomposer into preprocessing "
                    f"or use compound-aware tokenization."
                )

        # Agreement errors
        if report.agreement_patterns:
            top_agreement = report.agreement_patterns[0]
            if top_agreement.count > report.sentence_count * 0.1:
                recommendations.append(
                    f"Agreement errors are common ({top_agreement.count}x). "
                    f"Use morphological constraints during decoding or add agreement "
                    f"scoring as a reranking criterion."
                )

        # Word order issues
        if report.word_order_patterns:
            top_wo = report.word_order_patterns[0]
            if top_wo.count > report.sentence_count * 0.2:
                recommendations.append(
                    f"Significant word order reordering detected ({top_wo.count}x). "
                    f"This may reflect language typology (SVO→SOV). Verify this is "
                    f"linguistically appropriate for the language pair."
                )

        # Semantic issues
        if report.semantic_patterns:
            top_semantic = report.semantic_patterns[0]
            if top_semantic.count > report.sentence_count * 0.05:
                recommendations.append(
                    f"Semantic drift detected ({top_semantic.count}x). "
                    f"Prioritize semantic similarity metrics (e.g., BERTScore) during "
                    f"evaluation and consider back-translation validation."
                )

        # Quality-based recommendations
        poor_count = report.quality_distribution.get("poor", 0)
        if poor_count > report.sentence_count * 0.1:
            recommendations.append(
                f"{poor_count} sentences have poor quality. "
                f"Conduct error analysis on these cases and consider data augmentation "
                f"or domain-specific fine-tuning."
            )

        if not recommendations:
            recommendations.append(
                f"Translation quality is acceptable. Consider testing on out-of-domain "
                f"data and monitoring for morphological/semantic drift in production."
            )

        return recommendations
