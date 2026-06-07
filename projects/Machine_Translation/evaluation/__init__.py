"""Evaluation package for machine translation."""

from evaluation.metrics import BLEUMetric, ChrFMetric, BLEUScore, ChrFScore
from evaluation.linguistic_report import (
    LinguisticReportGenerator,
    LinguisticEvaluationReport,
    ErrorPattern,
)
from evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    SentenceEvaluator,
)

__all__ = [
    "BLEUMetric",
    "ChrFMetric",
    "BLEUScore",
    "ChrFScore",
    "LinguisticReportGenerator",
    "LinguisticEvaluationReport",
    "ErrorPattern",
    "EvaluationPipeline",
    "EvaluationResult",
    "SentenceEvaluator",
]
