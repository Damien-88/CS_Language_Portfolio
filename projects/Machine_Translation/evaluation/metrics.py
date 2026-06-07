"""Translation evaluation metrics (BLEU, chrF)."""

import math
from typing import List, Tuple, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
class BLEUScore:
    """BLEU score result."""
    score: float
    precisions: List[float]  # 1-gram, 2-gram, 3-gram, 4-gram
    bp: float  # Brevity penalty
    ratio: float  # Hypothesis length / Reference length

    def __repr__(self) -> str:
        return (
            f"BLEU = {self.score:.4f}, "
            f"precisions = {[f'{p:.4f}' for p in self.precisions]}, "
            f"BP = {self.bp:.4f}, ratio = {self.ratio:.4f}"
        )


@dataclass
class ChrFScore:
    """chrF score result."""
    score: float
    precision: float
    recall: float

    def __repr__(self) -> str:
        return f"chrF = {self.score:.4f}, precision = {self.precision:.4f}, recall = {self.recall:.4f}"


class BLEUMetric:
    """BLEU score computation (simplified but correct)."""

    def __init__(self, max_n: int = 4, smooth: bool = False):
        """
        Initialize BLEU metric.

        Args:
            max_n: Maximum n-gram order (default 4)
            smooth: Use smoothing (add-one smoothing if True)
        """
        self.max_n = max_n
        self.smooth = smooth

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list."""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams

    def _compute_bleu(
        self,
        hypothesis_tokens: List[str],
        reference_tokens_list: List[List[str]],
    ) -> BLEUScore:
        """Compute BLEU for single hypothesis-references pair."""

        precisions = []

        # Compute n-gram precisions
        for n in range(1, self.max_n + 1):
            hyp_ngrams = self._get_ngrams(hypothesis_tokens, n)

            # Get max counts from all references
            max_ref_ngrams = Counter()
            for ref_tokens in reference_tokens_list:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                for ngram in ref_ngrams:
                    max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], ref_ngrams[ngram])

            # Clipped count: min(hyp_count, max_ref_count)
            clipped_count = sum(
                min(hyp_ngrams[ngram], max_ref_ngrams[ngram])
                for ngram in hyp_ngrams
            )

            total_count = sum(hyp_ngrams.values())

            if total_count == 0:
                precision = 0.0
            else:
                precision = clipped_count / total_count
                if self.smooth:
                    precision = (clipped_count + 1) / (total_count + 1)

            precisions.append(precision)

        # Brevity penalty
        hyp_len = len(hypothesis_tokens)

        # Reference length = closest to hypothesis length
        ref_lens = [len(ref) for ref in reference_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - hyp_len))

        if hyp_len > closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0.0

        # Geometric mean of precisions
        log_precisions = [
            math.log(p) if p > 0 else float('-inf')
            for p in precisions
        ]

        if any(lp == float('-inf') for lp in log_precisions):
            geo_mean = 0.0
        else:
            geo_mean = math.exp(sum(log_precisions) / len(log_precisions))

        score = bp * geo_mean

        return BLEUScore(
            score=score,
            precisions=precisions,
            bp=bp,
            ratio=hyp_len / closest_ref_len if closest_ref_len > 0 else 0.0,
        )

    def score(
        self,
        hypothesis: str,
        references: List[str],
    ) -> BLEUScore:
        """
        Compute BLEU score for hypothesis against references.

        Args:
            hypothesis: Hypothesis translation
            references: List of reference translations

        Returns:
            BLEUScore object
        """
        hyp_tokens = hypothesis.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in references]

        return self._compute_bleu(hyp_tokens, ref_tokens_list)

    def corpus_score(
        self,
        hypotheses: List[str],
        references_list: List[List[str]],
    ) -> BLEUScore:
        """
        Compute corpus-level BLEU.

        Args:
            hypotheses: List of hypotheses
            references_list: List of reference sets (each hypothesis has multiple references)

        Returns:
            BLEUScore for entire corpus
        """
        all_hyp_ngrams = [Counter() for _ in range(self.max_n)]
        all_ref_ngrams = [Counter() for _ in range(self.max_n)]
        total_hyp_len = 0
        total_ref_len = 0

        for hyp, refs in zip(hypotheses, references_list):
            hyp_tokens = hyp.lower().split()
            ref_tokens_list = [ref.lower().split() for ref in refs]

            total_hyp_len += len(hyp_tokens)

            # Find closest reference length
            ref_lens = [len(ref) for ref in ref_tokens_list]
            closest_ref_len = min(ref_lens, key=lambda l: abs(l - len(hyp_tokens)))
            total_ref_len += closest_ref_len

            # Accumulate n-grams
            for n in range(1, self.max_n + 1):
                hyp_ngrams = self._get_ngrams(hyp_tokens, n)
                for ngram, count in hyp_ngrams.items():
                    all_hyp_ngrams[n-1][ngram] += count

                # Max from references
                for ref_tokens in ref_tokens_list:
                    ref_ngrams = self._get_ngrams(ref_tokens, n)
                    for ngram, count in ref_ngrams.items():
                        all_ref_ngrams[n-1][ngram] = max(
                            all_ref_ngrams[n-1][ngram], count
                        )

        # Compute corpus precisions
        precisions = []
        for n in range(self.max_n):
            clipped_count = sum(
                min(all_hyp_ngrams[n][ngram], all_ref_ngrams[n][ngram])
                for ngram in all_hyp_ngrams[n]
            )
            total_count = sum(all_hyp_ngrams[n].values())

            if total_count == 0:
                precision = 0.0
            else:
                precision = clipped_count / total_count
                if self.smooth:
                    precision = (clipped_count + 1) / (total_count + 1)

            precisions.append(precision)

        # Brevity penalty
        if total_hyp_len > total_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0

        # Geometric mean
        log_precisions = [
            math.log(p) if p > 0 else float('-inf')
            for p in precisions
        ]

        if any(lp == float('-inf') for lp in log_precisions):
            geo_mean = 0.0
        else:
            geo_mean = math.exp(sum(log_precisions) / len(log_precisions))

        score = bp * geo_mean

        return BLEUScore(
            score=score,
            precisions=precisions,
            bp=bp,
            ratio=total_hyp_len / total_ref_len if total_ref_len > 0 else 0.0,
        )


class ChrFMetric:
    """chrF (Character n-gram F-score) computation."""

    def __init__(self, order: int = 6, beta: float = 3.0):
        """
        Initialize chrF metric.

        Args:
            order: Order of character n-grams (default 6)
            beta: Beta for F-score weighting (default 3 = recall-focused)
        """
        self.order = order
        self.beta = beta

    def _get_char_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract character n-grams."""
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams

    def score(self, hypothesis: str, reference: str) -> ChrFScore:
        """
        Compute chrF score.

        Args:
            hypothesis: Hypothesis translation
            reference: Reference translation

        Returns:
            ChrFScore object
        """
        hyp_ngrams = self._get_char_ngrams(hypothesis, self.order)
        ref_ngrams = self._get_char_ngrams(reference, self.order)

        if not hyp_ngrams or not ref_ngrams:
            return ChrFScore(score=0.0, precision=0.0, recall=0.0)

        intersection = len(hyp_ngrams & ref_ngrams)

        precision = intersection / len(hyp_ngrams) if hyp_ngrams else 0.0
        recall = intersection / len(ref_ngrams) if ref_ngrams else 0.0

        beta_sq = self.beta ** 2
        chrf = (
            (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return ChrFScore(score=chrf, precision=precision, recall=recall)

    def corpus_score(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> ChrFScore:
        """
        Compute corpus-level chrF.

        Args:
            hypotheses: List of hypotheses
            references: List of references (one-to-one with hypotheses)

        Returns:
            ChrFScore for entire corpus
        """
        precisions = []
        recalls = []

        for hyp, ref in zip(hypotheses, references):
            score = self.score(hyp, ref)
            precisions.append(score.precision)
            recalls.append(score.recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        beta_sq = self.beta ** 2
        chrf = (
            (1 + beta_sq) * avg_precision * avg_recall / (beta_sq * avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0 else 0.0
        )

        return ChrFScore(score=chrf, precision=avg_precision, recall=avg_recall)
