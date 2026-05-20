"""Dataset-based evaluation utilities for analyzer quality checks."""

import argparse
import csv
import json
from pathlib import Path

try:
    from .analyzer import analyze_word
    from .utils import DATA_DIR, ensure_output_dir, save_json_file
except ImportError:
    from analyzer import analyze_word
    from utils import DATA_DIR, ensure_output_dir, save_json_file


DEFAULT_DATASET = DATA_DIR / "evaluation_dataset.csv"


def evaluate_dataset(dataset_path: Path = DEFAULT_DATASET):
    """Compute per-feature accuracies for labeled evaluation rows."""
    with dataset_path.open("r", encoding = "utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fields = ["lemma", "part_of_speech", "case", "gender", "number", "tense", "aspect", "person"]
    totals = {field: 0 for field in fields}
    correct = {field: 0 for field in fields}

    for row in rows:
        prediction = analyze_word(row["token"])
        for field in fields:
            expected = _normalize_label(row.get(field))
            if expected is None:
                # Empty labels are intentionally excluded from denominator counts.
                continue
            totals[field] += 1
            if _matches_expected(prediction, field, expected):
                correct[field] += 1

    metrics = {"examples": len(rows)}
    for field in fields:
        if totals[field]:
            metrics[f"{field}_accuracy"] = round(correct[field] / totals[field], 3)

    output_path = ensure_output_dir() / "evaluation_results.json"
    save_json_file(output_path, metrics)
    return metrics


def main():
    """CLI entry point for running evaluation against a CSV dataset."""
    parser = argparse.ArgumentParser(description = "Evaluate the Russian morphology analyzer")
    parser.add_argument("--dataset", type = Path, default = DEFAULT_DATASET, help = "CSV dataset with expected labels")
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(args.dataset), ensure_ascii = False, indent = 2))


def _normalize_label(value):
    """Normalize expected CSV labels and map empty values to None."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _matches_expected(prediction, field, expected):
    """Match expected label against primary prediction or candidate analyses."""
    primary = _normalize_label(prediction.get(field))
    if primary == expected:
        return True

    candidates = prediction.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if _normalize_label(candidate.get(field)) == expected:
                return True

    return False


if __name__ == "__main__":
    main()