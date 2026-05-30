from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import pytest
from german_compound_decomposer import GermanCompoundDecomposer


def _build_decomposer(
	lemmas: Iterable[str],
	frequencies: dict[str, int] | None = None,
) -> GermanCompoundDecomposer:
	# Keep tests deterministic by disabling optional spaCy dependency.
	decomposer = GermanCompoundDecomposer(use_spacy=False)
	# Inject a controlled lemma lexicon to isolate decomposition behavior.
	decomposer._lemma_set = {lemma.lower() for lemma in lemmas}

	# Override frequency priors when a test needs explicit ambiguity control.
	if frequencies is not None:
		# Normalize frequency keys so test inputs are case-insensitive.
		decomposer._root_frequencies = {k.lower(): v for k, v in frequencies.items()}
		# Recompute cached denominator used by probability scoring.
		decomposer._total_frequency = float(sum(decomposer._root_frequencies.values()) or 1.0)

	# Clear recursive memoization so each test starts from fresh cache state.
	decomposer._best_candidate.cache_clear()
	# Return fully configured test instance.
	return decomposer


def _component_texts(result: dict) -> List[str]:
	# Flatten component text sequence for concise assertions.
	return [component["text"] for component in result["components"]]


def _component_types(result: dict) -> List[str]:
	# Flatten component role sequence (modifier/head/fugenlaut).
	return [component["type"] for component in result["components"]]


# Parameterize canonical archetypes to validate baseline behavior compactly.
@pytest.mark.parametrize(
	"word,lemmas,expected_texts,expected_types",
	[
		(
			# Clean binary split with right-headed structure.
			"Hauptbahnhof",
			{"haupt", "bahnhof"},
			["haupt", "bahnhof"],
			["modifier", "head"],
		),
		(
			# Fugenlaut insertion between modifier and head.
			"Arbeitsplatz",
			{"arbeit", "platz"},
			["arbeit", "s", "platz"],
			["modifier", "fugenlaut", "head"],
		),
	],
)
def test_archetypal_compound_decompositions(
	word: str,
	lemmas: set[str],
	expected_texts: List[str],
	expected_types: List[str],
) -> None:
	# Build decomposer with tightly scoped lexical support for this case.
	decomposer = _build_decomposer(lemmas)

	# Execute decomposition under test.
	result = decomposer.decompose(word)

	# Validate exact component text sequence.
	assert _component_texts(result) == expected_texts
	# Validate exact component role sequence.
	assert _component_types(result) == expected_types
	# Confidence must always be bounded to valid probability range.
	assert 0.0 <= result["confidence_score"] <= 1.0


def test_stem_vocalic_mutation_buecherregal() -> None:
	# Include both canonical and mutated lexical forms.
	decomposer = _build_decomposer({"buch", "bücher", "regal"})

	# Analyze umlaut/mutation-sensitive compound.
	result = decomposer.decompose("Bücherregal")
	# Cache flattened texts for easier assertion readability.
	texts = _component_texts(result)
	# Cache flattened component roles for easier assertion readability.
	types = _component_types(result)

	# First lexical segment should resolve to expected stem family.
	assert texts[0] in {"buch", "bücher"}
	# Final lexical segment should be the head "regal".
	assert texts[-1] == "regal"
	# Leftmost lexical segment should be tagged modifier.
	assert types[0] == "modifier"
	# Rightmost lexical segment should be tagged head.
	assert types[-1] == "head"

	# The current decomposer may realize an interfix in this pattern
	# (e.g., buch + er + regal), which is acceptable for this archetype.
	if len(texts) == 3:
		# Middle segment should be expected linking morpheme.
		assert texts[1] == "er"
		# Middle segment role must be fugenlaut.
		assert types[1] == "fugenlaut"
	else:
		# Alternative valid output is a direct binary split.
		assert len(texts) == 2
	# Confidence must remain normalized.
	assert 0.0 <= result["confidence_score"] <= 1.0


def test_structural_homonym_prefers_highest_probability_path() -> None:
	# Configure lexicon/frequencies to force a meaningful ambiguity decision.
	decomposer = _build_decomposer(
		lemmas={"stau", "staub", "becken", "ecken"},
		frequencies={
			# Preferred path roots are intentionally more frequent.
			"stau": 1000,
			"becken": 900,
			# Competing path roots are intentionally less frequent.
			"staub": 50,
			"ecken": 20,
			# Background mass prevents trivial score saturation.
			"hintergrund": 2_000_000,
		},
	)

	# Candidate expected to win under frequency-informed scoring.
	preferred_path = (("stau", False), ("becken", False))
	# Candidate expected to lose under frequency-informed scoring.
	competing_path = (("staub", False), ("ecken", False))

	# Score preferred candidate using internal probability layer.
	preferred_probability = decomposer._calculate_path_probability(preferred_path)
	# Score competing candidate for direct comparison.
	competing_probability = decomposer._calculate_path_probability(competing_path)

	# Verify scoring layer itself encodes preference correctly.
	assert preferred_probability > competing_probability

	# Verify end-to-end decomposition returns preferred segmentation.
	result = decomposer.decompose("Staubecken")
	# Component sequence should match preferred path.
	assert _component_texts(result) == ["stau", "becken"]
	# Component roles should reflect modifier + head.
	assert _component_types(result) == ["modifier", "head"]
	# Confidence must remain normalized.
	assert 0.0 <= result["confidence_score"] <= 1.0


# Parameterize boundary and normalization edge cases.
@pytest.mark.parametrize(
	"word,lemmas,expected_texts,expected_types",
	[
		# Single-character token falls back to atomic head output.
		("A", set(), ["a"], ["head"]),
		# Non-compound lexical token should remain unsplit head.
		("Gehen", {"gehen"}, ["gehen"], ["head"]),
		# Hyphenated input should normalize and decompose correctly.
		("Arbeits-Platz", {"arbeit", "platz"}, ["arbeit", "s", "platz"], ["modifier", "fugenlaut", "head"]),
	],
)
def test_edge_cases(
	word: str,
	lemmas: set[str],
	expected_texts: List[str],
	expected_types: List[str],
) -> None:
	# Build decomposer with test-specific lexical inventory.
	decomposer = _build_decomposer(lemmas)

	# Execute decomposition on the selected edge-case input.
	result = decomposer.decompose(word)

	# Assert expected normalized/decomposed text components.
	assert _component_texts(result) == expected_texts
	# Assert expected component role labels.
	assert _component_types(result) == expected_types
	# Assert confidence range contract.
	assert 0.0 <= result["confidence_score"] <= 1.0


def test_frequency_dictionary_loader_from_file(tmp_path: Path) -> None:
	# Create temporary frequency file with mixed valid/invalid lines.
	freq_file = tmp_path / "freq.txt"
	# Include both supported separators and malformed rows.
	freq_file.write_text("stau 10\nbecken,20\n# comment\ninvalid\n", encoding="utf-8")

	# Load decomposer with explicit frequency file path.
	decomposer = GermanCompoundDecomposer(use_spacy=False, frequency_path=freq_file)

	# Validate whitespace-delimited frequency parsing.
	assert decomposer._root_frequencies["stau"] == 10
	# Validate comma-delimited frequency parsing.
	assert decomposer._root_frequencies["becken"] == 20
	# Validate total-frequency recomputation from loaded file.
	assert decomposer._total_frequency == 30.0