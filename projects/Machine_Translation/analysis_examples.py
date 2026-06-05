"""Examples of linguistic error analysis for machine translation."""

import json
from analysis.error_analyzer import TranslationErrorAnalyzer
from analysis.compound_processor import CompoundProcessor


def example_single_sentence_analysis():
    """Example 1: Analyze a single translated sentence."""
    print("=" * 70)
    print("EXAMPLE 1: Single Sentence Linguistic Analysis")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    # Example translation with errors
    source = "The company will not increase prices next year."
    target = "Die Firma wird Preise erhöhen nächstes Jahr."  # Missing negation
    reference = "Die Firma wird die Preise nächstes Jahr nicht erhöhen."

    analysis = analyzer.analyze(
        source=source,
        target=target,
        reference=reference,
        source_lang="en",
        target_lang="de",
    )

    print(f"Source (EN): {analysis.source_text}")
    print(f"Target (DE): {analysis.target_text}")
    print(f"Reference (DE): {analysis.reference_text}")
    print()
    print(f"Overall Quality: {analysis.overall_quality.upper()}")
    print()

    if analysis.errors:
        print(f"Errors Detected: {len(analysis.errors)}")
        print()
        for i, error in enumerate(analysis.errors, 1):
            print(f"[{i}] {error.category.value.upper()}")
            print(f"    Severity: {error.severity}")
            print(f"    Source: {error.source_span}")
            print(f"    Target: {error.target_span}")
            print(f"    Explanation: {error.explanation}")
            print(f"    Confidence: {error.confidence:.2f}")
            print()
    else:
        print("No errors detected.")

    print()


def example_batch_analysis():
    """Example 2: Analyze multiple sentence pairs."""
    print("=" * 70)
    print("EXAMPLE 2: Batch Analysis with Error Distribution")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    sources = [
        "Good morning, how are you?",
        "The weather is beautiful today.",
        "I have never seen this before.",
    ]

    targets = [
        "Guten Morgen, wie geht es dir?",
        "Das Wetter ist heute schön.",
        "Ich habe dieses gesehen zuvor nie.",  # Word order issue
    ]

    batch_analysis = analyzer.analyze_batch(
        sources=sources,
        targets=targets,
        source_lang="en",
        target_lang="de",
    )

    print(f"Analyzed {len(batch_analysis.analyses)} sentence pairs")
    print()

    # Error distribution
    error_dist = batch_analysis.error_distribution()
    print("Error Distribution Across Batch:")
    if error_dist:
        for error_type, count in sorted(error_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    else:
        print("  (No errors detected)")

    print()

    # Quality distribution
    quality_dist = batch_analysis.quality_distribution()
    print("Translation Quality Distribution:")
    for quality, count in sorted(quality_dist.items()):
        print(f"  {quality}: {count}")

    print()
    print()


def example_morphological_analysis():
    """Example 3: Focus on morphological errors."""
    print("=" * 70)
    print("EXAMPLE 3: Morphological Error Detection (German & Russian)")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    # German: case/gender mismatch
    source_de = "Der große Mann sieht den kleinen Jungen."
    target_de = "Der großer Mann sieht den klein Jungen."  # Wrong agreements
    reference_de = "Der große Mann sieht den kleinen Jungen."

    analysis = analyzer.analyze(
        source=source_de,
        target=target_de,
        reference=reference_de,
        source_lang="de",
        target_lang="de",
    )

    print("German Morphological Analysis:")
    print(f"Source: {analysis.source_text}")
    print(f"Target: {analysis.target_text}")
    print()

    morph_errors = [e for e in analysis.errors if e.category.value == "morphological_loss"]
    agreement_errors = [e for e in analysis.errors if e.category.value == "agreement_error"]

    print(f"Morphological Losses: {len(morph_errors)}")
    for err in morph_errors:
        print(f"  - {err.explanation}")

    print(f"Agreement Errors: {len(agreement_errors)}")
    for err in agreement_errors:
        print(f"  - {err.explanation}")

    print()
    print()


def example_semantic_drift():
    """Example 4: Semantic drift detection."""
    print("=" * 70)
    print("EXAMPLE 4: Semantic Drift and Meaning Changes")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    pairs = [
        (
            "The software is working correctly.",
            "Die Software funktioniert nicht richtig.",  # Negation flip!
            "Die Software funktioniert korrekt.",
        ),
        (
            "I like this product.",
            "Ich mag dieses Produkt.",
            "Mir gefällt dieses Produkt.",
        ),
    ]

    for i, (source, target, reference) in enumerate(pairs, 1):
        analysis = analyzer.analyze(
            source=source,
            target=target,
            reference=reference,
            source_lang="en",
            target_lang="de",
        )

        print(f"Pair {i}:")
        print(f"  Source: {analysis.source_text}")
        print(f"  Target: {analysis.target_text}")

        semantic_errors = [
            e for e in analysis.errors if e.category.value == "semantic_drift"
        ]

        if semantic_errors:
            print(f"  ⚠ Semantic Issues Detected:")
            for err in semantic_errors:
                print(f"    - {err.explanation} (confidence: {err.confidence:.2f})")
        else:
            print(f"  ✓ No semantic drift detected")

        print()

    print()


def example_word_order_analysis():
    """Example 5: Word order shift detection."""
    print("=" * 70)
    print("EXAMPLE 5: Word Order Shifts and Reordering")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    # SOV (Russian/German) vs SVO (English) differences
    source_en = "She quickly reads interesting books."
    target_ru = "Она интересные книги быстро читает."  # Different word order structure
    reference_ru = "Она быстро читает интересные книги."

    analysis = analyzer.analyze(
        source=source_en,
        target=target_ru,
        reference=reference_ru,
        source_lang="en",
        target_lang="ru",
    )

    print(f"Source (EN - SVO): {analysis.source_text}")
    print(f"Target (RU - SOV): {analysis.target_text}")
    print()

    word_order_errors = [e for e in analysis.errors if e.category.value == "word_order_shift"]

    if word_order_errors:
        print("Word Order Issues:")
        for err in word_order_errors:
            print(f"  - {err.explanation}")
    else:
        print("No significant word order shifts detected.")

    print()
    print()


def example_compound_analysis():
    """Example 6: German compound analysis (without decomposer)."""
    print("=" * 70)
    print("EXAMPLE 6: German Compound Analysis")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    # German compound example
    source_de = "Das Schmetterling ist sehr schön."
    target_en = "The butterfly is very beautiful."
    reference_en = "The butterfly is very beautiful."

    analysis = analyzer.analyze(
        source=source_de,
        target=target_en,
        reference=reference_en,
        source_lang="de",
        target_lang="en",
    )

    print(f"Source (DE): {analysis.source_text}")
    print(f"Target (EN): {analysis.target_text}")
    print()

    compound_errors = [
        e for e in analysis.errors if e.category.value == "compound_breakdown"
    ]

    if compound_errors:
        print("Compound Issues:")
        for err in compound_errors:
            print(f"  - {err.explanation}")
    else:
        print("No compound decomposition issues detected (or no long words found).")

    print()
    print()


def example_json_output():
    """Example 7: JSON output for downstream processing."""
    print("=" * 70)
    print("EXAMPLE 7: JSON Structured Output")
    print("=" * 70)

    analyzer = TranslationErrorAnalyzer()

    source = "I do not like this solution."
    target = "Ich mag diese Lösung."  # Negation lost
    reference = "Ich mag diese Lösung nicht."

    analysis = analyzer.analyze(
        source=source,
        target=target,
        reference=reference,
        source_lang="en",
        target_lang="de",
    )

    # Output as JSON
    json_output = analysis.to_json()
    print("JSON Output (Pretty-printed):")
    print(json_output)

    print()
    print()


def example_compound_processor():
    """Example 8: CompoundProcessor integration point."""
    print("=" * 70)
    print("EXAMPLE 8: Compound Processor (Integration Point)")
    print("=" * 70)

    # Initialize without decomposer (stub mode)
    processor = CompoundProcessor(decomposer=None)

    source_de = "Das Schmetterlingshaus ist interessant."
    target_de = "Das Schmetterling Haus ist interessant."

    # This will work without actual decomposer, but with limited functionality
    analysis = processor.analyze_compound_preservation(source_de, target_de)

    print("Compound Analysis Result:")
    print(f"  Source Compound: {analysis.source_compound}")
    print(f"  Source Decomposition: {analysis.source_decomposition}")
    print(f"  Compound Preserved: {analysis.compound_preserved}")
    print(f"  Alignment Score: {analysis.alignment_score:.2f}")
    print(f"  Notes: {analysis.notes}")

    print()
    print("💡 To integrate your GermanCompoundDecomposer:")
    print("   1. Initialize: processor.set_decomposer(your_decomposer)")
    print("   2. Use: error_span = processor.create_error_span(source_de, target_de)")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║  Linguistic Error Analysis for Machine Translation             ║")
    print("║  Computational Linguistics Perspective                         ║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        example_single_sentence_analysis()
        example_batch_analysis()
        example_morphological_analysis()
        example_semantic_drift()
        example_word_order_analysis()
        example_compound_analysis()
        example_json_output()
        example_compound_processor()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()