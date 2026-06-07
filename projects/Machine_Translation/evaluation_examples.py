"""Examples of translation evaluation with metrics and linguistic analysis."""

from evaluation.evaluation_pipeline import EvaluationPipeline, SentenceEvaluator
from evaluation.linguistic_report import LinguisticReportGenerator
from analysis.error_analyzer import TranslationErrorAnalyzer


def example_corpus_evaluation():
    """Example 1: Evaluate an entire corpus."""
    print("=" * 80)
    print("EXAMPLE 1: Corpus-Level Evaluation (EN → DE)")
    print("=" * 80)

    # Sample translations
    sources = [
        "The company will not increase prices next year.",
        "Good morning, how are you?",
        "I have never seen this before.",
        "Machine translation is fascinating.",
        "The weather is beautiful today.",
    ]

    hypotheses = [
        "Die Firma wird Preise erhöhen nächstes Jahr.",  # Missing negation
        "Guten Morgen, wie geht es dir?",
        "Ich habe dieses gesehen zuvor nie.",  # Word order issue
        "Maschinelle Übersetzung ist faszinierend.",
        "Das Wetter ist heute schön.",
    ]

    references_list = [
        ["Die Firma wird die Preise nächstes Jahr nicht erhöhen."],
        ["Guten Morgen, wie geht es Ihnen?"],
        ["Ich habe so etwas noch nie gesehen."],
        ["Maschinelle Übersetzung ist faszinierend."],
        ["Das Wetter ist heute wunderbar."],
    ]

    # Run evaluation
    pipeline = EvaluationPipeline(
        bleu_max_n=4,
        include_chrf=True,
        include_linguistic=True,
    )

    result = pipeline.evaluate(
        hypotheses=hypotheses,
        references_list=references_list,
        source_lang="en",
        target_lang="de",
        sources=sources,
    )

    # Print report
    pipeline.print_report(result)

    print()


def example_sentence_evaluation():
    """Example 2: Evaluate individual sentences."""
    print("=" * 80)
    print("EXAMPLE 2: Sentence-Level Evaluation with Detailed Analysis")
    print("=" * 80)

    evaluator = SentenceEvaluator()

    source = "The system should never modify user data without consent."
    hypothesis = "Das System sollte Benutzerdaten ändern ohne Zustimmung."  # Missing negation
    references = [
        "Das System sollte Benutzerdaten niemals ohne Zustimmung ändern.",
        "Das System darf Benutzerdaten ohne Zustimmung niemals verändern.",
    ]

    result = evaluator.evaluate_sentence(
        source=source,
        hypothesis=hypothesis,
        references=references,
        source_lang="en",
        target_lang="de",
    )

    print(f"Source: {source}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Reference: {references[0]}")
    print()

    print("Metrics:")
    print(f"  BLEU: {result['metrics']['bleu']['score']:.4f}")
    print(f"  chrF: {result['metrics']['chrf']['score']:.4f}")
    print()

    print("Linguistic Analysis:")
    print(f"  Overall Quality: {result['overall_quality'].upper()}")
    analysis = result['linguistic_analysis']
    print(f"  Errors Detected: {len(analysis['errors'])}")
    for error in analysis['errors']:
        print(f"    - {error['category'].upper()}: {error['explanation']}")

    print()
    print()


def example_batch_linguistic_analysis():
    """Example 3: Linguistic pattern analysis across batch."""
    print("=" * 80)
    print("EXAMPLE 3: Batch Linguistic Pattern Analysis (DE → EN)")
    print("=" * 80)

    sources = [
        "Der große Schmetterling sitzt auf der Blume.",
        "Die klugen Kinder spielen im Park.",
        "Das interessante Buch liegt auf dem Tisch.",
        "Ein alter Mann geht langsam die Straße.",
        "Die schöne Frau trägt ein rotes Kleid.",
        "Der Schmetterlingshaus ist sehr groß.",  # Compound
        "Die Großmutter hat die Enkelin immer geliebt.",
        "Das Abendessen wird sehr lecker sein.",
    ]

    hypotheses = [
        "The big butterfly sits on the flower.",
        "The smart children play in the park.",
        "The interesting book lies on the table.",
        "An old man goes slowly down the street.",
        "The beautiful woman wears a red dress.",
        "The butterfly house is very big.",
        "The grandmother has always loved the granddaughter.",
        "The dinner will be very delicious.",
    ]

    references_list = [
        ["The large butterfly is sitting on the flower."],
        ["The intelligent children are playing in the park."],
        ["The fascinating book is lying on the table."],
        ["An elderly man walks slowly down the street."],
        ["The beautiful woman is wearing a red dress."],
        ["The butterfly house is very large."],
        ["The grandmother has always loved her granddaughter."],
        ["The dinner will be delicious."],
    ]

    # Run evaluation
    pipeline = EvaluationPipeline(
        bleu_max_n=4,
        include_chrf=True,
        include_linguistic=True,
    )

    result = pipeline.evaluate(
        hypotheses=hypotheses,
        references_list=references_list,
        source_lang="de",
        target_lang="en",
        sources=sources,
    )

    # Print report
    pipeline.print_report(result)

    print()


def example_morphological_focus():
    """Example 4: Focus on morphological errors."""
    print("=" * 80)
    print("EXAMPLE 4: Morphological Error Focus (Russian → English)")
    print("=" * 80)

    # Russian sentences with case/gender/number
    sources = [
        "Большая кошка спит на мягком диване.",  # nom. fem., prep. neut.
        "Высокий человек держит тяжелый мешок.",  # nom. masc., acc. masc.
    ]

    hypotheses = [
        "Big cat sleeps on soft sofa.",  # Missing agreement nuances
        "Tall man holds heavy bag.",
    ]

    references_list = [
        ["The big cat is sleeping on a soft couch."],
        ["A tall man is holding a heavy sack."],
    ]

    analyzer = TranslationErrorAnalyzer()

    print("Analyzing morphological handling...")
    print()

    for src, hyp, ref in zip(sources, hypotheses, references_list):
        analysis = analyzer.analyze(
            source=src,
            target=hyp,
            reference=ref[0],
            source_lang="ru",
            target_lang="en",
        )

        print(f"Source (RU): {src}")
        print(f"Target (EN): {hyp}")
        print(f"Quality: {analysis.overall_quality.upper()}")

        morph_errors = [
            e for e in analysis.errors
            if e.category.value == "morphological_loss"
        ]
        agreement_errors = [
            e for e in analysis.errors
            if e.category.value == "agreement_error"
        ]

        if morph_errors:
            print(f"  Morphological losses: {len(morph_errors)}")
            for err in morph_errors[:2]:
                print(f"    - {err.explanation}")

        if agreement_errors:
            print(f"  Agreement errors: {len(agreement_errors)}")
            for err in agreement_errors[:2]:
                print(f"    - {err.explanation}")

        print()

    print()


def example_metrics_only():
    """Example 5: Quick metrics-only evaluation (no linguistic analysis)."""
    print("=" * 80)
    print("EXAMPLE 5: Fast Metrics-Only Evaluation")
    print("=" * 80)

    hypotheses = [
        "The cat sits on the mat.",
        "A dog runs in the park.",
    ]

    references_list = [
        ["The cat is sitting on the mat.", "A cat sits on the mat."],
        ["The dog runs in the park.", "A dog is running in the park."],
    ]

    # Fast evaluation without linguistic analysis
    pipeline = EvaluationPipeline(
        bleu_max_n=4,
        include_chrf=True,
        include_linguistic=False,  # Skip linguistic analysis
    )

    result = pipeline.evaluate(
        hypotheses=hypotheses,
        references_list=references_list,
        source_lang="en",
        target_lang="en",
    )

    print("Quick Metrics (no linguistic analysis):")
    print(f"  BLEU: {result.bleu.score:.4f}")
    if result.chrf:
        print(f"  chrF: {result.chrf.score:.4f}")

    print()
    print()


def example_json_export():
    """Example 6: Export evaluation results to JSON."""
    print("=" * 80)
    print("EXAMPLE 6: Export Evaluation Results to JSON")
    print("=" * 80)

    sources = ["Hello world", "Good morning"]
    hypotheses = ["Hallo Welt", "Guten Morgen"]
    references_list = [
        ["Hallo Welt"],
        ["Guten Morgen"],
    ]

    pipeline = EvaluationPipeline(include_linguistic=True)
    result = pipeline.evaluate(
        hypotheses=hypotheses,
        references_list=references_list,
        source_lang="en",
        target_lang="de",
        sources=sources,
    )

    # Export to JSON
    json_output = result.to_json()
    print("JSON Export (excerpt):")
    print(json_output[:500])
    print("...")

    # Save to file
    output_file = "evaluation_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_output)
    print(f"\nFull results saved to: {output_file}")

    print()
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║  Machine Translation Evaluation Examples                                 ║")
    print("║  Metrics + Linguistic Analysis                                           ║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        example_corpus_evaluation()
        example_sentence_evaluation()
        example_batch_linguistic_analysis()
        example_morphological_focus()
        example_metrics_only()
        example_json_export()

        print("=" * 80)
        print("All evaluation examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
