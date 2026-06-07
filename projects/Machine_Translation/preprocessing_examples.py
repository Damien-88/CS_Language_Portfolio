"""Examples demonstrating optional morphological preprocessing."""

from config import TranslationConfig, PreprocessingConfig
from pipeline.translator_pipeline import TranslatorPipeline
from preprocessing.morphological_preprocessing import GermanCompoundPreprocessor
from experiments.decomposition_experiment import MorphologicalPreprocessingExperiment


def example_preprocessing_disabled():
    """Example 1: Translation without morphological preprocessing."""
    print("=" * 80)
    print("EXAMPLE 1: Translation WITHOUT Morphological Preprocessing (Default)")
    print("=" * 80)

    config = TranslationConfig.default()
    config.preprocessing = PreprocessingConfig(enable_morphological=False)

    pipeline = TranslatorPipeline(config=config)

    source = "Das Schmetterlingshaus ist sehr groß."
    result = pipeline.translate(source, "de", "en")

    print(f"Source: {source}")
    print(f"Translation: {result.text}")
    print(f"Preprocessing enabled: {result.metadata.get('morphological_preprocessing', False)}")
    print()
    print()


def example_preprocessing_enabled_no_decomposer():
    """Example 2: Preprocessing enabled but no decomposer available (graceful degradation)."""
    print("=" * 80)
    print("EXAMPLE 2: Preprocessing Enabled But No Decomposer Available")
    print("=" * 80)

    config = TranslationConfig.default()
    config.preprocessing = PreprocessingConfig(enable_morphological=True)

    # No decomposer provided - should gracefully fall back
    pipeline = TranslatorPipeline(config=config, decomposer=None)

    source = "Das Schmetterlingshaus ist sehr groß."
    result = pipeline.translate(source, "de", "en")

    print(f"Source: {source}")
    print(f"Translation: {result.text}")
    print(f"Preprocessing enabled: {result.metadata.get('morphological_preprocessing', True)}")
    print(f"Decomposer available: {result.metadata.get('decomposer_available', False)}")
    print("⚠ Note: Preprocessing is enabled but no decomposer available - operating as passthrough")
    print()
    print()


def example_preprocessing_with_mock_decomposer():
    """Example 3: Preprocessing with mock decomposer."""
    print("=" * 80)
    print("EXAMPLE 3: Preprocessing with Mock Decomposer")
    print("=" * 80)

    # Create a mock decomposer for demonstration
    class MockDecomposer:
        """Mock German compound decomposer."""
        def decompose(self, word: str):
            """Simple mock: decompose some common compounds."""
            compounds = {
                "schmetterlingshaus": ["schmetterling", "haus"],
                "tageslicht": ["tages", "licht"],
                "abendrot": ["abend", "rot"],
                "blauwal": ["blau", "wal"],
            }
            return compounds.get(word.lower(), [word])

    mock_decomposer = MockDecomposer()

    config = TranslationConfig.default()
    config.preprocessing = PreprocessingConfig(enable_morphological=True)

    pipeline = TranslatorPipeline(config=config, decomposer=mock_decomposer)

    sources = [
        "Das Schmetterlingshaus ist sehr groß.",
        "Das Tageslicht ist schön.",
        "Das Blauwal schwimmt schnell.",
    ]

    print("Translations with compound decomposition:\n")
    for source in sources:
        result = pipeline.translate(source, "de", "en")

        print(f"Source: {source}")
        print(f"Translation: {result.text}")

        # Show decompositions if available
        if "decompositions" in result.metadata:
            decomps = result.metadata["decompositions"]
            decomps_str = ", ".join(
                [f"{k}→{v}" for k, v in list(decomps.items())[:3]]
            )
            print(f"Decompositions applied: {decomps_str}")
            print(f"Compounds found: {result.metadata.get('decomposition_count', 0)}")

        print()

    print()


def example_direct_preprocessor_usage():
    """Example 4: Using the preprocessor directly."""
    print("=" * 80)
    print("EXAMPLE 4: Direct Preprocessor Usage")
    print("=" * 80)

    class MockDecomposer:
        def decompose(self, word: str):
            compounds = {
                "schmetterlingshaus": ["schmetterling", "haus"],
                "tageslicht": ["tages", "licht"],
            }
            return compounds.get(word.lower(), [word])

    preprocessor = GermanCompoundPreprocessor(
        decomposer=MockDecomposer(),
        enable=True
    )

    text = "Das Schmetterlingshaus und das Tageslicht sind schön."
    result = preprocessor.preprocess(text)

    print(f"Original:    {result.original}")
    print(f"Preprocessed: {result.preprocessed}")
    print(f"Tokens:      {result.tokens}")
    print(f"Decompositions:")
    for token, decomp in result.decompositions.items():
        if len(decomp) > 1:
            print(f"  {token} → {decomp}")
    print(f"Metadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")

    print()
    print()


def example_config_driven_toggle():
    """Example 5: Config-driven enable/disable."""
    print("=" * 80)
    print("EXAMPLE 5: Config-Driven Toggle")
    print("=" * 80)

    class MockDecomposer:
        def decompose(self, word: str):
            return {
                "schmetterlingshaus": ["schmetterling", "haus"],
            }.get(word.lower(), [word])

    source = "Das Schmetterlingshaus ist groß."
    reference = "The butterfly house is big."

    decomposer = MockDecomposer()

    print(f"Source: {source}\n")

    # Test 1: Disabled
    print("1. Preprocessing DISABLED:")
    config_disabled = TranslationConfig.default()
    config_disabled.preprocessing = PreprocessingConfig(enable_morphological=False)
    pipeline_disabled = TranslatorPipeline(config=config_disabled, decomposer=decomposer)
    result_disabled = pipeline_disabled.translate(source, "de", "en")
    print(f"   Translation: {result_disabled.text}")
    print()

    # Test 2: Enabled
    print("2. Preprocessing ENABLED:")
    config_enabled = TranslationConfig.default()
    config_enabled.preprocessing = PreprocessingConfig(enable_morphological=True)
    pipeline_enabled = TranslatorPipeline(config=config_enabled, decomposer=decomposer)
    result_enabled = pipeline_enabled.translate(source, "de", "en")
    print(f"   Translation: {result_enabled.text}")
    if "decompositions" in result_enabled.metadata:
        print(f"   Decompositions: {result_enabled.metadata['decompositions']}")
    print()

    print("✓ Both modes work independently via config")
    print()
    print()


def example_experiment_comparison():
    """Example 6: Run comparison experiment."""
    print("=" * 80)
    print("EXAMPLE 6: Morphological Preprocessing Experiment")
    print("=" * 80)

    class MockDecomposer:
        def decompose(self, word: str):
            compounds = {
                "schmetterlingshaus": ["schmetterling", "haus"],
                "tageslicht": ["tages", "licht"],
                "blauwal": ["blau", "wal"],
                "abendrot": ["abend", "rot"],
            }
            return compounds.get(word.lower(), [word])

    experiment = MorphologicalPreprocessingExperiment(decomposer=MockDecomposer())

    sources = [
        "Das Schmetterlingshaus ist sehr groß.",
        "Das Tageslicht ist wunderschön.",
        "Der Blauwal ist ein großes Tier.",
    ]

    references = [
        "The butterfly house is very large.",
        "The daylight is beautiful.",
        "The blue whale is a large animal.",
    ]

    print("Running comparison experiment...\n")
    comparisons = experiment.compare_batch(sources, references, "de", "en")

    experiment.print_comparison_report(comparisons, show_details=True)

    print()
    print()


def example_dynamic_decomposer_injection():
    """Example 7: Dynamically inject/replace decomposer."""
    print("=" * 80)
    print("EXAMPLE 7: Dynamic Decomposer Injection")
    print("=" * 80)

    class DecomposerV1:
        def decompose(self, word: str):
            return {"schmetterlingshaus": ["schmetterling", "haus"]}.get(
                word.lower(), [word]
            )

    class DecomposerV2:
        def decompose(self, word: str):
            # More sophisticated decomposition
            return {
                "schmetterlingshaus": ["schmetterlings", "haus"],
                "blauwal": ["blau", "wal"],
            }.get(word.lower(), [word])

    config = TranslationConfig.default()
    config.preprocessing = PreprocessingConfig(enable_morphological=True)

    pipeline = TranslatorPipeline(config=config, decomposer=DecomposerV1())
    source = "Das Schmetterlingshaus ist groß."

    print(f"Source: {source}\n")

    result1 = pipeline.translate(source, "de", "en")
    print(f"With Decomposer V1: {result1.text}")

    # Dynamically replace decomposer
    pipeline.set_decomposer(DecomposerV2())

    result2 = pipeline.translate(source, "de", "en")
    print(f"With Decomposer V2: {result2.text}")

    print("\n✓ Decomposer can be replaced at runtime without reinitializing pipeline")
    print()
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║  Optional Morphological Preprocessing Examples                            ║")
    print("║  German Compound Decomposition Before Translation                         ║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        example_preprocessing_disabled()
        example_preprocessing_enabled_no_decomposer()
        example_preprocessing_with_mock_decomposer()
        example_direct_preprocessor_usage()
        example_config_driven_toggle()
        example_experiment_comparison()
        example_dynamic_decomposer_injection()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
