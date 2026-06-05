"""Example usage of the Machine Translation API."""

from projects.Machine_Translation.api import translate, translate_batch, get_pipeline


def example_single_translation():
    """Example: Translate a single sentence."""
    print("=" * 60)
    print("EXAMPLE 1: Single Translation (EN → DE)")
    print("=" * 60)

    text = "Hello, how are you doing today?"
    result = translate(text, "en", "de")

    print(f"Source ({result.source_lang}): {text}")
    print(f"Target ({result.target_lang}): {result.text}")
    print(f"Model: {result.model_name}")
    print(f"Tokens: {result.source_tokens} → {result.target_tokens}")
    print()


def example_batch_translation():
    """Example: Translate multiple sentences."""
    print("=" * 60)
    print("EXAMPLE 2: Batch Translation (DE → EN)")
    print("=" * 60)

    texts = [
        "Guten Morgen, wie geht es dir?",
        "Das ist ein sehr interessantes Projekt.",
        "Ich lebe in Deutschland.",
    ]

    results = translate_batch(texts, "de", "en")

    for i, result in enumerate(results, 1):
        print(f"[{i}] {result.source_lang.upper()} → {result.target_lang.upper()}")
        print(f"    Source: {texts[i-1]}")
        print(f"    Target: {result.text}")
    print()


def example_multiple_pairs():
    """Example: Translate between different language pairs."""
    print("=" * 60)
    print("EXAMPLE 3: Multiple Language Pairs")
    print("=" * 60)

    # Show available pairs
    pipeline = get_pipeline()
    pairs = pipeline.supported_pairs()
    print(f"Supported language pairs: {pairs}")
    print()

    # Translate the same text through different pairs
    text = "Machine translation is fascinating"
    print(f"Original (EN): {text}\n")

    result_de = translate(text, "en", "de")
    print(f"EN → DE: {result_de.text}")

    # Chain translation: EN → DE → EN (round-trip)
    result_de_to_en = translate(result_de.text, "de", "en")
    print(f"(DE → EN back-translation): {result_de_to_en.text}")
    print()


def example_error_handling():
    """Example: Error handling for unsupported pairs."""
    print("=" * 60)
    print("EXAMPLE 4: Error Handling")
    print("=" * 60)

    try:
        # This pair is not supported in default config
        result = translate("Hello", "en", "fr")
    except ValueError as e:
        print(f"Expected error caught: {e}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  Machine Translation System - Usage Examples           ║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        example_single_translation()
        example_batch_translation()
        example_multiple_pairs()
        example_error_handling()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()