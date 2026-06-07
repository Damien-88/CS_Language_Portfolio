"""Core translation model wrapper using HuggingFace Transformers."""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import LanguagePairConfig, TranslationConfig
from models.base import BaseTranslator, TranslationResult
from preprocessing.language_utils import get_preprocessor, get_postprocessor
from preprocessing.morphological_preprocessing import PreprocessorFactory


class TranslationModel(BaseTranslator):
    """Wrapper for HuggingFace transformer-based translation model."""

    def __init__(
        self,
        pair_config: LanguagePairConfig,
        global_config: TranslationConfig,
        decomposer: Optional[object] = None,
    ):
        """
        Initialize translation model.

        Args:
            pair_config: Configuration for this language pair
            global_config: Global translation configuration
            decomposer: Optional external GermanCompoundDecomposer instance
        """
        self.pair_config = pair_config
        self.global_config = global_config
        self.decomposer = decomposer

        # Load model and tokenizer
        self.device = torch.device(global_config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pair_config.model_name,
            cache_dir=global_config.cache_dir,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pair_config.model_name,
            cache_dir=global_config.cache_dir,
        ).to(self.device)
        self.model.eval()

        # Get language-specific processors
        self.preprocessor = get_preprocessor(pair_config.source_lang)
        self.postprocessor = get_postprocessor(pair_config.target_lang)

        # Initialize optional morphological preprocessor (e.g., German compounds)
        enable_morph = (
            global_config.preprocessing.enable_morphological
            if global_config.preprocessing
            else False
        )
        self.morphological_preprocessor = PreprocessorFactory.create(
            language=pair_config.source_lang,
            decomposer=decomposer,
            enable_morphological=enable_morph,
        )

        # Track preprocessing metadata
        self.preprocessing_metadata = {
            "morphological_enabled": enable_morph,
            "decomposer_available": decomposer is not None,
        }

    def translate(self, text: str) -> TranslationResult:
        """
        Translate a single text.

        Args:
            text: Source text to translate

        Returns:
            TranslationResult with translated text and metadata
        """
        results = self.translate_batch([text])
        return results[0]

    def translate_batch(self, texts: list[str]) -> list[TranslationResult]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts

        Returns:
            List of TranslationResult objects
        """
        # Step 1: Optional morphological preprocessing (e.g., German compound decomposition)
        morpho_results = []
        for text in texts:
            morpho_result = self.morphological_preprocessor.preprocess(text)
            morpho_results.append(morpho_result)

        # Step 2: Standard language preprocessing
        preprocessed = [
            self.preprocessor.preprocess(mr.preprocessed)
            for mr in morpho_results
        ]

        # Step 3: Tokenize
        inputs = self.tokenizer(
            preprocessed,
            max_length=self.pair_config.max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        # Step 4: Generate translations with beam search
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.pair_config.max_length,
                num_beams=self.global_config.num_beams,
                early_stopping=self.global_config.early_stopping,
                no_repeat_ngram_size=2,
            )

        # Step 5: Decode outputs
        raw_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Step 6: Postprocess and create results
        results = []
        for i, raw_translation in enumerate(raw_translations):
            processed_translation = self.postprocessor.postprocess(raw_translation)
            output_tokens = len(self.tokenizer.encode(processed_translation))

            # Build metadata
            metadata = {
                "model": self.pair_config.model_name,
                "beam_size": self.global_config.num_beams,
                "morphological_preprocessing": self.preprocessing_metadata[
                    "morphological_enabled"
                ],
            }

            # Add decomposition info if morphological preprocessing was applied
            if morpho_results[i].metadata.get("decomposition_count", 0) > 0:
                metadata["decompositions"] = morpho_results[i].decompositions
                metadata["decomposition_count"] = morpho_results[i].metadata.get(
                    "decomposition_count", 0
                )
                metadata["raw_input_text"] = texts[i]
                metadata["decomposed_input_text"] = morpho_results[i].preprocessed

            result = TranslationResult(
                text=processed_translation,
                source_lang=self.pair_config.source_lang,
                target_lang=self.pair_config.target_lang,
                model_name=self.pair_config.model_name,
                source_tokens=input_tokens,
                target_tokens=output_tokens,
                metadata=metadata,
            )
            results.append(result)

        return results

    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        """Check if this model supports a language pair."""
        return (
            src_lang.lower() == self.pair_config.source_lang.lower()
            and tgt_lang.lower() == self.pair_config.target_lang.lower()
        )