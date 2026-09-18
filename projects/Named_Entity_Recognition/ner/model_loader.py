"""
Model loading utilities for the Named Entity Recognition project.

This module loads the tokenizer and pretrained transformer model used
for named entity recognition.

Keeping model loading separate from entity extraction gives the project
a clear separation of responsibilities:

    model_loader.py
        Loads the tokenizer and NER model.

    entity_extractor.py
        Uses the loaded model to identify entities in text.

The model checkpoint is configured in config.py.

The model is loaded only when load_ner_model() is called. Importing this
module does not download the model or allocate model memory.
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import MODEL_NAME


def load_ner_model():
    """
    Load the tokenizer and pretrained NER model.

    Returns
    -------
    tuple
        A tuple containing:

        tokenizer:
            The Hugging Face tokenizer associated with the NER model.

        model:
            The transformer model configured for token classification.

    Notes
    -----
    AutoTokenizer and AutoModelForTokenClassification allow the project
    to work with compatible Hugging Face NER checkpoints without changing
    the model-loading implementation.

    The model is downloaded the first time it is loaded and is normally
    retrieved from the local Hugging Face cache on subsequent runs.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    return tokenizer, model