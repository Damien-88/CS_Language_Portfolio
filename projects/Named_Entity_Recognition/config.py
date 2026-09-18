"""
Project-wide configuration for the Named Entity Recognition system.

This module stores configuration values that are shared across multiple parts of the project.

Keeping configuration in one place makes the project easier to maintain:
If we change the NER model, for example, we can update its name here instead of searching through the entire codebase.

-----------------------------------------------------------------------------------------------------------------------
Supported languages
-----------------------------------------------------------------------------------------------------------------------
These are the three languages used throughout this portfolio project.
ISO 639-1 language codes are used because they are short and widely recognized:

en = English
de = German
ru = Russian
"""

SUPPORTED_LANGUAGES = ["en", "de", "ru"]

"""
-----------------------------------------------------------------------------------------------------------------------
Default NER model
-----------------------------------------------------------------------------------------------------------------------
This is a multilingual XLM-RoBERTa model fine-tuned for named entity recognition across 40 languages, including English, 
German, and Russian.

The model currently provides three entity categories:
"""

PER = "person"
ORG = "organization"
LOC = "location"

"""
We will keep the model name in configuration so that we can experiment
with other NER checkpoints later without changing the model-loading code.
"""

MODEL_NAME = "nbroad/jplu-xlm-r-ner-40-lang"

"""
---------------------------------------------------------------------------
Entity types
---------------------------------------------------------------------------
These are the entity types actually supported by our initial model.

Notice that we are using the model's labels rather than the broader taxonomy originally described in the project README. 
This keeps our implementation honest: we cannot evaluate entity categories that the selected model was not trained to 
predict.

"""

ENTITY_TYPES = [
"PER",
"ORG",
"LOC",
]