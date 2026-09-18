"""
Named Entity Recognition package.

This package contains the core components used to extract and normalize
named entities from English, German, and Russian text.

Main components:

model_loader.py
    Loads and configures the transformer-based NER model.

entity_extractor.py
    Runs NER inference and converts model predictions into structured
    entity objects.

entity_normalizer.py
    Performs basic normalization of extracted entity mentions.

The package is intentionally separated into these components so that
model loading, extraction, and normalization can be tested independently.
"""