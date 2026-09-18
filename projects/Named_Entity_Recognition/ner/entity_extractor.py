"""
Entity extraction utilities for the Named Entity Recognition project.

This module runs NER inference on input text and converts the model's
token-level predictions into structured entity spans.

The model predicts an entity label for each tokenizer position:

    LOC
    ORG
    PER
    O

This module groups consecutive entity predictions into complete entities
and uses character offsets to recover the corresponding text span.

Example:

    "Angela Merkel visited Berlin."

may produce:

    [
        {
            "text": "Angela Merkel",
            "entity_type": "PER",
            "start": 0,
            "end": 13,
            "score": 0.9984
        },
        {
            "text": "Berlin",
            "entity_type": "LOC",
            "start": 21,
            "end": 28,
            "score": 0.9931
        }
    ]
"""

import torch

from config import ENTITY_TYPES
from ner.model_loader import load_ner_model


def extract_entities(text):
    """
    Extract named entities from input text.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    list
        A list of dictionaries representing the entities detected in the input text.

    Each entity contains:

        text:
            The original text corresponding to the entity.

        entity_type:
            The predicted entity type, such as PER, ORG, or LOC.

        start:
            Character offset where the entity begins.

        end:
            Character offset where the entity ends.

        score:
            Confidence score for the entity prediction.
    """

    if not text:
        return []

    tokenizer, model = load_ner_model()

    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

    offset_mapping = inputs.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_ids = torch.argmax(outputs.logits, dim=-1)[0]
    predicted_scores = torch.max(probabilities, dim=-1).values[0]

    entities = []
    current_entity = None

    for index, label_id in enumerate(predicted_ids):
        label = model.config.id2label[label_id.item()]
        start = offset_mapping[index][0].item()
        end = offset_mapping[index][1].item()

        score = predicted_scores[index].item()

        # Special tokens have an offset of [0, 0].
        if start == end:
            continue

        # Remove whitespace from the entity boundaries.
        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        # Ignore the span if nothing remains after trimming.
        if start >= end:
            continue

        # Ignore positions that are not entity predictions
        if label not in ENTITY_TYPES:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

            continue

        # Start a new entity
        if current_entity is None:
            current_entity = {
                "text": text[start:end],
                "entity_type": label,
                "start": start,
                "end": end,
                "score": score
            }

        elif current_entity["entity_type"] == label:
            current_entity["text"] = text[current_entity["start"]:end]
            current_entity["end"] = end

            # Keep the lowest token confidence as the entity score
            current_entity["score"] = min(current_entity["score"], score)

        else:
            entities.append(current_entity)

            current_entity = {
                "text": text[start:end],
                "entity_type": label,
                "start": start,
                "end": end,
                "score": score
            }

    if current_entity is not None:
        entities.append(current_entity)

    return entities