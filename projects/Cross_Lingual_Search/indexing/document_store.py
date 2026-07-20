"""
Simple multilingual document store.
This module maintains the mapping between document metadata and their 
corresponding vector representations in the semantic index.
FAISS stores only vectors. This class stores the actual documents and allows
retrieved vector IDs to be resolved back into readable results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class Document:
    """
    Represents a searchable multilingual document.
    """

    document_id: int
    text: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)

class DocumentStore:
    """
    Stores documents independently of the vector index.
    The vector stores embeddings. 
    This class stores: original text, language, metadata
    """

    def __init__(self):
        self.documents = []

    def add_document(self, text, language, metadata = None):
        """ Add a document."""

        doc = Document(
            document_id = len(self.documents),
            text = text,
            language = language,
            metadata = metadata or {}
        )

        self.documents.append(doc)

        return doc
    
    def get(self, document_id):
        """ Retrieve a document by its ID. """

        return self.documents[document_id]
    
    def all_documents(self):
        """ Return all stored documents. """

        return list(self.documents)
    
    def __len__(self):
        """ Return the number of stored documents. """

        return len(self.documents)