import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from indexing.document_store import DocumentStore

store = DocumentStore()

store.add_document("The dog is sleeping.", "en")
store.add_document("Der Hund schläft.", "de")
store.add_document("Собака спит.", "ru")

print(store.get(1)) # Should print the German document with ID 1.