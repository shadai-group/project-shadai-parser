"""Domain entities for parser-shadai."""

from .chunk import Chunk, ChunkMetadata
from .document import Document, DocumentMetadata, DocumentType, Language

__all__ = [
    "Document",
    "DocumentMetadata",
    "DocumentType",
    "Language",
    "Chunk",
    "ChunkMetadata",
]
