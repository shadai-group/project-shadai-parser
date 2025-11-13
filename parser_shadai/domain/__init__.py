"""
Domain layer for parser-shadai.

This layer contains:
- Entities: Core business objects (Document, Chunk)
- Value Objects: Immutable data (DocumentMetadata, ChunkMetadata)
- Interfaces: Contracts for infrastructure (Protocols)
- Services: Business logic that spans multiple entities

Following clean architecture principles:
- No external dependencies
- No framework dependencies
- Pure business logic
- Testable in isolation
"""

from .entities import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    DocumentType,
    Language,
)
from .interfaces import (
    DocumentParser,
    LLMProvider,
    LLMResponse,
    TextChunker,
    TokenUsage,
)

__all__ = [
    # Entities
    "Document",
    "DocumentMetadata",
    "DocumentType",
    "Language",
    "Chunk",
    "ChunkMetadata",
    # Interfaces
    "DocumentParser",
    "TextChunker",
    "LLMProvider",
    "LLMResponse",
    "TokenUsage",
]
