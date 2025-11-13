"""
Chunk entity - Represents a text chunk with metadata.

Part of the document processing domain model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass(frozen=True)
class ChunkMetadata:
    """
    Chunk metadata value object (immutable).

    Contains metadata extracted from chunk content by LLM.
    """

    summary: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # Position information
    chunk_index: int = 0
    total_chunks: int = 0

    # Source information
    page_numbers: List[int] = field(default_factory=list)

    # Additional metadata (schema depends on document type)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    Chunk entity - Represents a segment of document text.

    Chunks are the atomic units of document processing.
    Each chunk has:
    - Content (text)
    - Position information
    - Extracted metadata
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    chunk_index: int = 0

    # Content
    content: str = ""

    # Metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Processing state
    is_processed: bool = False
    processing_error: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate chunk after initialization."""
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")

        if self.chunk_index < 0:
            raise ValueError(f"Chunk index must be >= 0, got {self.chunk_index}")

    @property
    def has_metadata(self) -> bool:
        """Check if chunk has extracted metadata."""
        return (
            self.metadata.summary is not None
            or len(self.metadata.key_concepts) > 0
            or len(self.metadata.custom_fields) > 0
        )

    @property
    def content_length(self) -> int:
        """Get length of chunk content in characters."""
        return len(self.content)

    @property
    def is_valid(self) -> bool:
        """Check if chunk is valid."""
        return self.content_length > 0 and self.processing_error is None

    def set_metadata(self, metadata: ChunkMetadata) -> None:
        """
        Update chunk metadata.

        Args:
            metadata: New metadata value object
        """
        if not isinstance(metadata, ChunkMetadata):
            raise TypeError(f"Metadata must be ChunkMetadata, got {type(metadata)}")

        object.__setattr__(self, "metadata", metadata)

    def mark_as_processed(self) -> None:
        """Mark chunk as successfully processed."""
        if not self.is_valid:
            raise ValueError("Cannot mark invalid chunk as processed")

        object.__setattr__(self, "is_processed", True)

    def set_error(self, error_message: str) -> None:
        """
        Record a processing error.

        Args:
            error_message: Error description
        """
        object.__setattr__(self, "processing_error", error_message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary for serialization.

        Returns:
            Dictionary representation of chunk
        """
        return {
            "chunk_id": str(self.id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_length": self.content_length,
            "metadata": {
                "summary": self.metadata.summary,
                "key_concepts": self.metadata.key_concepts,
                "entities": self.metadata.entities,
                "topics": self.metadata.topics,
                "chunk_index": self.metadata.chunk_index,
                "total_chunks": self.metadata.total_chunks,
                "page_numbers": self.metadata.page_numbers,
                "custom_fields": self.metadata.custom_fields,
            },
            "is_processed": self.is_processed,
            "is_valid": self.is_valid,
            "processing_error": self.processing_error,
        }
