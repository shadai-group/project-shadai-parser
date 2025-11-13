"""
Text Chunker interface (Protocol).

Defines the contract for text chunking strategies.
"""

from typing import List, Protocol, runtime_checkable

from parser_shadai.domain.entities import Chunk


@runtime_checkable
class TextChunker(Protocol):
    """
    Text Chunker protocol (interface).

    Defines the contract all text chunkers must implement.
    Using Protocol allows structural subtyping without inheritance.

    Implementations: SmartChunker, TextChunker, SemanticChunker, etc.
    """

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text content to chunk

        Returns:
            List of Chunk entities with position information

        Raises:
            ValidationError: If text is empty or invalid
        """
        ...

    def chunk_with_metadata(self, text: str, document_metadata: dict) -> List[Chunk]:
        """
        Split text into chunks with document metadata context.

        Args:
            text: Text content to chunk
            document_metadata: Document metadata for context

        Returns:
            List of Chunk entities with inherited metadata

        Raises:
            ValidationError: If text is empty or invalid
        """
        ...
