"""
Factory for creating text chunkers.

Centralizes chunker creation logic following the Factory Pattern.
"""

import logging
from typing import Optional

from parser_shadai.agents.semantic_chunker import (
    SemanticChunkConfig,
    SemanticChunker,
)
from parser_shadai.agents.text_chunker import ChunkConfig, SmartChunker, TextChunker
from parser_shadai.domain.interfaces import TextChunker as TextChunkerProtocol
from parser_shadai.shared import CHUNK_OVERLAP_SIZE_DEFAULT, CHUNK_SIZE_DEFAULT
from parser_shadai.shared.errors import ConfigurationError

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """
    Factory for creating text chunker instances.

    Follows Factory Pattern + Strategy Pattern:
    - Centralizes chunker creation logic
    - Returns domain interface (TextChunker protocol)
    - Enables easy switching between chunking strategies
    """

    # Chunker types
    SEMANTIC = "semantic"
    SMART = "smart"
    TEXT = "text"

    _CHUNKERS = {
        SEMANTIC: SemanticChunker,
        SMART: SmartChunker,
        TEXT: TextChunker,
    }

    @classmethod
    def create(
        cls,
        chunker_type: str = SMART,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        language: str = "en",
    ) -> TextChunkerProtocol:
        """
        Create a text chunker instance.

        Args:
            chunker_type: Type of chunker (semantic, smart, text)
            chunk_size: Chunk size in characters (default: 4000)
            overlap_size: Overlap size in characters (default: 400)
            language: Language for semantic chunker ("en", "es", "multilingual")

        Returns:
            Chunker instance implementing TextChunker protocol

        Raises:
            ConfigurationError: If chunker_type is unsupported

        Example:
            >>> chunker = ChunkerFactory.create(
            ...     chunker_type="semantic",
            ...     chunk_size=4000,
            ...     overlap_size=400,
            ...     language="en"
            ... )
            >>> chunks = chunker.chunk_text("Long text here...")
        """
        chunker_type_lower = chunker_type.lower().strip()

        # Validate chunker type
        if chunker_type_lower not in cls._CHUNKERS:
            raise ConfigurationError(
                message=f"Unsupported chunker type: {chunker_type}",
                config_key="chunker_type",
                config_value=chunker_type,
                expected_type=f"One of: {', '.join(cls._CHUNKERS.keys())}",
            )

        # Use defaults if not provided
        chunk_size = chunk_size or CHUNK_SIZE_DEFAULT
        overlap_size = overlap_size or CHUNK_OVERLAP_SIZE_DEFAULT

        # Create chunker-specific config
        chunker_class = cls._CHUNKERS[chunker_type_lower]

        if chunker_type_lower == cls.SEMANTIC:
            # Semantic chunker uses special config with language
            config = SemanticChunkConfig(
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                language=language,
            )
        else:
            # Smart and text chunkers use standard config
            config = ChunkConfig(
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )

        # Create chunker instance
        chunker = chunker_class(config)

        logger.info(
            f"✓ Created {chunker_type_lower} chunker "
            f"(chunk_size={chunk_size}, overlap={overlap_size})"
        )

        return chunker

    @classmethod
    def create_smart_chunker(
        cls,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
    ) -> SmartChunker:
        """
        Create a smart chunker instance.

        Smart chunker uses sentence/paragraph boundaries for better coherence.

        Args:
            chunk_size: Chunk size in characters
            overlap_size: Overlap size in characters

        Returns:
            SmartChunker instance
        """
        return cls.create(
            chunker_type=cls.SMART,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

    @classmethod
    def create_text_chunker(
        cls,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
    ) -> TextChunker:
        """
        Create a basic text chunker instance.

        Basic chunker splits on character count without considering boundaries.

        Args:
            chunk_size: Chunk size in characters
            overlap_size: Overlap size in characters

        Returns:
            TextChunker instance
        """
        return cls.create(
            chunker_type=cls.TEXT,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """
        Get list of supported chunker types.

        Returns:
            List of chunker types
        """
        return list(cls._CHUNKERS.keys())

    @classmethod
    def is_type_supported(cls, chunker_type: str) -> bool:
        """
        Check if chunker type is supported.

        Args:
            chunker_type: Chunker type to check

        Returns:
            True if supported, False otherwise
        """
        return chunker_type.lower().strip() in cls._CHUNKERS

    @classmethod
    def create_semantic_chunker(
        cls,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        language: str = "en",
    ) -> SemanticChunker:
        """
        Create a semantic chunker instance.

        Semantic chunker uses spaCy for proper sentence boundary detection.
        Correctly handles abbreviations (Dr., U.S.), decimals, and initials.

        Args:
            chunk_size: Chunk size in characters
            overlap_size: Overlap size in characters
            language: Language code ("en", "es", or "multilingual")

        Returns:
            SemanticChunker instance

        Example:
            >>> chunker = ChunkerFactory.create_semantic_chunker(
            ...     chunk_size=4000,
            ...     language="es"
            ... )
            >>> chunks = chunker.chunk_text("El Dr. García trabaja...")
        """
        return cls.create(
            chunker_type=cls.SEMANTIC,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            language=language,
        )
