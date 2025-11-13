"""
Simple dependency injection container.

Provides centralized configuration and object creation
without requiring external DI libraries.

Follows the Service Locator pattern (lightweight alternative to DI containers).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from parser_shadai.domain.interfaces import DocumentParser, LLMProvider, TextChunker
from parser_shadai.infrastructure.factories import (
    ChunkerFactory,
    LLMProviderFactory,
    ParserFactory,
)
from parser_shadai.shared import (
    CHUNK_OVERLAP_SIZE_DEFAULT,
    CHUNK_SIZE_DEFAULT,
    DEFAULT_LANGUAGE,
    TEMPERATURE_DEFAULT,
)

logger = logging.getLogger(__name__)


@dataclass
class ParserConfiguration:
    """
    Configuration for parser-shadai.

    Centralizes all configuration in one place following
    the Configuration Object pattern.
    """

    # LLM Provider config
    llm_provider_name: str = "gemini"
    llm_model: str = "gemini-2.0-flash-exp"
    llm_credentials: Union[str, Dict[str, Any]] = ""
    temperature: float = TEMPERATURE_DEFAULT

    # Chunking config
    chunker_type: str = "smart"
    chunk_size: int = CHUNK_SIZE_DEFAULT
    chunk_overlap: int = CHUNK_OVERLAP_SIZE_DEFAULT

    # Document processing config
    language: str = DEFAULT_LANGUAGE
    auto_detect_language: bool = False  # Disabled for production reliability
    auto_detect_document_type: bool = True

    # OCR config
    extract_text_from_images: bool = True  # Enable OCR fallback
    ocr_dpi: int = 200

    # Performance config
    enable_caching: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours
    max_concurrent_chunks: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.llm_credentials:
            logger.warning("LLM credentials not set - provider creation will fail")

        if self.chunk_size < 100:
            raise ValueError(f"chunk_size must be >= 100, got {self.chunk_size}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )


class Container:
    """
    Simple dependency injection container.

    Manages object lifecycle and dependency wiring.
    Uses factories to create objects with proper dependencies.

    This is a lightweight alternative to full DI frameworks.
    Follows Service Locator + Factory patterns.
    """

    def __init__(self, config: Optional[ParserConfiguration] = None):
        """
        Initialize container.

        Args:
            config: Parser configuration (uses defaults if not provided)
        """
        self.config = config or ParserConfiguration()
        self._llm_provider: Optional[LLMProvider] = None
        self._chunker: Optional[TextChunker] = None

        logger.info("Container initialized with configuration")

    def get_llm_provider(self) -> LLMProvider:
        """
        Get LLM provider instance (singleton per container).

        Returns:
            LLM provider implementing LLMProvider protocol

        Raises:
            ConfigurationError: If provider creation fails
        """
        if self._llm_provider is None:
            logger.info("Creating LLM provider (first call)")
            self._llm_provider = LLMProviderFactory.create(
                provider_name=self.config.llm_provider_name,
                credentials=self.config.llm_credentials,
                model=self.config.llm_model,
            )

        return self._llm_provider

    def get_parser(self, file_path: str) -> DocumentParser:
        """
        Get document parser for file type.

        Args:
            file_path: Path to file (determines parser type)

        Returns:
            Parser instance implementing DocumentParser protocol

        Raises:
            ConfigurationError: If parser creation fails
        """
        llm_provider = self.get_llm_provider()

        return ParserFactory.create(
            file_path=file_path,
            llm_provider=llm_provider,
        )

    def get_chunker(self) -> TextChunker:
        """
        Get text chunker instance (singleton per container).

        Returns:
            Chunker implementing TextChunker protocol
        """
        if self._chunker is None:
            logger.info("Creating text chunker (first call)")
            self._chunker = ChunkerFactory.create(
                chunker_type=self.config.chunker_type,
                chunk_size=self.config.chunk_size,
                overlap_size=self.config.chunk_overlap,
            )

        return self._chunker

    def create_document_agent(self, file_path: str):
        """
        Create DocumentAgent with all dependencies injected.

        This is the main factory method for creating document agents.

        Args:
            file_path: Path to file being processed

        Returns:
            DocumentAgent instance with injected dependencies

        Example:
            >>> config = ParserConfiguration(
            ...     llm_provider_name="gemini",
            ...     llm_credentials="api-key-here",
            ...     llm_model="gemini-2.0-flash-exp"
            ... )
            >>> container = Container(config)
            >>> agent = container.create_document_agent("document.pdf")
            >>> result = agent.process_document("document.pdf")
        """
        from parser_shadai.agents.document_agent import (
            DocumentAgent,
            ProcessingConfig,
        )

        # Get dependencies
        llm_provider = self.get_llm_provider()
        # TODO: Once DocumentAgent is refactored to accept parser + chunker,
        # uncomment these lines:
        # parser = self.get_parser(file_path)
        # chunker = self.get_chunker()

        # Create processing config
        processing_config = ProcessingConfig(
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.chunk_overlap,
            temperature=self.config.temperature,
            language=self.config.language,
            auto_detect_language=self.config.auto_detect_language,
            use_optimized_extraction=True,
            max_concurrent_chunks=self.config.max_concurrent_chunks,
            enable_caching=self.config.enable_caching,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
        )

        # Create agent with dependency injection
        # NOTE: This will be updated once DocumentAgent accepts injected deps
        agent = DocumentAgent(
            llm_provider=llm_provider,
            config=processing_config,
        )

        # TODO: Once DocumentAgent is refactored to accept parser + chunker,
        # we'll inject them here:
        # agent = DocumentAgent(
        #     llm_provider=llm_provider,
        #     parser=parser,
        #     chunker=chunker,
        #     config=processing_config,
        # )

        logger.info("✓ Created DocumentAgent with injected dependencies")
        return agent

    def reset(self) -> None:
        """
        Reset container (clear singletons).

        Useful for testing or when config changes.
        """
        logger.info("Resetting container (clearing singletons)")
        self._llm_provider = None
        self._chunker = None
