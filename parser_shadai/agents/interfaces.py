"""
Abstract interfaces and protocols for Phase 4 SOLID refactoring.

Defines contracts for key components to enable:
- Dependency Inversion Principle (DIP): Depend on abstractions, not concretions
- Open/Closed Principle (OCP): Open for extension, closed for modification
- Interface Segregation Principle (ISP): Clients shouldn't depend on unused methods
- Liskov Substitution Principle (LSP): Subtypes must be substitutable
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

from parser_shadai.agents.metadata_schemas import ChunkNode, DocumentType
from parser_shadai.llm_providers.base import BaseLLMProvider

# ============================================================================
# Cache Interface (DIP - Depend on abstraction, not concrete CacheManager)
# ============================================================================


class ICacheProvider(Protocol):
    """
    Protocol for cache providers.

    Allows different cache implementations (memory, Redis, etc.)
    without changing client code.
    """

    def get(self, namespace: str, content: str, **kwargs) -> Optional[Any]:
        """Get value from cache."""
        ...

    def set(self, namespace: str, content: str, value: Any, **kwargs) -> None:
        """Set value in cache."""
        ...

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cache."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


# ============================================================================
# Metadata Extraction Strategy (OCP - Open for extension)
# ============================================================================


class IMetadataExtractor(ABC):
    """
    Abstract base class for metadata extraction strategies.

    Enables different extraction approaches:
    - FullMetadataExtractor: Extract all fields per chunk (legacy)
    - OptimizedMetadataExtractor: Document-level + minimal chunks (Phase 2)
    - StreamingMetadataExtractor: Future streaming approach
    """

    @abstractmethod
    def extract_document_metadata(
        self, document_sample: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Extract document-level metadata.

        Args:
            document_sample: Sample text from document

        Returns:
            Tuple of (metadata dict, usage dict)
        """
        pass

    @abstractmethod
    def extract_chunk_metadata(
        self,
        chunk: str,
        chunk_id: str,
        page_number: Optional[int] = None,
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> Tuple[ChunkNode, Optional[Dict[str, int]]]:
        """
        Extract metadata for a single chunk.

        Args:
            chunk: Chunk content
            chunk_id: Unique identifier
            page_number: Page number
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            Tuple of (ChunkNode, usage dict)
        """
        pass


# ============================================================================
# Document Type Detection Strategy (OCP - Open for extension)
# ============================================================================


class IDocumentTypeDetector(ABC):
    """
    Abstract base class for document type detection strategies.

    Enables different detection approaches:
    - LLMDocumentTypeDetector: Use LLM for detection
    - RuleBasedDocumentTypeDetector: Use keywords/patterns
    - HybridDocumentTypeDetector: Combine multiple approaches
    """

    @abstractmethod
    def detect_type(
        self, text: str, metadata: Dict[str, Any]
    ) -> Tuple[DocumentType, Optional[Dict[str, int]]]:
        """
        Detect document type.

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            Tuple of (DocumentType, usage dict)
        """
        pass


# ============================================================================
# Text Chunking Strategy (OCP - Open for extension)
# ============================================================================


class ITextChunker(ABC):
    """
    Abstract base class for text chunking strategies.

    Enables different chunking approaches:
    - SimpleTextChunker: Fixed-size chunks with overlap
    - SmartChunker: Semantic boundary-aware chunking
    - TokenBasedChunker: Token-count based chunking
    """

    @abstractmethod
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text into manageable pieces.

        Args:
            text: Full text to chunk

        Returns:
            List of chunk data dictionaries
        """
        pass

    @abstractmethod
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunk data

        Returns:
            Statistics dictionary
        """
        pass


# ============================================================================
# Batch Processor Interface (DIP - Depend on abstraction)
# ============================================================================


class IBatchProcessor(ABC):
    """
    Abstract base class for batch processing strategies.

    Enables different processing approaches:
    - AsyncBatchProcessor: Concurrent async processing
    - SequentialBatchProcessor: Sequential processing
    - StreamingBatchProcessor: Streaming with backpressure
    """

    @abstractmethod
    async def process_batch(
        self, items: List[Any], processor_fn: Any, **kwargs: Any
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process a batch of items.

        Args:
            items: Items to process
            processor_fn: Processing function
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed items, aggregated results)
        """
        pass


# ============================================================================
# Language Detection Strategy (OCP - Open for extension)
# ============================================================================


class ILanguageDetector(ABC):
    """
    Abstract base class for language detection strategies.

    Enables different detection approaches:
    - LLMLanguageDetector: Use LLM for detection
    - LibraryLanguageDetector: Use langdetect/fasttext
    - CachedLanguageDetector: Wrap with caching
    """

    @abstractmethod
    def detect_language(self, text: str) -> Tuple[str, Optional[Dict[str, int]]]:
        """
        Detect language from text.

        Args:
            text: Sample text

        Returns:
            Tuple of (language code, usage dict)
        """
        pass


# ============================================================================
# Provider Factory Interface (DIP - Depend on abstraction)
# ============================================================================


class IProviderFactory(ABC):
    """
    Abstract base class for LLM provider factories.

    Enables different provider creation strategies:
    - SimpleProviderFactory: Direct instantiation
    - ConfigurableProviderFactory: Load from config
    - PooledProviderFactory: Provider pooling
    """

    @abstractmethod
    def create_provider(
        self, provider_name: str, credentials: Any, **kwargs: Any
    ) -> BaseLLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider_name: Name of provider (gemini, anthropic, openai)
            credentials: API credentials
            **kwargs: Additional configuration

        Returns:
            LLM provider instance
        """
        pass


# ============================================================================
# Document Processor Interface (SRP - Single responsibility)
# ============================================================================


class IDocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    Single responsibility: Process documents and return results.
    Delegates specific tasks to strategy implementations.
    """

    @abstractmethod
    def process_document(
        self,
        document_path: str,
        document_type: Optional[DocumentType] = None,
        auto_detect_type: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a document.

        Args:
            document_path: Path to document
            document_type: Document type (if known)
            auto_detect_type: Whether to auto-detect type

        Returns:
            Processing results dictionary
        """
        pass


# ============================================================================
# Result Compiler Interface (SRP - Single responsibility)
# ============================================================================


class IResultCompiler(ABC):
    """
    Abstract base class for result compilation.

    Single responsibility: Compile and format processing results.
    """

    @abstractmethod
    def compile_results(
        self,
        document_path: str,
        document_metadata: Dict[str, Any],
        chunk_nodes: List[ChunkNode],
        document_type: DocumentType,
        usage: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Compile processing results.

        Args:
            document_path: Original document path
            document_metadata: Document metadata
            chunk_nodes: Processed chunks
            document_type: Detected document type
            usage: Token usage

        Returns:
            Compiled results dictionary
        """
        pass
