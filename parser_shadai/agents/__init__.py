"""
Agents module for Parser Shadai.
Contains intelligent processing agents for documents and images.
"""

from .async_processor import AsyncBatchProcessor, process_chunks_sync
from .cache_manager import (
    CacheConfig,
    CacheManager,
    clear_global_cache,
    get_cache_manager,
)
from .document_agent import DocumentAgent, ProcessingConfig
from .image_agent import ImageAgent, ImageProcessingConfig
from .interfaces import (
    IBatchProcessor,
    ICacheProvider,
    IDocumentProcessor,
    IDocumentTypeDetector,
    ILanguageDetector,
    IMetadataExtractor,
    IProviderFactory,
    IResultCompiler,
    ITextChunker,
)
from .main_agent import AgentConfig, MainProcessingAgent
from .metadata_schemas import (
    ChunkNode,
    ChunkProcessor,
    DocumentType,
    MetadataSchema,
    MetadataSchemas,
)
from .optimized_metadata_extractor import OptimizedMetadataExtractor
from .strategies import (
    LegacyMetadataExtractor,
    OptimizedMetadataStrategy,
    ProviderFactory,
)
from .text_chunker import SmartChunker, TextChunker

__all__ = [
    "DEFAULT_CATEGORIES",
    "get_categories_for_document_type",
    "Language",
    "get_language_prompt",
    "get_language_name",
    "get_supported_languages",
    "LanguageDetector",
    "DocumentAgent",
    "ImageAgent",
    "MainProcessingAgent",
    "ProcessingConfig",
    "ImageProcessingConfig",
    "AgentConfig",
    "DocumentType",
    "MetadataSchema",
    "ChunkNode",
    "MetadataSchemas",
    "ChunkProcessor",
    "TextChunker",
    "SmartChunker",
    "OptimizedMetadataExtractor",
    "AsyncBatchProcessor",
    "process_chunks_sync",
    "CacheConfig",
    "CacheManager",
    "get_cache_manager",
    "clear_global_cache",
    # Phase 4 - SOLID interfaces
    "IMetadataExtractor",
    "IDocumentTypeDetector",
    "ITextChunker",
    "IBatchProcessor",
    "ILanguageDetector",
    "IProviderFactory",
    "IDocumentProcessor",
    "IResultCompiler",
    "ICacheProvider",
    # Phase 4 - Strategy implementations
    "LegacyMetadataExtractor",
    "OptimizedMetadataStrategy",
    "ProviderFactory",
]
