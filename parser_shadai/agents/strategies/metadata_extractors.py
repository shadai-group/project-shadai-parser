"""
Metadata extraction strategy implementations (Strategy Pattern + OCP).

Provides different strategies for metadata extraction:
- LegacyMetadataExtractor: Original per-chunk full extraction
- OptimizedMetadataStrategy: Document-level + minimal chunks (Phase 2)

Follows Open/Closed Principle: Open for extension, closed for modification.
"""

from typing import Any, Dict, Optional, Tuple

from parser_shadai.agents.interfaces import IMetadataExtractor
from parser_shadai.agents.metadata_schemas import (
    ChunkNode,
    ChunkProcessor,
    DocumentType,
)
from parser_shadai.agents.optimized_metadata_extractor import OptimizedMetadataExtractor
from parser_shadai.llm_providers.base import BaseLLMProvider


class LegacyMetadataExtractor(IMetadataExtractor):
    """
    Legacy metadata extraction strategy (pre-Phase 2).

    Extracts all schema fields (7-14 fields) for each chunk sequentially.
    Kept for backward compatibility and comparison.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        document_type: DocumentType,
        language: str = "en",
        temperature: float = 0.2,
    ):
        """
        Initialize legacy metadata extractor.

        Args:
            llm_provider: LLM provider instance
            document_type: Type of document
            language: Language code
            temperature: LLM temperature
        """
        self.llm_provider = llm_provider
        self.document_type = document_type
        self.language = language
        self.temperature = temperature

        # Use ChunkProcessor for extraction
        self.chunk_processor = ChunkProcessor(
            llm_provider=llm_provider, document_type=document_type, language=language
        )

    def extract_document_metadata(
        self, document_sample: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Extract document-level metadata (legacy approach).

        Legacy approach doesn't have separate document-level extraction,
        so this returns minimal metadata.

        Args:
            document_sample: Sample text from document

        Returns:
            Tuple of (minimal metadata, None)
        """
        return {
            "document_type": self.document_type.value,
            "summary": document_sample[:500] + "..."
            if len(document_sample) > 500
            else document_sample,
        }, None

    def extract_chunk_metadata(
        self,
        chunk: str,
        chunk_id: str,
        page_number: Optional[int] = None,
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> Tuple[ChunkNode, Optional[Dict[str, int]]]:
        """
        Extract full metadata for a single chunk (legacy approach).

        Extracts all schema fields (7-14 fields) per chunk.

        Args:
            chunk: Chunk content
            chunk_id: Unique identifier
            page_number: Page number
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            Tuple of (ChunkNode with full metadata, usage dict)
        """
        chunk_node, usage = self.chunk_processor.extract_metadata(
            chunk=chunk, chunk_id=chunk_id, page_number=page_number
        )

        # Add chunk indices
        chunk_node.chunk_index = chunk_index
        chunk_node.total_chunks = total_chunks

        return chunk_node, usage


class OptimizedMetadataStrategy(IMetadataExtractor):
    """
    Optimized metadata extraction strategy (Phase 2).

    Extracts comprehensive metadata once at document level,
    then minimal metadata (2 fields) per chunk.

    Benefits:
    - 75-80% fewer LLM calls for metadata extraction
    - Better document-level context
    - Consistent metadata across chunks
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        document_type: DocumentType,
        language: str = "en",
        temperature: float = 0.2,
        enable_caching: bool = True,
    ):
        """
        Initialize optimized metadata extractor.

        Args:
            llm_provider: LLM provider instance
            document_type: Type of document
            language: Language code
            temperature: LLM temperature
            enable_caching: Enable caching
        """
        self.llm_provider = llm_provider
        self.document_type = document_type
        self.language = language
        self.temperature = temperature
        self.enable_caching = enable_caching

        # Use OptimizedMetadataExtractor internally
        self.extractor = OptimizedMetadataExtractor(
            llm_provider=llm_provider,
            document_type=document_type,
            language=language,
            temperature=temperature,
            enable_caching=enable_caching,
        )

    def extract_document_metadata(
        self, document_sample: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Extract comprehensive document-level metadata (optimized approach).

        Extracts all schema fields from document sample (first 5000 chars).
        Uses caching to avoid redundant extractions.

        Args:
            document_sample: Sample text from document

        Returns:
            Tuple of (comprehensive metadata dict, usage dict)
        """
        return self.extractor.extract_document_metadata(document_sample=document_sample)

    def extract_chunk_metadata(
        self,
        chunk: str,
        chunk_id: str,
        page_number: Optional[int] = None,
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> Tuple[ChunkNode, Optional[Dict[str, int]]]:
        """
        Extract minimal metadata for a single chunk (optimized approach).

        Extracts only summary + key_concepts (2 fields) per chunk.

        Args:
            chunk: Chunk content
            chunk_id: Unique identifier
            page_number: Page number
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            Tuple of (ChunkNode with minimal metadata, usage dict)
        """
        return self.extractor.extract_chunk_minimal_metadata(
            chunk=chunk,
            chunk_id=chunk_id,
            page_number=page_number,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
