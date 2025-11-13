"""
Document processing agent for PDFs and text documents.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from parser_shadai.agents.async_processor import process_chunks_sync
from parser_shadai.agents.cache_manager import (
    CacheConfig,
    cache_document_type,
    cache_language,
    get_cache_manager,
    get_cached_document_type,
    get_cached_language,
)
from parser_shadai.agents.language_detector import LanguageDetector
from parser_shadai.agents.metadata_schemas import (
    ChunkNode,
    ChunkProcessor,
    DocumentType,
)
from parser_shadai.agents.text_chunker import ChunkConfig, SmartChunker, TextChunker
from parser_shadai.llm_providers.base import BaseLLMProvider
from parser_shadai.parsers.pdf_parser import PDFParser


@dataclass
class ProcessingConfig:
    """
    Configuration for document processing.

    Optimized defaults (Phase 1 + Phase 2 + Phase 3):
    - chunk_size: 4000 (vs 1000) = 75% fewer chunks
    - auto_detect_language: True = metadata in document's language (with caching)
    - language: "en" = default if auto-detection disabled
    - temperature: 0.2 = more consistent results
    - use_optimized_extraction: True = document-level metadata + async batch processing
    - enable_caching: True = cache expensive LLM operations
    """

    chunk_size: int = 4000  # Increased from 1000 for 75% fewer LLM calls
    overlap_size: int = 400  # Proportional overlap
    use_smart_chunking: bool = True
    extract_images: bool = False  # Disabled for speed (was True)
    max_pages: Optional[int] = None
    temperature: float = 0.2  # Lower for more consistent metadata extraction (was 0.3)
    language: str = "multilingual"  # Default language (production-safe)
    auto_detect_language: bool = (
        False  # DISABLED by default for production reliability (was True)
    )
    # Phase 2 optimization: document-level metadata + async batch processing
    use_optimized_extraction: bool = True  # Enable Phase 2 optimizations
    max_concurrent_chunks: int = 10  # Max concurrent chunk processing
    # Phase 3 optimization: smart caching
    enable_caching: bool = True  # Enable caching for expensive operations
    cache_ttl_seconds: int = 86400  # Cache TTL: 24 hours


class DocumentAgent:
    """
    Intelligent document processing agent.

    This agent follows a structured workflow:
    1. Extract text and metadata from PDF
    2. Determine document type and metadata schema
    3. Chunk text into manageable pieces
    4. Extract metadata for each chunk using LLM
    5. Return structured results
    """

    def __init__(
        self, llm_provider: BaseLLMProvider, config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize document agent.

        Args:
            llm_provider: LLM provider for processing
            config: Processing configuration
        """
        self.llm_provider = llm_provider
        self.config = config or ProcessingConfig()
        self.pdf_parser = PDFParser(llm_provider)

        # Initialize chunker
        chunk_config = ChunkConfig(
            chunk_size=self.config.chunk_size, overlap_size=self.config.overlap_size
        )

        if self.config.use_smart_chunking:
            self.chunker = SmartChunker(chunk_config)
        else:
            self.chunker = TextChunker(chunk_config)

        # Initialize cache manager (Phase 3)
        if self.config.enable_caching:
            cache_config = CacheConfig(
                enabled=True, ttl_seconds=self.config.cache_ttl_seconds
            )
            self.cache = get_cache_manager(config=cache_config)
        else:
            self.cache = None

    def set_language(self, text: str):
        """
        Set language for processing with caching (Phase 3).

        If auto_detect_language is enabled, detect using LLM with caching.
        Otherwise, use the configured default language.

        Args:
            text: Sample text for language detection

        Returns:
            Usage dict if language was detected, None otherwise
        """
        if self.config.auto_detect_language:
            # Check cache first (Phase 3)
            if self.config.enable_caching:
                cached_lang = get_cached_language(content=text[:1000])
                if cached_lang:
                    self.language = cached_lang
                    self.config.language = cached_lang
                    print(f"✓ Language from cache: {cached_lang}")
                    return None  # No LLM usage for cache hit

            # Detect with LLM if not cached
            try:
                self.language_detector = LanguageDetector()
                self.language, usage = self.language_detector.detect_language_with_llm(
                    text, self.llm_provider
                )
                print(f"✓ Detected language: {self.language}")
            except Exception as e:
                # CRITICAL FIX: Fallback to multilingual if detection fails
                print(f"Warning: Language detection failed: {e}, using multilingual")
                self.language = "multilingual"
                usage = {}

            # Force the detected language to be used
            self.config.language = self.language

            # Cache for future requests (Phase 3)
            if self.config.enable_caching:
                cache_language(content=text[:1000], language=self.language)

            return usage
        else:
            # Use configured default language (no LLM call needed)
            self.language = self.config.language or "en"
            self.config.language = self.language
            return None  # No tokens used

    def process_document(
        self,
        document_path: str,
        document_type: DocumentType = DocumentType.GENERAL,
        auto_detect_type: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a document following the complete workflow.

        Args:
            document_path: Path to the document file
            document_type: Type of document (used if auto_detect_type is False)
            auto_detect_type: Whether to automatically detect document type

        Returns:
            Dictionary containing processed document data
        """
        # Initialize usage tracking
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            # Step 1: Extract text and basic metadata from PDF
            pdf_text = self.pdf_parser.extract_text(document_path)
            pdf_metadata = self.pdf_parser.extract_metadata(document_path)
            page_count = self.pdf_parser.get_page_count(document_path)

            # Language detection (if enabled)
            lang_usage = self.set_language(
                pdf_text[:1000] if len(pdf_text) > 1000 else pdf_text
            )
            if lang_usage:
                total_usage["prompt_tokens"] += lang_usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += lang_usage.get(
                    "completion_tokens", 0
                )

            # Step 2: Determine document type if auto-detection is enabled
            if auto_detect_type:
                document_type, type_detection_usage = self._detect_document_type(
                    pdf_text, pdf_metadata
                )
                if type_detection_usage:
                    total_usage["prompt_tokens"] += type_detection_usage.get(
                        "prompt_tokens", 0
                    )
                    total_usage["completion_tokens"] += type_detection_usage.get(
                        "completion_tokens", 0
                    )

            # Step 3: Chunk the text
            chunks_data = self._chunk_document_text(pdf_text, page_count)

            # Step 4: Extract metadata for each chunk
            chunk_nodes, chunk_usage = self._extract_chunk_metadata(
                chunks_data=chunks_data, document_type=document_type, full_text=pdf_text
            )
            total_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += chunk_usage.get("completion_tokens", 0)

            # Step 5: Compile results
            results = self._compile_results(
                document_path, pdf_metadata, chunk_nodes, document_type
            )

            # Add usage information to results
            results["usage"] = total_usage

            return results

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise RuntimeError(f"Document processing failed: {str(e)}")

    def _detect_document_type(self, text: str, metadata: Dict[str, Any]):
        """
        Detect document type using LLM analysis with caching (Phase 3).

        Args:
            text: Document text content
            metadata: PDF metadata

        Returns:
            Tuple of (detected document type, usage dict)
        """
        # Check cache first (Phase 3)
        if self.config.enable_caching:
            cached_type = get_cached_document_type(content=text[:5000])
            if cached_type:
                # Convert string back to DocumentType enum
                type_mapping = {
                    "legal": DocumentType.LEGAL,
                    "medical": DocumentType.MEDICAL,
                    "financial": DocumentType.FINANCIAL,
                    "technical": DocumentType.TECHNICAL,
                    "academic": DocumentType.ACADEMIC,
                    "business": DocumentType.BUSINESS,
                    "general": DocumentType.GENERAL,
                }
                document_type = type_mapping.get(cached_type, DocumentType.GENERAL)
                print(f"✓ Document type from cache: {document_type.value}")
                return document_type, None  # No LLM usage for cache hit

        # Get language prompt for the detected language
        from parser_shadai.agents.language_config import get_language_prompt

        language_prompt = get_language_prompt(self.language)

        # Create prompt for document type detection
        prompt = f"""
        {language_prompt}

        CRITICAL INSTRUCTION: Analyze the document in {self.language.upper()} and respond in {self.language.upper()}.

        You are a professional document classifier. Analyze the following document and determine its type based on content, structure, and context.

        DOCUMENT METADATA: {metadata}
        DOCUMENT CONTENT (first 2000 characters): {text[:2000]}

        AVAILABLE DOCUMENT TYPES:
        - legal: Legal documents, contracts, court papers, legal briefs, regulations
        - medical: Medical reports, patient records, clinical studies, health documents
        - financial: Financial statements, invoices, banking documents, accounting records
        - technical: Technical manuals, documentation, specifications, engineering documents
        - academic: Research papers, theses, academic articles, educational materials
        - business: Business plans, reports, proposals, corporate documents
        - general: General documents that don't fit other specific categories

        CLASSIFICATION INSTRUCTIONS:
        1. Consider the vocabulary, terminology, and language patterns specific to each domain
        2. Analyze document structure and formatting conventions
        3. Look for domain-specific elements (legal clauses, medical terminology, financial figures, etc.)
        4. Consider the context provided by metadata (title, author, subject)
        5. Return ONLY the document type code (e.g., "legal", "medical", etc.)

        DOCUMENT TYPE:
        """

        try:
            response = self.llm_provider.generate_text(
                prompt, temperature=self.config.temperature, max_tokens=50
            )

            detected_type = response.content.strip().lower()

            # Map response to DocumentType enum
            type_mapping = {
                "legal": DocumentType.LEGAL,
                "medical": DocumentType.MEDICAL,
                "financial": DocumentType.FINANCIAL,
                "technical": DocumentType.TECHNICAL,
                "academic": DocumentType.ACADEMIC,
                "business": DocumentType.BUSINESS,
                "general": DocumentType.GENERAL,
            }

            document_type = type_mapping.get(detected_type, DocumentType.GENERAL)

            # Cache the result (Phase 3)
            if self.config.enable_caching:
                cache_document_type(content=text[:5000], document_type=document_type)

            return document_type, response.usage

        except Exception as e:
            print(f"Warning: Could not detect document type: {e}")
            return DocumentType.GENERAL, None

    def _chunk_document_text(self, text: str, page_count: int) -> List[Dict[str, Any]]:
        """
        Chunk document text into manageable pieces.

        Args:
            text: Full document text
            page_count: Number of pages in document

        Returns:
            List of chunk data dictionaries
        """
        if self.config.use_smart_chunking:
            return self.chunker.chunk_with_context(text)
        else:
            return self.chunker.chunk_text(text)

    def _extract_chunk_metadata(
        self,
        chunks_data: List[Dict[str, Any]],
        document_type: DocumentType,
        full_text: str = "",
    ):
        """
        Extract metadata for each chunk using LLM.

        Uses optimized approach (Phase 2) if enabled:
        - Extract document-level metadata once from first 5000 chars
        - Extract minimal metadata per chunk (summary + key concepts)
        - Process chunks concurrently for speed

        Args:
            chunks_data: List of chunk data
            document_type: Type of document
            full_text: Full document text for document-level metadata extraction

        Returns:
            Tuple of (List of ChunkNode objects with metadata, usage dict)
        """
        # Use optimized extraction if enabled (Phase 2)
        if self.config.use_optimized_extraction:
            return self._extract_chunk_metadata_optimized(
                chunks_data=chunks_data,
                document_type=document_type,
                full_text=full_text,
            )

        # Fallback to legacy extraction (original approach)
        return self._extract_chunk_metadata_legacy(
            chunks_data=chunks_data, document_type=document_type
        )

    def _extract_chunk_metadata_optimized(
        self,
        chunks_data: List[Dict[str, Any]],
        document_type: DocumentType,
        full_text: str,
    ):
        """
        Optimized metadata extraction (Phase 2).

        Extracts comprehensive metadata once at document level,
        then minimal metadata per chunk with async batch processing.

        Args:
            chunks_data: List of chunk data
            document_type: Type of document
            full_text: Full document text

        Returns:
            Tuple of (List of ChunkNode objects with metadata, usage dict)
        """
        # Get document sample (first 5000 chars)
        document_sample = full_text[:5000] if len(full_text) > 5000 else full_text

        # Process chunks with async batch processor
        chunk_nodes, results = process_chunks_sync(
            llm_provider=self.llm_provider,
            chunks_data=chunks_data,
            document_sample=document_sample,
            document_type=document_type,
            language=self.config.language,
            temperature=self.config.temperature,
            max_concurrent=self.config.max_concurrent_chunks,
            enable_caching=self.config.enable_caching,
        )

        # Extract usage from results
        total_usage = results.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})

        return chunk_nodes, total_usage

    def _extract_chunk_metadata_legacy(
        self, chunks_data: List[Dict[str, Any]], document_type: DocumentType
    ):
        """
        Legacy metadata extraction (original approach).

        Kept for backward compatibility. Extracts all schema fields
        for each chunk sequentially.

        Args:
            chunks_data: List of chunk data
            document_type: Type of document

        Returns:
            Tuple of (List of ChunkNode objects with metadata, usage dict)
        """
        chunk_processor = ChunkProcessor(
            self.llm_provider, document_type, self.config.language
        )
        chunk_nodes = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        for i, chunk_data in enumerate(chunks_data):
            try:
                chunk_node, chunk_usage = chunk_processor.extract_metadata(
                    chunk=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    page_number=chunk_data.get("page_number"),
                )

                # Add chunk processing metadata
                chunk_node.chunk_index = chunk_data["chunk_index"]
                chunk_node.total_chunks = chunk_data["total_chunks"]

                chunk_nodes.append(chunk_node)

                # Aggregate usage
                if chunk_usage:
                    total_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += chunk_usage.get(
                        "completion_tokens", 0
                    )

            except Exception as e:
                print(f"Warning: Error processing chunk {i + 1}: {e}")
                # Create fallback chunk node
                fallback_node = ChunkNode(
                    chunk_id=chunk_data["chunk_id"],
                    content=chunk_data["content"],
                    metadata={
                        "summary": chunk_data["content"][:200] + "..."
                        if len(chunk_data["content"]) > 200
                        else chunk_data["content"],
                        "document_type": document_type.value,
                        "processing_error": str(e),
                    },
                    page_number=chunk_data.get("page_number"),
                    chunk_index=chunk_data["chunk_index"],
                    total_chunks=chunk_data["total_chunks"],
                )
                chunk_nodes.append(fallback_node)

        return chunk_nodes, total_usage

    def _compile_results(
        self,
        document_path: str,
        pdf_metadata: Dict[str, Any],
        chunk_nodes: List[ChunkNode],
        document_type: DocumentType,
    ) -> Dict[str, Any]:
        """
        Compile final results from processing.

        Args:
            document_path: Path to original document
            pdf_metadata: PDF metadata
            chunk_nodes: Processed chunk nodes
            document_type: Detected document type

        Returns:
            Compiled results dictionary
        """
        # Get chunk statistics
        chunk_stats = self.chunker.get_chunk_statistics(
            [
                {
                    "char_count": node.metadata.get("char_count", len(node.content)),
                    "word_count": node.metadata.get(
                        "word_count", len(node.content.split())
                    ),
                    "page_number": node.page_number,
                }
                for node in chunk_nodes
            ]
        )

        # Extract summary from all chunks
        all_summaries = [
            node.metadata.get("summary", "")
            for node in chunk_nodes
            if node.metadata.get("summary")
        ]
        combined_summary = (
            " ".join(all_summaries)[:1000] + "..."
            if len(" ".join(all_summaries)) > 1000
            else " ".join(all_summaries)
        )

        return {
            "document_info": {
                "file_path": document_path,
                "file_name": os.path.basename(document_path),
                "document_type": document_type.value,
                "page_count": pdf_metadata.get("page_count", 0),
                "processing_timestamp": self._get_timestamp(),
            },
            "pdf_metadata": pdf_metadata,
            "chunk_statistics": chunk_stats,
            "document_summary": combined_summary,
            "chunks": [node.to_dict() for node in chunk_nodes],
            "total_chunks": len(chunk_nodes),
            "processing_config": {
                "chunk_size": self.config.chunk_size,
                "overlap_size": self.config.overlap_size,
                "use_smart_chunking": self.config.use_smart_chunking,
                "temperature": self.config.temperature,
            },
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def process_text_document(
        self,
        text: str,
        document_type: DocumentType = DocumentType.GENERAL,
        auto_detect_type: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a text document (not from PDF).

        Args:
            text: Document text content
            document_type: Type of document
            auto_detect_type: Whether to auto-detect type

        Returns:
            Dictionary containing processed document data
        """
        try:
            # Step 1: Detect document type if needed
            if auto_detect_type:
                document_type = self._detect_document_type(text, {})

            # Step 2: Chunk the text
            chunks_data = self._chunk_document_text(text, 1)

            # Step 3: Extract metadata for each chunk
            chunk_nodes = self._extract_chunk_metadata(chunks_data, document_type)

            # Step 4: Compile results
            results = self._compile_results(
                "text_document", {"page_count": 1}, chunk_nodes, document_type
            )

            return results

        except Exception as e:
            print(f"Error processing text document: {str(e)}")
            raise RuntimeError(f"Text document processing failed: {str(e)}")

    def get_chunk_by_id(
        self, results: Dict[str, Any], chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID from results."""
        for chunk in results.get("chunks", []):
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None
