"""
Async batch processor for Phase 2 optimization.

Enables concurrent processing of multiple chunks with semaphore control
to prevent overwhelming the LLM API.

Expected improvement: 2-3x speedup for chunk processing when combined
with optimized metadata extraction.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from parser_shadai.agents.metadata_schemas import ChunkNode, DocumentType
from parser_shadai.agents.optimized_metadata_extractor import OptimizedMetadataExtractor
from parser_shadai.llm_providers.base import BaseLLMProvider


class AsyncBatchProcessor:
    """
    Process multiple chunks concurrently with semaphore control.

    Features:
    - Concurrent processing with configurable max concurrent tasks
    - Automatic retry on failure with exponential backoff
    - Progress tracking and error aggregation
    - Token usage aggregation across all chunks
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        document_type: DocumentType,
        language: str = "en",
        temperature: float = 0.2,
        max_concurrent: int = 10,
        max_retries: int = 2,
        enable_caching: bool = True,
    ):
        """
        Initialize async batch processor.

        Args:
            llm_provider: LLM provider instance
            document_type: Type of document being processed
            language: Language code for processing (default: "en")
            temperature: Temperature for LLM calls (default: 0.2)
            max_concurrent: Maximum concurrent LLM calls (default: 10)
            max_retries: Maximum retries per chunk on failure (default: 2)
            enable_caching: Enable caching for metadata extraction (default: True)
        """
        self.llm_provider = llm_provider
        self.document_type = document_type
        self.language = language
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.enable_caching = enable_caching

        # Initialize extractor
        self.extractor = OptimizedMetadataExtractor(
            llm_provider=llm_provider,
            document_type=document_type,
            language=language,
            temperature=temperature,
            enable_caching=enable_caching,
        )

        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(value=max_concurrent)

    async def process_chunks_batch(
        self, chunks_data: List[Dict[str, Any]], document_sample: str
    ) -> Tuple[List[ChunkNode], Dict[str, Any]]:
        """
        Process multiple chunks concurrently.

        Args:
            chunks_data: List of chunk dictionaries with content, chunk_id, etc.
            document_sample: Sample text for document-level metadata extraction

        Returns:
            Tuple of (list of ChunkNode objects, aggregated usage dict)
        """
        # Step 1: Extract document-level metadata once
        print("⚡ Extracting document-level metadata...")
        doc_metadata, doc_usage = self.extractor.extract_document_metadata(
            document_sample=document_sample
        )

        # Initialize usage tracking
        total_usage = {
            "prompt_tokens": doc_usage.get("prompt_tokens", 0) if doc_usage else 0,
            "completion_tokens": doc_usage.get("completion_tokens", 0)
            if doc_usage
            else 0,
        }

        # Step 2: Process all chunks concurrently
        print(
            f"⚡ Processing {len(chunks_data)} chunks concurrently (max {self.max_concurrent} at a time)..."
        )

        tasks = [
            self._process_single_chunk_with_retry(chunk_data=chunk_data, retry_count=0)
            for chunk_data in chunks_data
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Aggregate results and usage
        chunk_nodes = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({"chunk_index": i, "error": str(result)})
                # Create fallback chunk node
                chunk_data = chunks_data[i]
                fallback_node = ChunkNode(
                    chunk_id=chunk_data["chunk_id"],
                    content=chunk_data["content"],
                    metadata={
                        "summary": chunk_data["content"][:200] + "..."
                        if len(chunk_data["content"]) > 200
                        else chunk_data["content"],
                        "document_type": self.document_type.value,
                        "processing_error": str(result),
                    },
                    page_number=chunk_data.get("page_number"),
                    chunk_index=chunk_data["chunk_index"],
                    total_chunks=chunk_data["total_chunks"],
                )
                chunk_nodes.append(fallback_node)
            else:
                chunk_node, chunk_usage = result
                chunk_nodes.append(chunk_node)

                # Aggregate usage
                if chunk_usage:
                    total_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += chunk_usage.get(
                        "completion_tokens", 0
                    )

        # Add document metadata to aggregated results
        aggregated_results = {
            "usage": total_usage,
            "document_metadata": doc_metadata,
            "errors": errors if errors else None,
            "chunks_processed": len(chunk_nodes),
            "chunks_with_errors": len(errors),
        }

        print(f"✅ Processed {len(chunk_nodes)} chunks successfully")
        if errors:
            print(f"⚠️  {len(errors)} chunks had errors")

        return chunk_nodes, aggregated_results

    async def _process_single_chunk_with_retry(
        self, chunk_data: Dict[str, Any], retry_count: int
    ) -> Tuple[ChunkNode, Optional[Dict[str, int]]]:
        """
        Process a single chunk with retry logic.

        Args:
            chunk_data: Chunk data dictionary
            retry_count: Current retry attempt count

        Returns:
            Tuple of (ChunkNode, usage dict)
        """
        async with self.semaphore:
            try:
                # Synchronous extractor call wrapped in async
                chunk_node, usage = await asyncio.to_thread(
                    self.extractor.extract_chunk_minimal_metadata,
                    chunk=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    page_number=chunk_data.get("page_number"),
                    chunk_index=chunk_data["chunk_index"],
                    total_chunks=chunk_data["total_chunks"],
                )
                return chunk_node, usage

            except Exception as e:
                if retry_count < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**retry_count
                    print(
                        f"⚠️  Chunk {chunk_data['chunk_id']} failed, retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    return await self._process_single_chunk_with_retry(
                        chunk_data=chunk_data, retry_count=retry_count + 1
                    )
                else:
                    # Max retries exceeded
                    print(
                        f"❌ Chunk {chunk_data['chunk_id']} failed after {self.max_retries} retries: {e}"
                    )
                    raise


def process_chunks_sync(
    llm_provider: BaseLLMProvider,
    chunks_data: List[Dict[str, Any]],
    document_sample: str,
    document_type: DocumentType,
    language: str = "en",
    temperature: float = 0.2,
    max_concurrent: int = 10,
    enable_caching: bool = True,
) -> Tuple[List[ChunkNode], Dict[str, Any]]:
    """
    Synchronous wrapper for async batch processing.

    This function provides a convenient sync interface for the async processor,
    handling event loop creation and execution.

    Args:
        llm_provider: LLM provider instance
        chunks_data: List of chunk dictionaries
        document_sample: Sample text for document-level metadata
        document_type: Type of document
        language: Language code (default: "en")
        temperature: Temperature for LLM calls (default: 0.2)
        max_concurrent: Maximum concurrent tasks (default: 10)
        enable_caching: Enable caching for metadata extraction (default: True)

    Returns:
        Tuple of (list of ChunkNode objects, aggregated results dict)
    """
    # Create processor
    processor = AsyncBatchProcessor(
        llm_provider=llm_provider,
        document_type=document_type,
        language=language,
        temperature=temperature,
        max_concurrent=max_concurrent,
        enable_caching=enable_caching,
    )

    # Run async processing in new event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Execute batch processing
    chunk_nodes, results = loop.run_until_complete(
        processor.process_chunks_batch(
            chunks_data=chunks_data, document_sample=document_sample
        )
    )

    return chunk_nodes, results
