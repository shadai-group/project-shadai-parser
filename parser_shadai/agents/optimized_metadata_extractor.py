"""
Optimized metadata extractor for Phase 2 optimization.

Key optimization: Extract comprehensive metadata ONCE at document level,
then extract only minimal metadata (summary + key concepts) per chunk.

This reduces LLM calls from 7-14 fields per chunk to just 2 fields per chunk,
plus 1 comprehensive document-level extraction.

Expected improvement: 70-75% reduction in metadata extraction time.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from parser_shadai.agents.cache_manager import (
    cache_document_metadata,
    get_cached_document_metadata,
)
from parser_shadai.agents.language_config import get_language_prompt
from parser_shadai.agents.metadata_schemas import (
    ChunkNode,
    DocumentType,
    MetadataSchemas,
)
from parser_shadai.llm_providers.base import BaseLLMProvider


class OptimizedMetadataExtractor:
    """
    Optimized metadata extraction with document-level + minimal chunk-level approach.

    Strategy:
    1. Extract comprehensive metadata once from document sample (first 5000 chars)
    2. Extract minimal metadata per chunk (summary + key concepts only)
    3. Combine document-level metadata with chunk summaries in final output
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
            document_type: Type of document being processed
            language: Language code for processing (default: "en")
            temperature: Temperature for LLM calls (default: 0.2)
            enable_caching: Enable caching for metadata extraction (default: True)
        """
        self.llm_provider = llm_provider
        self.document_type = document_type
        self.language = language
        self.temperature = temperature
        self.enable_caching = enable_caching
        self.schema = MetadataSchemas.get_schema(document_type=document_type)

        # Cache for document-level metadata
        self._document_metadata: Optional[Dict[str, Any]] = None

    def extract_document_metadata(
        self, document_sample: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Extract comprehensive metadata once for the entire document with caching (Phase 3).

        This method extracts all schema fields from a representative sample
        of the document (typically first 5000 characters).

        Args:
            document_sample: Representative sample of document text

        Returns:
            Tuple of (document metadata dict, usage dict)
        """
        # Check cache first (Phase 3)
        if self.enable_caching:
            cached_metadata = get_cached_document_metadata(
                content=document_sample, document_type=self.document_type
            )
            if cached_metadata:
                self._document_metadata = cached_metadata
                print("✓ Document metadata from cache")
                return cached_metadata, None  # No LLM usage for cache hit

        prompt = self._create_document_metadata_prompt(text=document_sample)

        try:
            response = self.llm_provider.generate_text(
                prompt=prompt, temperature=self.temperature, max_tokens=1000
            )

            metadata = self._parse_json_response(response_text=response.content)

            # Add document type if not present
            if "document_type" not in metadata:
                metadata["document_type"] = self.document_type.value

            # Cache for later use (in-memory)
            self._document_metadata = metadata

            # Cache for future requests (Phase 3)
            if self.enable_caching:
                cache_document_metadata(
                    content=document_sample,
                    document_type=self.document_type,
                    metadata=metadata,
                )

            return metadata, response.usage

        except Exception as e:
            # Fallback metadata
            fallback = {
                "document_type": self.document_type.value,
                "summary": document_sample[:500] + "..."
                if len(document_sample) > 500
                else document_sample,
                "extraction_error": str(e),
            }
            self._document_metadata = fallback
            return fallback, None

    def extract_chunk_minimal_metadata(
        self,
        chunk: str,
        chunk_id: str,
        page_number: Optional[int] = None,
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> Tuple[ChunkNode, Optional[Dict[str, int]]]:
        """
        Extract minimal metadata for a single chunk (summary + key concepts only).

        This dramatically reduces LLM token usage compared to extracting
        all 7-14 schema fields per chunk.

        Args:
            chunk: Text content of the chunk
            chunk_id: Unique identifier for the chunk
            page_number: Page number where chunk was found
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            Tuple of (ChunkNode with minimal metadata, usage dict)
        """
        prompt = self._create_chunk_minimal_prompt(chunk_text=chunk)

        try:
            response = self.llm_provider.generate_text(
                prompt=prompt, temperature=self.temperature, max_tokens=300
            )

            minimal_metadata = self._parse_json_response(response_text=response.content)

            # Combine with document-level metadata for complete context
            combined_metadata = {
                **minimal_metadata,
                "document_type": self.document_type.value,
                "chunk_specific": True,
            }

            chunk_node = ChunkNode(
                chunk_id=chunk_id,
                content=chunk,
                metadata=combined_metadata,
                page_number=page_number,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )

            return chunk_node, response.usage

        except Exception as e:
            # Fallback minimal metadata
            fallback_metadata = {
                "summary": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "key_concepts": [],
                "document_type": self.document_type.value,
                "extraction_error": str(e),
            }

            chunk_node = ChunkNode(
                chunk_id=chunk_id,
                content=chunk,
                metadata=fallback_metadata,
                page_number=page_number,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )

            return chunk_node, None

    def get_cached_document_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get cached document-level metadata.

        Returns:
            Cached document metadata or None if not yet extracted
        """
        return self._document_metadata

    def _create_document_metadata_prompt(self, text: str) -> str:
        """
        Create prompt for extracting comprehensive document-level metadata.

        Args:
            text: Document sample text

        Returns:
            Formatted prompt string
        """
        fields = self.schema.get_all_fields()
        fields_str = ", ".join(fields)
        language_prompt = get_language_prompt(language=self.language)

        prompt = f"""
        {language_prompt}

        CRITICAL INSTRUCTION: Analyze the document in {self.language.upper()} and respond in {self.language.upper()}.

        You are analyzing a {self.document_type.value} document. Extract comprehensive metadata that applies to the ENTIRE document based on this representative sample.

        DOCUMENT TYPE: {self.document_type.value}
        FIELDS TO EXTRACT: {fields_str}

        DOCUMENT SAMPLE:
        {text}

        EXTRACTION INSTRUCTIONS:
        1. **LANGUAGE**: All responses must be in {self.language.upper()}
        2. **DOCUMENT-LEVEL**: Extract information that characterizes the entire document, not just this sample
        3. **COMPREHENSIVE**: Extract all relevant fields from the schema
        4. **NULL VALUES**: Use null or empty string for fields not found in the sample
        5. **JSON FORMAT**: Return only valid JSON, no additional text
        6. **SUMMARY**: Provide a comprehensive document summary in {self.language.upper()}

        JSON Response (in {self.language.upper()}):
        """
        return prompt

    def _create_chunk_minimal_prompt(self, chunk_text: str) -> str:
        """
        Create prompt for extracting minimal chunk-level metadata.

        Only extracts: summary + key_concepts (2 fields instead of 7-14).

        Args:
            chunk_text: Chunk content

        Returns:
            Formatted prompt string
        """
        language_prompt = get_language_prompt(language=self.language)

        prompt = f"""
        {language_prompt}

        CRITICAL INSTRUCTION: Analyze the text in {self.language.upper()} and respond in {self.language.upper()}.

        Extract minimal metadata for this text chunk. Return ONLY these 2 fields in JSON format:

        CHUNK TEXT:
        {chunk_text}

        FIELDS TO EXTRACT:
        1. "summary": A concise 1-2 sentence summary of this chunk's content in {self.language.upper()}
        2. "key_concepts": A list of 3-5 key concepts, terms, or topics covered in this chunk

        INSTRUCTIONS:
        - Keep summary under 200 characters
        - Keep key_concepts as a simple list of strings
        - All text must be in {self.language.upper()}
        - Return only valid JSON

        JSON Response format:
        {{
            "summary": "your summary here",
            "key_concepts": ["concept1", "concept2", "concept3"]
        }}
        """
        return prompt

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, create basic metadata
                return {
                    "summary": response_text[:200] + "..."
                    if len(response_text) > 200
                    else response_text,
                    "raw_response": response_text,
                }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "summary": response_text[:200] + "..."
                if len(response_text) > 200
                else response_text,
                "raw_response": response_text,
            }
