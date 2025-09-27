"""
Agents module for Parser Shadai.
Contains intelligent processing agents for documents and images.
"""

from .document_agent import DocumentAgent, ProcessingConfig
from .image_agent import ImageAgent, ImageProcessingConfig
from .main_agent import MainProcessingAgent, AgentConfig
from .metadata_schemas import (
    DocumentType,
    MetadataSchema,
    ChunkNode,
    MetadataSchemas,
    ChunkProcessor,
)
from .text_chunker import TextChunker, SmartChunker

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
]
