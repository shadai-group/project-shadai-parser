"""
Shared utilities for parser-shadai.

Cross-cutting concerns that are used across multiple layers.
"""

from .config import (
    CACHE_DEFAULT_TTL_SECONDS,
    CHUNK_MAX_CONCURRENT,
    CHUNK_OVERLAP_SIZE_DEFAULT,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    DEFAULT_LANGUAGE,
    DOCUMENT_TYPE_SAMPLE_SIZE,
    IMAGE_CONSECUTIVE_FAILURE_THRESHOLD,
    LANGUAGE_DETECTION_SAMPLE_SIZE,
    LANGUAGE_SAMPLE_SIZE,
    MAX_RETRIES,
    MIN_EXTRACTED_TEXT_LENGTH,
    OCR_DEFAULT_DPI,
    RETRY_BACKOFF_MULTIPLIER,
    SUPPORTED_LANGUAGES,
    TEMPERATURE_DEFAULT,
)
from .errors import (
    ConfigurationError,
    LanguageDetectionError,
    LLMProviderError,
    MetadataExtractionError,
    ParserError,
    ParsingError,
    ValidationError,
)

__all__ = [
    # Exceptions
    "ParserError",
    "ParsingError",
    "LanguageDetectionError",
    "MetadataExtractionError",
    "LLMProviderError",
    "ConfigurationError",
    "ValidationError",
    # Constants
    "LANGUAGE_DETECTION_SAMPLE_SIZE",
    "LANGUAGE_SAMPLE_SIZE",
    "DOCUMENT_TYPE_SAMPLE_SIZE",
    "CHUNK_SIZE_DEFAULT",
    "CHUNK_SIZE_MIN",
    "CHUNK_OVERLAP_SIZE_DEFAULT",
    "CHUNK_MAX_CONCURRENT",
    "MIN_EXTRACTED_TEXT_LENGTH",
    "MAX_RETRIES",
    "RETRY_BACKOFF_MULTIPLIER",
    "TEMPERATURE_DEFAULT",
    "OCR_DEFAULT_DPI",
    "CACHE_DEFAULT_TTL_SECONDS",
    "IMAGE_CONSECUTIVE_FAILURE_THRESHOLD",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
]
