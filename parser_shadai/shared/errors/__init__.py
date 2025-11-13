"""Custom exceptions for parser-shadai."""

from .exceptions import (
    ConfigurationError,
    LanguageDetectionError,
    LLMProviderError,
    MetadataExtractionError,
    ParserError,
    ParsingError,
    ValidationError,
)

__all__ = [
    "ParserError",
    "ParsingError",
    "LanguageDetectionError",
    "MetadataExtractionError",
    "LLMProviderError",
    "ConfigurationError",
    "ValidationError",
]
