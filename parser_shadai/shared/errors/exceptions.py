"""
Custom exception hierarchy for parser-shadai.

Following clean architecture principles, all errors are domain-specific
and inherit from a base ParserError exception.
"""

from typing import Any, Dict, Optional


class ParserError(Exception):
    """
    Base exception for all parser-shadai errors.

    All custom exceptions should inherit from this class to maintain
    a consistent error hierarchy.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize parser error.

        Args:
            message: Human-readable error message
            context: Additional context about the error (dict)
            cause: Original exception that caused this error (if any)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        """Return formatted error message with context."""
        base_message = self.message

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_message = f"{base_message} ({context_str})"

        if self.cause:
            base_message = f"{base_message} [Caused by: {type(self.cause).__name__}: {str(self.cause)}]"

        return base_message


class ParsingError(ParserError):
    """
    Raised when document parsing fails.

    Examples:
        - PDF file is corrupted
        - Image file cannot be read
        - Unsupported file format
        - OCR extraction failed
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize parsing error.

        Args:
            message: Error message
            file_path: Path to the file that failed to parse
            file_type: Type of file (pdf, image, etc.)
            cause: Original exception
        """
        context = {}
        if file_path:
            context["file_path"] = file_path
        if file_type:
            context["file_type"] = file_type

        super().__init__(message=message, context=context, cause=cause)


class LanguageDetectionError(ParserError):
    """
    Raised when language detection fails.

    Examples:
        - LLM returns empty response
        - Unsupported language detected
        - Network timeout during detection
    """

    def __init__(
        self,
        message: str,
        sample_text: Optional[str] = None,
        detected_language: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize language detection error.

        Args:
            message: Error message
            sample_text: Text sample that failed detection (first 100 chars)
            detected_language: Language code that was detected (if any)
            cause: Original exception
        """
        context = {}
        if sample_text:
            context["sample_text"] = sample_text[:100]  # Truncate for logging
        if detected_language:
            context["detected_language"] = detected_language

        super().__init__(message=message, context=context, cause=cause)


class MetadataExtractionError(ParserError):
    """
    Raised when metadata extraction fails.

    Examples:
        - LLM fails to extract structured metadata
        - Schema validation fails
        - Chunk processing timeout
    """

    def __init__(
        self,
        message: str,
        chunk_id: Optional[str] = None,
        document_type: Optional[str] = None,
        retry_count: Optional[int] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize metadata extraction error.

        Args:
            message: Error message
            chunk_id: ID of chunk that failed extraction
            document_type: Type of document being processed
            retry_count: Number of retries attempted
            cause: Original exception
        """
        context = {}
        if chunk_id:
            context["chunk_id"] = chunk_id
        if document_type:
            context["document_type"] = document_type
        if retry_count is not None:
            context["retry_count"] = retry_count

        super().__init__(message=message, context=context, cause=cause)


class LLMProviderError(ParserError):
    """
    Raised when LLM provider fails.

    Examples:
        - API key invalid
        - Rate limit exceeded
        - Network timeout
        - Unsupported model
    """

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        status_code: Optional[int] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize LLM provider error.

        Args:
            message: Error message
            provider_name: Name of LLM provider (gemini, openai, etc.)
            model_name: Model being used
            status_code: HTTP status code (if applicable)
            cause: Original exception
        """
        context = {}
        if provider_name:
            context["provider"] = provider_name
        if model_name:
            context["model"] = model_name
        if status_code:
            context["status_code"] = status_code

        super().__init__(message=message, context=context, cause=cause)


class ConfigurationError(ParserError):
    """
    Raised when configuration is invalid.

    Examples:
        - Required config value missing
        - Invalid chunk size
        - Unsupported language code
        - Invalid cache TTL
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that is invalid
            config_value: Invalid value provided
            expected_type: Expected type/format
            cause: Original exception
        """
        context = {}
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)
        if expected_type:
            context["expected_type"] = expected_type

        super().__init__(message=message, context=context, cause=cause)


class ValidationError(ParserError):
    """
    Raised when input validation fails.

    Examples:
        - Empty text provided
        - Invalid file path
        - Unsupported file extension
        - Negative chunk size
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            field_name: Name of field that failed validation
            invalid_value: Invalid value provided
            validation_rule: Rule that was violated
            cause: Original exception
        """
        context = {}
        if field_name:
            context["field"] = field_name
        if invalid_value is not None:
            context["value"] = str(invalid_value)
        if validation_rule:
            context["rule"] = validation_rule

        super().__init__(message=message, context=context, cause=cause)
