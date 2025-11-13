"""
Document entity - Core domain model.

Represents a document being processed by the parser.
Following clean architecture, this is a pure domain entity with:
- No external dependencies
- Business logic encapsulated
- Immutable value objects
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class DocumentType(str, Enum):
    """
    Document type enumeration.

    Maps to metadata schemas for structured extraction.
    """

    GENERAL = "general"
    INVOICE = "invoice"
    CONTRACT = "contract"
    REPORT = "report"
    ARTICLE = "article"
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"
    FINANCIAL = "financial"


class Language(str, Enum):
    """
    Supported language codes (ISO 639-1).

    Limited to languages with tested LLM prompt templates.
    """

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    MULTILINGUAL = "multilingual"

    @classmethod
    def from_string(cls, lang_code: str) -> "Language":
        """
        Convert string to Language enum safely.

        Args:
            lang_code: ISO language code (e.g., "en", "es")

        Returns:
            Language enum value

        Raises:
            ValueError: If language code is unsupported
        """
        lang_code = lang_code.lower().strip()

        # Try exact match first
        for lang in cls:
            if lang.value == lang_code:
                return lang

        # Fallback to multilingual for unsupported languages
        return cls.MULTILINGUAL


@dataclass(frozen=True)
class DocumentMetadata:
    """
    Document metadata value object (immutable).

    Contains high-level information about the document
    extracted from the file or inferred by LLM.
    """

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = 0
    file_size_bytes: int = 0
    file_type: Optional[str] = None  # pdf, png, jpg, etc.

    # Document classification
    document_type: DocumentType = DocumentType.GENERAL
    language: Language = Language.MULTILINGUAL

    # Additional metadata (schema depends on document_type)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def with_language(self, language: Language) -> "DocumentMetadata":
        """
        Return new metadata with updated language.

        Args:
            language: New language value

        Returns:
            New DocumentMetadata instance with updated language
        """
        return DocumentMetadata(
            title=self.title,
            author=self.author,
            subject=self.subject,
            keywords=self.keywords,
            creation_date=self.creation_date,
            modification_date=self.modification_date,
            page_count=self.page_count,
            file_size_bytes=self.file_size_bytes,
            file_type=self.file_type,
            document_type=self.document_type,
            language=language,
            custom_fields=self.custom_fields,
        )

    def with_document_type(self, doc_type: DocumentType) -> "DocumentMetadata":
        """
        Return new metadata with updated document type.

        Args:
            doc_type: New document type

        Returns:
            New DocumentMetadata instance with updated type
        """
        return DocumentMetadata(
            title=self.title,
            author=self.author,
            subject=self.subject,
            keywords=self.keywords,
            creation_date=self.creation_date,
            modification_date=self.modification_date,
            page_count=self.page_count,
            file_size_bytes=self.file_size_bytes,
            file_type=self.file_type,
            document_type=doc_type,
            language=self.language,
            custom_fields=self.custom_fields,
        )


@dataclass
class Document:
    """
    Document entity - Aggregate root.

    Represents a document being processed through the parsing pipeline.
    Contains document content, metadata, and business logic.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    file_path: str = ""

    # Content
    text: str = ""
    raw_content: bytes = field(default_factory=bytes)

    # Metadata
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Processing state
    is_processed: bool = False
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate document after initialization."""
        if not self.file_path and not self.text:
            raise ValueError("Document must have either file_path or text content")

    @property
    def has_text(self) -> bool:
        """Check if document has extractable text."""
        return len(self.text.strip()) > 0

    @property
    def text_length(self) -> int:
        """Get length of text content in characters."""
        return len(self.text)

    @property
    def is_valid(self) -> bool:
        """Check if document is valid for processing."""
        return self.has_text and len(self.processing_errors) == 0

    def set_text(self, text: str) -> None:
        """
        Set document text content.

        Args:
            text: Extracted text content
        """
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text)}")

        self.text = text.strip()

    def set_metadata(self, metadata: DocumentMetadata) -> None:
        """
        Update document metadata.

        Args:
            metadata: New metadata value object
        """
        if not isinstance(metadata, DocumentMetadata):
            raise TypeError(f"Metadata must be DocumentMetadata, got {type(metadata)}")

        self.metadata = metadata

    def add_error(self, error_message: str) -> None:
        """
        Record a processing error.

        Args:
            error_message: Error description
        """
        self.processing_errors.append(error_message)

    def mark_as_processed(self) -> None:
        """Mark document as successfully processed."""
        if not self.is_valid:
            raise ValueError("Cannot mark invalid document as processed")

        self.is_processed = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary for serialization.

        Returns:
            Dictionary representation of document
        """
        return {
            "id": str(self.id),
            "file_path": self.file_path,
            "text_length": self.text_length,
            "metadata": {
                "title": self.metadata.title,
                "author": self.metadata.author,
                "subject": self.metadata.subject,
                "keywords": self.metadata.keywords,
                "page_count": self.metadata.page_count,
                "file_size_bytes": self.metadata.file_size_bytes,
                "file_type": self.metadata.file_type,
                "document_type": self.metadata.document_type.value,
                "language": self.metadata.language.value,
                "custom_fields": self.metadata.custom_fields,
            },
            "is_processed": self.is_processed,
            "is_valid": self.is_valid,
            "processing_errors": self.processing_errors,
        }
