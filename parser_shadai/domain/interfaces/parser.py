"""
Document Parser interface (Protocol).

Defines the contract for document parsers.
"""

from typing import Dict, Any, Protocol, runtime_checkable

from parser_shadai.domain.entities import Document


@runtime_checkable
class DocumentParser(Protocol):
    """
    Document Parser protocol (interface).

    Defines the contract all document parsers must implement.
    Using Protocol allows structural subtyping without inheritance.

    Implementations: PDFParser, ImageParser, DOCXParser, etc.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from document.

        Args:
            file_path: Absolute path to document file

        Returns:
            Extracted text content

        Raises:
            ParsingError: If text extraction fails
        """
        ...

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from document file.

        Args:
            file_path: Absolute path to document file

        Returns:
            Dictionary containing metadata (title, author, etc.)

        Raises:
            ParsingError: If metadata extraction fails
        """
        ...

    def parse(self, file_path: str) -> Document:
        """
        Parse document file into domain entity.

        Main method that combines text extraction and metadata extraction.

        Args:
            file_path: Absolute path to document file

        Returns:
            Document entity with text and metadata

        Raises:
            ParsingError: If parsing fails
        """
        ...
