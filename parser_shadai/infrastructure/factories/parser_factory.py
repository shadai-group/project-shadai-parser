"""
Factory for creating document parsers.

Centralizes parser creation logic following the Factory Pattern.
"""

import logging
from typing import Optional

from parser_shadai.domain.interfaces import DocumentParser, LLMProvider
from parser_shadai.parsers.image_parser import ImageParser
from parser_shadai.parsers.pdf_parser import PDFParser
from parser_shadai.shared.errors import ConfigurationError

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory for creating document parser instances.

    Follows Factory Pattern:
    - Centralizes parser creation logic
    - Returns domain interface (DocumentParser)
    - Enables dependency injection
    """

    # Supported file types
    PDF = "pdf"
    IMAGE = "image"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"

    # File extension to parser type mapping
    _EXTENSION_TO_TYPE = {
        ".pdf": PDF,
        ".png": IMAGE,
        ".jpg": IMAGE,
        ".jpeg": IMAGE,
    }

    @classmethod
    def create(
        cls,
        file_path: str,
        llm_provider: Optional[LLMProvider] = None,
    ) -> DocumentParser:
        """
        Create a parser instance based on file type.

        Args:
            file_path: Path to file (used to determine parser type)
            llm_provider: Optional LLM provider for advanced parsing

        Returns:
            Parser instance implementing DocumentParser protocol

        Raises:
            ConfigurationError: If file type is unsupported

        Example:
            >>> provider = LLMProviderFactory.create("gemini", "api-key", "gemini-2.0-flash")
            >>> parser = ParserFactory.create("document.pdf", provider)
            >>> document = parser.parse("document.pdf")
        """
        parser_type = cls._get_parser_type(file_path)

        logger.info(f"Creating parser for type: {parser_type} (file: {file_path})")

        if parser_type == cls.PDF:
            if not llm_provider:
                raise ConfigurationError(
                    message="PDF parser requires LLM provider",
                    config_key="llm_provider",
                    config_value=None,
                    expected_type="LLMProvider instance",
                )
            parser = PDFParser(llm_provider=llm_provider)

        elif parser_type == cls.IMAGE:
            if not llm_provider:
                raise ConfigurationError(
                    message="Image parser requires LLM provider",
                    config_key="llm_provider",
                    config_value=None,
                    expected_type="LLMProvider instance",
                )
            parser = ImageParser(llm_provider=llm_provider)

        else:
            raise ConfigurationError(
                message="Unsupported file type for parsing",
                config_key="file_type",
                config_value=parser_type,
                expected_type=f"One of: {', '.join([cls.PDF, cls.IMAGE])}",
            )

        logger.info(f"✓ Successfully created {parser_type} parser")
        return parser

    @classmethod
    def create_pdf_parser(cls, llm_provider: LLMProvider) -> PDFParser:
        """
        Create a PDF parser instance.

        Args:
            llm_provider: LLM provider for parsing

        Returns:
            PDFParser instance

        Example:
            >>> provider = LLMProviderFactory.create("gemini", "key", "model")
            >>> parser = ParserFactory.create_pdf_parser(provider)
        """
        logger.info("Creating PDF parser")
        return PDFParser(llm_provider=llm_provider)

    @classmethod
    def create_image_parser(cls, llm_provider: LLMProvider) -> ImageParser:
        """
        Create an image parser instance.

        Args:
            llm_provider: LLM provider for parsing

        Returns:
            ImageParser instance

        Example:
            >>> provider = LLMProviderFactory.create("gemini", "key", "model")
            >>> parser = ParserFactory.create_image_parser(provider)
        """
        logger.info("Creating Image parser")
        return ImageParser(llm_provider=llm_provider)

    @classmethod
    def _get_parser_type(cls, file_path: str) -> str:
        """
        Determine parser type from file path.

        Args:
            file_path: Path to file

        Returns:
            Parser type (pdf, image)

        Raises:
            ConfigurationError: If file extension is unsupported
        """
        # Get file extension (lowercase)
        extension = None
        for ext in cls._EXTENSION_TO_TYPE.keys():
            if file_path.lower().endswith(ext):
                extension = ext
                break

        if not extension:
            raise ConfigurationError(
                message="Unsupported file extension",
                config_key="file_path",
                config_value=file_path,
                expected_type=f"One of: {', '.join(cls._EXTENSION_TO_TYPE.keys())}",
            )

        return cls._EXTENSION_TO_TYPE[extension]

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of supported extensions
        """
        return list(cls._EXTENSION_TO_TYPE.keys())

    @classmethod
    def is_extension_supported(cls, file_path: str) -> bool:
        """
        Check if file extension is supported.

        Args:
            file_path: Path to file

        Returns:
            True if supported, False otherwise
        """
        try:
            cls._get_parser_type(file_path)
            return True
        except ConfigurationError:
            return False
