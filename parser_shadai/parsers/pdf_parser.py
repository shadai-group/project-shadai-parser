"""
PDF Parser for extracting text and images from PDF documents.
"""

import PyPDF2
from pdf2image import convert_from_path
from typing import List, Dict, Any
from PIL import Image
import io
import base64

from parser_shadai.llm_providers.base import BaseLLMProvider, LLMResponse

# Optional OCR support
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR fallback will not be available.")


class PDFParser:
    """PDF Parser for extracting content from PDF documents."""

    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize PDF parser with an LLM provider.

        Args:
            llm_provider: LLM provider instance for processing content
        """
        self.llm_provider = llm_provider

    def extract_text_with_ocr(self, pdf_path: str, dpi: int = 200) -> str:
        """
        Extract text from PDF using OCR (Tesseract).

        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion (default: 200)

        Returns:
            Extracted text content via OCR
        """
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("OCR requested but pytesseract is not installed")

        try:
            print(f"⚙️  Using OCR extraction (Tesseract) at {dpi} DPI...")
            images = convert_from_path(pdf_path, dpi=dpi)
            text = ""

            for page_num, image in enumerate(images, 1):
                print(f"  Processing page {page_num}/{len(images)} with OCR...")
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"

            print(f"✓ OCR extraction complete: {len(text)} characters")
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error extracting text with OCR: {str(e)}")

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text content from PDF with multi-strategy fallback.

        Strategy:
        1. Try native text extraction (PyPDF2) - fast, free
        2. If minimal text (<50 chars), try OCR (Tesseract) - medium speed
        3. Return best result with warnings

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (extracted text, strategy used)
        """
        try:
            # Strategy 1: Native extraction (fast, free)
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

                text = text.strip()

            # Check if we got sufficient text
            if len(text) > 50:
                print(f"✓ Native extraction successful: {len(text)} characters")
                return text

            # Strategy 2: OCR fallback (medium speed, low cost)
            print(
                f"⚠️  Native extraction returned minimal text ({len(text)} chars), trying OCR..."
            )

            if TESSERACT_AVAILABLE:
                try:
                    ocr_text = self.extract_text_with_ocr(pdf_path)
                    if len(ocr_text) > 50:
                        return ocr_text
                    else:
                        print(
                            f"⚠️  OCR also returned minimal text ({len(ocr_text)} chars)"
                        )
                        # Return the better result
                        return ocr_text if len(ocr_text) > len(text) else text
                except Exception as ocr_error:
                    print(f"⚠️  OCR extraction failed: {ocr_error}, using native text")
                    return text
            else:
                print(
                    "⚠️  OCR not available (pytesseract not installed), using native text"
                )
                return text

        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

    def extract_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Extract images from PDF pages.

        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion (default: 200)

        Returns:
            List of PIL Images
        """
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            return images
        except Exception as e:
            raise RuntimeError(f"Error extracting images from PDF: {str(e)}")

    def extract_images_base64(self, pdf_path: str, dpi: int = 200) -> List[str]:
        """
        Extract images from PDF pages as base64 strings.

        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion (default: 200)

        Returns:
            List of base64 encoded image strings
        """
        try:
            images = self.extract_images(pdf_path, dpi)
            base64_images = []

            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                base64_images.append(img_b64)

            return base64_images
        except Exception as e:
            raise RuntimeError(f"Error extracting images as base64 from PDF: {str(e)}")

    def parse_with_llm(
        self, pdf_path: str, prompt: str, use_images: bool = True, **kwargs
    ) -> LLMResponse:
        """
        Parse PDF content using LLM with both text and images.

        Args:
            pdf_path: Path to the PDF file
            prompt: Prompt to send to the LLM
            use_images: Whether to include images in the analysis (default: True)
            **kwargs: Additional parameters for the LLM

        Returns:
            LLMResponse from the LLM provider
        """
        try:
            # Extract text content
            text_content = self.extract_text(pdf_path)

            if use_images:
                # Extract images and process with LLM
                images = self.extract_images(pdf_path, kwargs.get("dpi", 200))
                return self.llm_provider.generate_with_images(
                    f"{prompt}\n\nPDF Text Content:\n{text_content}", images, **kwargs
                )
            else:
                # Use only text content
                return self.llm_provider.generate_text(
                    f"{prompt}\n\nPDF Text Content:\n{text_content}", **kwargs
                )
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF with LLM: {str(e)}")

    def parse_text_only(self, pdf_path: str, prompt: str, **kwargs) -> LLMResponse:
        """
        Parse PDF text content only using LLM.

        Args:
            pdf_path: Path to the PDF file
            prompt: Prompt to send to the LLM
            **kwargs: Additional parameters for the LLM

        Returns:
            LLMResponse from the LLM provider
        """
        return self.parse_with_llm(pdf_path, prompt, use_images=False, **kwargs)

    def parse_images_only(self, pdf_path: str, prompt: str, **kwargs) -> LLMResponse:
        """
        Parse PDF images only using LLM.

        Args:
            pdf_path: Path to the PDF file
            prompt: Prompt to send to the LLM
            **kwargs: Additional parameters for the LLM

        Returns:
            LLMResponse from the LLM provider
        """
        try:
            images = self.extract_images(pdf_path, kwargs.get("dpi", 200))
            return self.llm_provider.generate_with_images(prompt, images, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF images with LLM: {str(e)}")

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in the PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages
        """
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            raise RuntimeError(f"Error getting page count: {str(e)}")

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata

                return {
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "subject": metadata.get("/Subject", ""),
                    "creator": metadata.get("/Creator", ""),
                    "producer": metadata.get("/Producer", ""),
                    "creation_date": metadata.get("/CreationDate", ""),
                    "modification_date": metadata.get("/ModDate", ""),
                    "page_count": len(pdf_reader.pages),
                }
        except Exception as e:
            raise RuntimeError(f"Error extracting PDF metadata: {str(e)}")

    def search_text(
        self, pdf_path: str, search_term: str, case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for text in PDF and return matches with page numbers.

        Args:
            pdf_path: Path to the PDF file
            search_term: Text to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of dictionaries with match information
        """
        try:
            matches = []
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()

                    if not case_sensitive:
                        text_lower = text.lower()
                        search_lower = search_term.lower()
                        if search_lower in text_lower:
                            # Find the actual match with original case
                            start_idx = text_lower.find(search_lower)
                            match_text = text[start_idx : start_idx + len(search_term)]
                            matches.append(
                                {
                                    "page": page_num + 1,
                                    "text": match_text,
                                    "context": text[
                                        max(0, start_idx - 50) : start_idx
                                        + len(search_term)
                                        + 50
                                    ],
                                }
                            )
                    else:
                        if search_term in text:
                            start_idx = text.find(search_term)
                            matches.append(
                                {
                                    "page": page_num + 1,
                                    "text": search_term,
                                    "context": text[
                                        max(0, start_idx - 50) : start_idx
                                        + len(search_term)
                                        + 50
                                    ],
                                }
                            )

            return matches
        except Exception as e:
            raise RuntimeError(f"Error searching text in PDF: {str(e)}")
