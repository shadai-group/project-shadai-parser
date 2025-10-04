"""
Image processing agent for analyzing images with LLM.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from parser_shadai.agents.image_categories import get_all_categories
from parser_shadai.agents.language_config import get_language_prompt
from parser_shadai.agents.language_detector import LanguageDetector
from parser_shadai.agents.metadata_schemas import DocumentType
from parser_shadai.llm_providers.base import BaseLLMProvider
from parser_shadai.parsers.image_parser import ImageParser


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing."""

    max_image_size: tuple = (1024, 1024)
    extract_text: bool = True
    describe_content: bool = True
    classify_type: bool = True
    extract_metadata: bool = True
    temperature: float = 0.3
    categories: List[str] = None
    language: str = None  # Will be set by language detection
    auto_detect_language: bool = True


class ImageAgent:
    """
    Intelligent image processing agent.

    This agent processes images with a different workflow than documents:
    1. Load and preprocess image
    2. Extract text content (OCR)
    3. Generate detailed description
    4. Classify image type
    5. Extract structured metadata
    6. Return comprehensive analysis
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: Optional[ImageProcessingConfig] = None,
    ):
        """
        Initialize image agent.

        Args:
            llm_provider: LLM provider for processing
            config: Image processing configuration
        """
        self.llm_provider = llm_provider
        self.config = config or ImageProcessingConfig()
        self.image_parser = ImageParser(llm_provider)

    def set_language(self, text: str):
        self.language_detector = LanguageDetector()
        if self.config.auto_detect_language:
            self.language, usage = self.language_detector.detect_language_with_llm(
                text, self.llm_provider
            )
            # Force the detected language to be used
            self.config.language = self.language
            return usage
        else:
            # This should not happen since auto_detect_language is now True by default
            raise ValueError(
                "Language detection is required but auto_detect_language is disabled"
            )
            return None

    def process_image(
        self, image_path: str, document_type: DocumentType = DocumentType.GENERAL
    ) -> Dict[str, Any]:
        """
        Process a single image following the complete workflow.

        Args:
            image_path: Path to the image file
            document_type: Type of document context (affects analysis focus)

        Returns:
            Dictionary containing processed image data
        """
        # Initialize usage tracking
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            # Step 1: Load and preprocess image
            image = self.image_parser.load_image(image_path)
            image_info = self.image_parser.get_image_info(image)

            # Resize if needed
            if (
                image.size[0] > self.config.max_image_size[0]
                or image.size[1] > self.config.max_image_size[1]
            ):
                image = self.image_parser.resize_image(
                    image, self.config.max_image_size
                )

            # Step 2: Extract text content (OCR)
            extracted_text = ""
            if self.config.extract_text:
                try:
                    response = self.image_parser.extract_text_from_image(
                        image, temperature=self.config.temperature
                    )
                    lang_usage = self.set_language(response.content)
                    if lang_usage:
                        total_usage["prompt_tokens"] += lang_usage.get(
                            "prompt_tokens", 0
                        )
                        total_usage["completion_tokens"] += lang_usage.get(
                            "completion_tokens", 0
                        )
                    if response.usage:
                        total_usage["prompt_tokens"] += response.usage.get(
                            "prompt_tokens", 0
                        )
                        total_usage["completion_tokens"] += response.usage.get(
                            "completion_tokens", 0
                        )
                    extracted_text = response.content if response.content else ""
                except Exception as e:
                    print(f"Warning: Text extraction failed: {e}")
                    extracted_text = ""

            # Step 3: Generate detailed description
            description = ""
            if self.config.describe_content:
                try:
                    response = self.image_parser.describe_image(
                        image, temperature=self.config.temperature
                    )
                    if response.usage:
                        total_usage["prompt_tokens"] += response.usage.get(
                            "prompt_tokens", 0
                        )
                        total_usage["completion_tokens"] += response.usage.get(
                            "completion_tokens", 0
                        )
                    description = response.content if response.content else ""
                except Exception as e:
                    print(f"Warning: Description generation failed: {e}")
                    description = ""

            # Step 4: Classify image type
            classification = ""
            if self.config.classify_type:
                try:
                    response = self.image_parser.classify_image(
                        image,
                        temperature=self.config.temperature,
                        categories=self.config.categories or get_all_categories(),
                    )
                    if response.usage:
                        total_usage["prompt_tokens"] += response.usage.get(
                            "prompt_tokens", 0
                        )
                        total_usage["completion_tokens"] += response.usage.get(
                            "completion_tokens", 0
                        )
                    classification = response.content if response.content else ""
                except Exception as e:
                    print(f"Warning: Classification failed: {e}")
                    classification = ""

            # Step 5: Extract structured metadata
            metadata, metadata_usage = self._extract_image_metadata(
                image, extracted_text, description, classification, document_type
            )
            if metadata_usage:
                total_usage["prompt_tokens"] += metadata_usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += metadata_usage.get(
                    "completion_tokens", 0
                )

            # Step 6: Compile results
            results = self._compile_image_results(
                image_path,
                image_info,
                extracted_text,
                description,
                classification,
                metadata,
            )

            # Add usage information to results
            results["usage"] = total_usage

            return results

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Image processing failed: {str(e)}")

    def process_multiple_images(
        self, image_paths: List[str], document_type: DocumentType = DocumentType.GENERAL
    ) -> Dict[str, Any]:
        """
        Process multiple images and provide comparative analysis.

        Args:
            image_paths: List of image file paths
            document_type: Type of document context

        Returns:
            Dictionary containing processed images data
        """
        try:
            processed_images = []
            for i, image_path in enumerate(image_paths):
                try:
                    image_result = self.process_image(image_path, document_type)
                    processed_images.append(image_result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Add error result
                    processed_images.append(
                        {"image_path": image_path, "error": str(e), "processed": False}
                    )

            # Generate comparative analysis
            comparison = self._generate_comparison_analysis(processed_images)

            return {
                "total_images": len(image_paths),
                "successfully_processed": len(
                    [img for img in processed_images if img.get("processed", True)]
                ),
                "images": processed_images,
                "comparison_analysis": comparison,
                "processing_timestamp": self._get_timestamp(),
            }

        except Exception as e:
            print(f"Error processing multiple images: {str(e)}")
            raise RuntimeError(f"Multiple image processing failed: {str(e)}")

    def _extract_image_metadata(
        self,
        image: Image.Image,
        extracted_text: str,
        description: str,
        classification: str,
        document_type: DocumentType,
    ):
        """
        Extract structured metadata from image analysis.

        Args:
            image: PIL Image object
            extracted_text: Extracted text content
            description: Generated description
            classification: Image classification
            document_type: Document type context

        Returns:
            Tuple of (Dictionary containing structured metadata, usage dict)
        """
        # Create prompt for metadata extraction
        language_prompt = get_language_prompt(self.config.language)
        prompt = f"""
        {language_prompt}

        CRITICAL INSTRUCTION: You MUST respond in {self.config.language.upper()} language. All extracted metadata, descriptions, and field values must be in {self.config.language.upper()}.

        You are an expert image analyst. Analyze the following image information and extract structured metadata based on the document type: {document_type.value}

        IMAGE ANALYSIS DATA:
        - Description: {description}
        - Extracted Text: {extracted_text}
        - Classification: {classification}
        - Image Size: {image.size}
        - Image Mode: {image.mode}

        METADATA FIELDS TO EXTRACT (all in {self.config.language.upper()}):
        - summary: Brief summary of the image content
        - document_type: {document_type.value}
        - image_type: Type/category of image
        - text_content: Any text found in the image
        - visual_elements: Key visual elements (objects, people, text, etc.)
        - colors: Dominant colors or color scheme
        - layout: Description of layout and composition
        - quality_assessment: Assessment of image quality and clarity
        - relevance_score: How relevant this image is to the document type (0-1)
        - action_items: Any action items or next steps suggested by the image
        - technical_details: Technical aspects (resolution, format, etc.)

        EXTRACTION INSTRUCTIONS:
        1. **LANGUAGE REQUIREMENT**: All responses must be in {self.config.language.upper()} language
        2. **COMPREHENSIVE ANALYSIS**: Provide detailed analysis for each field
        3. **CONTEXT AWARENESS**: Consider the document type context for relevance
        4. **STRUCTURED OUTPUT**: Return only valid JSON format
        5. **LANGUAGE CONSISTENCY**: Ensure all text fields maintain consistency in {self.config.language.upper()}

        JSON Response (in {self.config.language.upper()}):
        """

        try:
            response = self.llm_provider.generate_text(
                prompt, temperature=self.config.temperature, max_tokens=1000
            )

            # Parse JSON response
            import json
            import re

            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
            else:
                # Fallback metadata
                metadata = {
                    "summary": description[:200] + "..."
                    if len(description) > 200
                    else description,
                    "document_type": document_type.value,
                    "image_type": classification,
                    "text_content": extracted_text,
                    "visual_elements": "Analysis failed",
                    "colors": "Unknown",
                    "layout": "Unknown",
                    "quality_assessment": "Unknown",
                    "relevance_score": 0.5,
                    "action_items": "None identified",
                    "technical_details": f"Size: {image.size}, Mode: {image.mode}",
                }

            return metadata, response.usage

        except Exception as e:
            print(f"Warning: Could not extract image metadata: {e}")
            return {
                "summary": description[:200] + "..."
                if len(description) > 200
                else description,
                "document_type": document_type.value,
                "image_type": classification,
                "text_content": extracted_text,
                "error": str(e),
            }, None

    def _generate_image_description(
        self, image: Image.Image, extracted_text: str = ""
    ) -> str:
        """
        Generate detailed description of image content.

        Args:
            image: PIL Image object
            extracted_text: Any text extracted from the image

        Returns:
            Detailed description string
        """
        # Get language prompt for the detected language
        language_prompt = get_language_prompt(self.config.language)

        context_info = (
            f"\nExtracted text from image: {extracted_text}" if extracted_text else ""
        )

        prompt = f"""
        {language_prompt}

        CRITICAL INSTRUCTION: You MUST respond in {self.config.language.upper()} language. Provide a detailed description entirely in {self.config.language.upper()}.

        You are an expert image analyst. Analyze this image and provide a comprehensive description.

        DESCRIPTION REQUIREMENTS (all in {self.config.language.upper()}):
        1. **LANGUAGE**: All descriptions must be in {self.config.language.upper()}
        2. **VISUAL ELEMENTS**: Describe objects, people, text, and visual components
        3. **LAYOUT**: Describe the arrangement and composition
        4. **COLORS**: Mention dominant colors and color schemes
        5. **STYLE**: Describe the visual style (professional, casual, artistic, etc.)
        6. **CONTEXT**: Infer the purpose or context of the image
        7. **DETAILS**: Include relevant details that might be important for document analysis

        {context_info}

        Provide a detailed description in {self.config.language.upper()}:
        """

        try:
            response = self.image_parser.describe_image(
                image,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=800,
            )
            return response.content.strip()
        except Exception as e:
            print(f"Warning: Could not generate image description: {e}")
            return f"Image analysis failed: {str(e)}"

    def _classify_image(self, image: Image.Image, extracted_text: str = "") -> str:
        """
        Classify image type and category.

        Args:
            image: PIL Image object
            extracted_text: Any text extracted from the image

        Returns:
            Classification string
        """
        # Get language prompt for the detected language
        language_prompt = get_language_prompt(self.config.language)

        context_info = f"\nExtracted text: {extracted_text}" if extracted_text else ""
        categories = self.config.categories or get_all_categories()

        prompt = f"""
        {language_prompt}

        CRITICAL INSTRUCTION: You MUST respond in {self.config.language.upper()} language. Provide classification entirely in {self.config.language.upper()}.

        You are an expert image classifier. Analyze this image and classify it into the most appropriate category.

        CLASSIFICATION REQUIREMENTS:
        1. **LANGUAGE**: All responses must be in {self.config.language.upper()}
        2. **PRECISION**: Choose the single most accurate category
        3. **CONTEXT**: Consider the image content, style, and purpose

        AVAILABLE CATEGORIES: {", ".join(categories)}

        {context_info}

        Based on the image analysis, return the most appropriate category in {self.config.language.upper()}:
        """

        try:
            response = self.image_parser.classify_image(
                image,
                categories=categories,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=50,
            )
            return response.content.strip()
        except Exception as e:
            print(f"Warning: Could not classify image: {e}")
            return "unknown"

    def _compile_image_results(
        self,
        image_path: str,
        image_info: Dict[str, Any],
        extracted_text: str,
        description: str,
        classification: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compile final results from image processing.

        Args:
            image_path: Path to original image
            image_info: Basic image information
            extracted_text: Extracted text content
            description: Generated description
            classification: Image classification
            metadata: Structured metadata

        Returns:
            Compiled results dictionary
        """
        return {
            "image_info": {
                "file_path": image_path,
                "file_name": os.path.basename(image_path),
                "size": image_info["size"],
                "mode": image_info["mode"],
                "format": image_info["format"],
                "has_transparency": image_info["has_transparency"],
                "processing_timestamp": self._get_timestamp(),
            },
            "extracted_text": extracted_text,
            "description": description,
            "classification": classification,
            "metadata": metadata,
            "processing_config": {
                "max_image_size": self.config.max_image_size,
                "extract_text": self.config.extract_text,
                "describe_content": self.config.describe_content,
                "classify_type": self.config.classify_type,
                "temperature": self.config.temperature,
            },
        }

    def _generate_comparison_analysis(
        self, processed_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis of multiple images.

        Args:
            processed_images: List of processed image results

        Returns:
            Dictionary containing comparison analysis
        """
        try:
            # Extract descriptions and classifications
            descriptions = [
                img.get("description", "")
                for img in processed_images
                if img.get("processed", True)
            ]
            classifications = [
                img.get("classification", "")
                for img in processed_images
                if img.get("processed", True)
            ]

            if not descriptions:
                return {"error": "No successfully processed images for comparison"}

            # Create comparison prompt
            language_prompt = get_language_prompt(self.config.language)
            prompt = f"""
            {language_prompt}

            Compare the following images and provide analysis:

            Image Descriptions:
            {chr(10).join(f"Image {i + 1}: {desc}" for i, desc in enumerate(descriptions))}

            Image Classifications:
            {chr(10).join(f"Image {i + 1}: {cls}" for i, cls in enumerate(classifications))}

            Provide analysis including:
            - common_themes: Common themes across images
            - differences: Key differences between images
            - overall_assessment: Overall assessment of the image set
            - recommendations: Recommendations based on the images
            - quality_comparison: Comparison of image quality
            - content_relationships: How the images relate to each other

            Return as JSON format.
            """

            response = self.llm_provider.generate_text(
                prompt, temperature=self.config.temperature, max_tokens=800
            )

            # Parse JSON response
            import json
            import re

            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "common_themes": "Analysis not available",
                    "differences": "Analysis not available",
                    "overall_assessment": response.content,
                    "raw_response": response.content,
                }

        except Exception as e:
            return {
                "error": f"Comparison analysis failed: {str(e)}",
                "common_themes": "Analysis not available",
                "differences": "Analysis not available",
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()
