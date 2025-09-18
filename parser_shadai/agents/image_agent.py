"""
Image processing agent for analyzing images with LLM.
"""

from typing import List, Dict, Any, Optional, Union
import os
from dataclasses import dataclass
from PIL import Image

from parser_shadai.agents.image_categories import get_all_categories
from parser_shadai.agents.language_detector import LanguageDetector

from parser_shadai.llm_providers.base import BaseLLMProvider
from parser_shadai.parsers.image_parser import ImageParser
from parser_shadai.agents.metadata_schemas import DocumentType
from parser_shadai.agents.language_config import get_language_prompt


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
    language: str = "en"
    auto_detect_language: bool = False
    
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
    
    def __init__(self, llm_provider: BaseLLMProvider, config: Optional[ImageProcessingConfig] = None):
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
        self.language = self.language_detector.detect_language_with_llm(text, self.llm_provider) if self.config.auto_detect_language else self.config.language
        print(f"Language: {self.language}")

    def process_image(self, 
                     image_path: str, 
                     document_type: DocumentType = DocumentType.GENERAL) -> Dict[str, Any]:
        """
        Process a single image following the complete workflow.
        
        Args:
            image_path: Path to the image file
            document_type: Type of document context (affects analysis focus)
            
        Returns:
            Dictionary containing processed image data
        """
        try:
            print(f"Processing image: {image_path}")
            
            # Step 1: Load and preprocess image
            print("Step 1: Loading and preprocessing image...")
            image = self.image_parser.load_image(image_path)
            image_info = self.image_parser.get_image_info(image)
            
            # Resize if needed
            if image.size[0] > self.config.max_image_size[0] or image.size[1] > self.config.max_image_size[1]:
                image = self.image_parser.resize_image(image, self.config.max_image_size)
                print(f"Resized image to {image.size}")
            
            # Step 2: Extract text content (OCR)x
            extracted_text = ""
            if self.config.extract_text:
                print("Step 2: Extracting text from image...")
                try:
                    response = self.image_parser.extract_text_from_image(
                        image,
                        temperature=self.config.temperature
                    )
                    self.set_language(response.content)
                    extracted_text = response.content if response.content else ""
                    print(f"Extracted {len(extracted_text)} characters of text")
                except Exception as e:
                    print(f"Warning: Text extraction failed: {e}")
                    extracted_text = ""
            
            # Step 3: Generate detailed description
            description = ""
            if self.config.describe_content:
                print("Step 3: Generating image description...")
                try:
                    response = self.image_parser.describe_image(
                        image,
                        temperature=self.config.temperature
                    )
                    description = response.content if response.content else ""
                    print("Image description generated")
                except Exception as e:
                    print(f"Warning: Description generation failed: {e}")
                    description = ""
            
            # Step 4: Classify image type
            classification = ""
            if self.config.classify_type:
                print("Step 4: Classifying image type...")
                try:
                    response = self.image_parser.classify_image(
                        image,
                        temperature=self.config.temperature,
                        categories = self.config.categories or get_all_categories()
                    )
                    classification = response.content if response.content else ""
                    print(f"Image classified as: {classification}")
                except Exception as e:
                    print(f"Warning: Classification failed: {e}")
                    classification = ""
            
            # Step 5: Extract structured metadata
            print("Step 5: Extracting structured metadata...")
            metadata = self._extract_image_metadata(
                image, 
                extracted_text, 
                description, 
                classification, 
                document_type
            )
            
            # Step 6: Compile results
            print("Step 6: Compiling results...")
            results = self._compile_image_results(
                image_path,
                image_info,
                extracted_text,
                description,
                classification,
                metadata
            )
            
            print("Image processing completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Image processing failed: {str(e)}")
    
    def process_multiple_images(self, 
                               image_paths: List[str], 
                               document_type: DocumentType = DocumentType.GENERAL) -> Dict[str, Any]:
        """
        Process multiple images and provide comparative analysis.
        
        Args:
            image_paths: List of image file paths
            document_type: Type of document context
            
        Returns:
            Dictionary containing processed images data
        """
        try:
            print(f"Processing {len(image_paths)} images...")
            
            processed_images = []
            for i, image_path in enumerate(image_paths):
                print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
                try:
                    image_result = self.process_image(image_path, document_type)
                    processed_images.append(image_result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Add error result
                    processed_images.append({
                        "image_path": image_path,
                        "error": str(e),
                        "processed": False
                    })
            
            # Generate comparative analysis
            print("Generating comparative analysis...")
            comparison = self._generate_comparison_analysis(processed_images)
            
            return {
                "total_images": len(image_paths),
                "successfully_processed": len([img for img in processed_images if img.get("processed", True)]),
                "images": processed_images,
                "comparison_analysis": comparison,
                "processing_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            print(f"Error processing multiple images: {str(e)}")
            raise RuntimeError(f"Multiple image processing failed: {str(e)}")
    
    def _extract_image_metadata(self, 
                               image: Image.Image, 
                               extracted_text: str, 
                               description: str, 
                               classification: str,
                               document_type: DocumentType) -> Dict[str, Any]:
        """
        Extract structured metadata from image analysis.
        
        Args:
            image: PIL Image object
            extracted_text: Extracted text content
            description: Generated description
            classification: Image classification
            document_type: Document type context
            
        Returns:
            Dictionary containing structured metadata
        """
        # Create prompt for metadata extraction
        language_prompt = get_language_prompt(self.config.language)
        prompt = f"""
{language_prompt}

Analyze the following image information and extract structured metadata based on the document type: {document_type.value}

Image Description: {description}
Extracted Text: {extracted_text}
Classification: {classification}
Image Size: {image.size}
Image Mode: {image.mode}

Extract the following metadata fields:
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

Return the result as a JSON object with these fields.
"""
        
        try:
            response = self.llm_provider.generate_text(
                prompt,
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
            else:
                # Fallback metadata
                metadata = {
                    "summary": description[:200] + "..." if len(description) > 200 else description,
                    "document_type": document_type.value,
                    "image_type": classification,
                    "text_content": extracted_text,
                    "raw_response": response.content
                }
            
            # Add technical details
            metadata.update({
                "image_width": image.size[0],
                "image_height": image.size[1],
                "image_mode": image.mode,
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
                "file_size_estimate": len(image.tobytes())
            })
            
            return metadata
            
        except Exception as e:
            print(f"Warning: Metadata extraction failed: {e}")
            return {
                "summary": description[:200] + "..." if len(description) > 200 else description,
                "document_type": document_type.value,
                "image_type": classification,
                "text_content": extracted_text,
                "extraction_error": str(e)
            }
    
    def _compile_image_results(self, 
                              image_path: str, 
                              image_info: Dict[str, Any], 
                              extracted_text: str, 
                              description: str, 
                              classification: str,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
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
                "processing_timestamp": self._get_timestamp()
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
                "temperature": self.config.temperature
            }
        }
    
    def _generate_comparison_analysis(self, processed_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparative analysis of multiple images.
        
        Args:
            processed_images: List of processed image results
            
        Returns:
            Dictionary containing comparison analysis
        """
        try:
            # Extract descriptions and classifications
            descriptions = [img.get("description", "") for img in processed_images if img.get("processed", True)]
            classifications = [img.get("classification", "") for img in processed_images if img.get("processed", True)]
            
            if not descriptions:
                return {"error": "No successfully processed images for comparison"}
            
            # Create comparison prompt
            language_prompt = get_language_prompt(self.config.language)
            prompt = f"""
            {language_prompt}

            Compare the following images and provide analysis:

            Image Descriptions:
            {chr(10).join(f"Image {i+1}: {desc}" for i, desc in enumerate(descriptions))}

            Image Classifications:
            {chr(10).join(f"Image {i+1}: {cls}" for i, cls in enumerate(classifications))}

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
                prompt,
                temperature=self.config.temperature,
                max_tokens=800
            )
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "common_themes": "Analysis not available",
                    "differences": "Analysis not available", 
                    "overall_assessment": response.content,
                    "raw_response": response.content
                }
                
        except Exception as e:
            return {
                "error": f"Comparison analysis failed: {str(e)}",
                "common_themes": "Analysis not available",
                "differences": "Analysis not available"
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()