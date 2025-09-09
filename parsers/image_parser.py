"""
Image Parser for processing images with LLM providers.
"""

from typing import List, Dict, Any, Optional, Union
from PIL import Image, ImageOps
import base64
import io
import os

from ..llm_providers.base import BaseLLMProvider, LLMResponse


class ImageParser:
    """Image Parser for processing images with LLM providers."""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize image parser with an LLM provider.
        
        Args:
            llm_provider: LLM provider instance for processing images
        """
        self.llm_provider = llm_provider
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Error loading image: {str(e)}")
    
    def load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        Load multiple images from file paths.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of PIL Image objects
        """
        try:
            return [self.load_image(path) for path in image_paths]
        except Exception as e:
            raise RuntimeError(f"Error loading images: {str(e)}")
    
    def resize_image(self, image: Image.Image, max_size: tuple = (1024, 1024), maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize image while optionally maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            max_size: Maximum size as (width, height) tuple
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image object
        """
        try:
            if maintain_aspect_ratio:
                return ImageOps.fit(image, max_size, Image.Resampling.LANCZOS)
            else:
                return image.resize(max_size, Image.Resampling.LANCZOS)
        except Exception as e:
            raise RuntimeError(f"Error resizing image: {str(e)}")
    
    def convert_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded image string
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error converting image to base64: {str(e)}")
    
    def convert_images_to_base64(self, images: List[Image.Image], format: str = 'PNG') -> List[str]:
        """
        Convert multiple PIL Images to base64 strings.
        
        Args:
            images: List of PIL Image objects
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            List of base64 encoded image strings
        """
        try:
            return [self.convert_to_base64(img, format) for img in images]
        except Exception as e:
            raise RuntimeError(f"Error converting images to base64: {str(e)}")
    
    def analyze_image(self, image: Union[str, Image.Image], prompt: str, **kwargs) -> LLMResponse:
        """
        Analyze a single image using LLM.
        
        Args:
            image: Image file path, PIL Image, or base64 string
            prompt: Prompt to send to the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse from the LLM provider
        """
        try:
            if isinstance(image, str):
                # Check if it's base64 or file path
                try:
                    base64.b64decode(image)
                    # It's base64, convert to PIL Image
                    img_data = base64.b64decode(image)
                    pil_img = Image.open(io.BytesIO(img_data))
                except:
                    # It's a file path
                    pil_img = self.load_image(image)
            elif isinstance(image, Image.Image):
                pil_img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Resize if needed
            if kwargs.get('resize', True):
                max_size = kwargs.get('max_size', (1024, 1024))
                pil_img = self.resize_image(pil_img, max_size)
            
            return self.llm_provider.generate_with_images(prompt, [pil_img], **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error analyzing image: {str(e)}")
    
    def analyze_images(self, images: List[Union[str, Image.Image]], prompt: str, **kwargs) -> LLMResponse:
        """
        Analyze multiple images using LLM.
        
        Args:
            images: List of images (file paths, PIL Images, or base64 strings)
            prompt: Prompt to send to the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse from the LLM provider
        """
        try:
            prepared_images = []
            
            for img in images:
                if isinstance(img, str):
                    # Check if it's base64 or file path
                    try:
                        base64.b64decode(img)
                        # It's base64, convert to PIL Image
                        img_data = base64.b64decode(img)
                        pil_img = Image.open(io.BytesIO(img_data))
                    except:
                        # It's a file path
                        pil_img = self.load_image(img)
                elif isinstance(img, Image.Image):
                    pil_img = img
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                
                # Resize if needed
                if kwargs.get('resize', True):
                    max_size = kwargs.get('max_size', (1024, 1024))
                    pil_img = self.resize_image(pil_img, max_size)
                
                prepared_images.append(pil_img)
            
            return self.llm_provider.generate_with_images(prompt, prepared_images, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error analyzing images: {str(e)}")
    
    def extract_text_from_image(self, image: Union[str, Image.Image], **kwargs) -> LLMResponse:
        """
        Extract text from image using OCR via LLM.
        
        Args:
            image: Image file path, PIL Image, or base64 string
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse containing extracted text
        """
        prompt = kwargs.get('prompt', "Extract all text from this image. Return only the text content, no additional commentary.")
        return self.analyze_image(image, prompt, **kwargs)
    
    def describe_image(self, image: Union[str, Image.Image], **kwargs) -> LLMResponse:
        """
        Generate a description of the image.
        
        Args:
            image: Image file path, PIL Image, or base64 string
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse containing image description
        """
        prompt = kwargs.get('prompt', "Describe this image in detail. Include all visible elements, text, colors, objects, and any other relevant information.")
        return self.analyze_image(image, prompt, **kwargs)
    
    def classify_image(self, image: Union[str, Image.Image], categories: Optional[List[str]] = None, **kwargs) -> LLMResponse:
        """
        Classify image into categories.
        
        Args:
            image: Image file path, PIL Image, or base64 string
            categories: List of possible categories (optional)
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse containing classification
        """
        if categories:
            prompt = f"Classify this image into one of these categories: {', '.join(categories)}. Return only the category name."
        else:
            prompt = "Classify this image. Return only the category or type of image."
        
        return self.analyze_image(image, prompt, **kwargs)
    
    def compare_images(self, images: List[Union[str, Image.Image]], prompt: str = None, **kwargs) -> LLMResponse:
        """
        Compare multiple images.
        
        Args:
            images: List of images to compare
            prompt: Custom comparison prompt (optional)
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse containing comparison analysis
        """
        if not prompt:
            prompt = "Compare these images. Describe the similarities and differences between them."
        
        return self.analyze_images(images, prompt, **kwargs)
    
    def get_image_info(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Get basic information about the image.
        
        Args:
            image: Image file path, PIL Image, or base64 string
            
        Returns:
            Dictionary containing image information
        """
        try:
            if isinstance(image, str):
                # Check if it's base64 or file path
                try:
                    base64.b64decode(image)
                    # It's base64, convert to PIL Image
                    img_data = base64.b64decode(image)
                    pil_img = Image.open(io.BytesIO(img_data))
                except:
                    # It's a file path
                    pil_img = self.load_image(image)
            elif isinstance(image, Image.Image):
                pil_img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            return {
                "size": pil_img.size,
                "mode": pil_img.mode,
                "format": pil_img.format,
                "width": pil_img.width,
                "height": pil_img.height,
                "has_transparency": pil_img.mode in ('RGBA', 'LA') or 'transparency' in pil_img.info
            }
        except Exception as e:
            raise RuntimeError(f"Error getting image info: {str(e)}")
