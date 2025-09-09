"""
Base LLM Provider class that defines the interface for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO
from PIL import Image


class MessageRole(Enum):
    """Enum for message roles in LLM conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a message in a conversation."""
    role: MessageRole
    content: str
    images: Optional[List[str]] = None  # Base64 encoded images


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All LLM providers must implement these methods.
    """
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the LLM service
            model: Model name to use (optional, provider may have default)
        """
        self.api_key = api_key
        self.model = model
        self._validate_api_key()
    
    @abstractmethod
    def _validate_api_key(self) -> None:
        """Validate the API key. Should raise an exception if invalid."""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text response from a text prompt.
        
        Args:
            prompt: Text prompt to send to the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse object with the generated text
        """
        pass
    
    @abstractmethod
    def generate_with_images(self, prompt: str, images: List[Union[str, Image.Image]], **kwargs) -> LLMResponse:
        """
        Generate response from text prompt and images.
        
        Args:
            prompt: Text prompt to send to the LLM
            images: List of images (file paths, PIL Images, or base64 strings)
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse object with the generated text
        """
        pass
    
    @abstractmethod
    def generate_with_documents(self, prompt: str, documents: List[str], **kwargs) -> LLMResponse:
        """
        Generate response from text prompt and documents.
        
        Args:
            prompt: Text prompt to send to the LLM
            documents: List of document file paths or content
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse object with the generated text
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate response from a conversation.
        
        Args:
            messages: List of Message objects representing the conversation
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse object with the generated text
        """
        pass
    
    def _prepare_image(self, image: Union[str, Image.Image]) -> str:
        """
        Convert image to base64 string for API calls.
        
        Args:
            image: Image as file path, PIL Image, or base64 string
            
        Returns:
            Base64 encoded image string
        """
        if isinstance(image, str):
            # Check if it's already base64
            try:
                base64.b64decode(image)
                return image
            except:
                # Assume it's a file path
                with open(image, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            # Convert PIL Image to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _prepare_images(self, images: List[Union[str, Image.Image]]) -> List[str]:
        """
        Convert multiple images to base64 strings.
        
        Args:
            images: List of images
            
        Returns:
            List of base64 encoded image strings
        """
        return [self._prepare_image(img) for img in images]
    
    def _get_model_name(self) -> str:
        """Get the model name being used."""
        return self.model or "default"
