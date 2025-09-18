"""
Main LLM Client for easy access to all providers and parsers.
"""

from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

from .llm_providers.base import BaseLLMProvider
from .llm_providers.gemini import GeminiProvider
from .llm_providers.anthropic import AnthropicProvider
from .llm_providers.openai import OpenAIProvider
from .parsers.pdf_parser import PDFParser
from .parsers.image_parser import ImageParser


class LLMClient:
    """
    Main client for accessing LLM providers and parsers.
    Provides a unified interface for all functionality.
    """
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client with specified provider.
        
        Args:
            provider: LLM provider name ("gemini", "anthropic", "openai")
            api_key: API key for the provider (if None, will try to load from environment)
            model: Model name (optional, will use provider default)
        """
        # Load environment variables
        load_dotenv()
        
        # Set up provider
        self.provider_name = provider.lower()
        self.llm_provider = self._create_provider(provider, api_key, model)
        
        # Initialize parsers
        self.pdf_parser = PDFParser(self.llm_provider)
        self.image_parser = ImageParser(self.llm_provider)
    
    def _create_provider(self, provider: str, api_key: Optional[str], model: Optional[str]) -> BaseLLMProvider:
        """Create the appropriate LLM provider."""
        provider = provider.lower()
        
        if api_key is None:
            api_key = self._get_api_key_from_env(provider)
        
        if provider == "gemini":
            return GeminiProvider(api_key, model)
        elif provider == "anthropic":
            return AnthropicProvider(api_key, model)
        elif provider == "openai":
            return OpenAIProvider(api_key, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: gemini, anthropic, openai")
    
    def _get_api_key_from_env(self, provider: str) -> str:
        """Get API key from environment variables."""
        env_vars = {
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "openai": "OPENAI_API_KEY"
        }
        
        env_var = env_vars.get(provider)
        if not env_var:
            raise ValueError(f"No environment variable mapping for provider: {provider}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found. Please set {env_var} environment variable or pass api_key parameter.")
        
        return api_key
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt."""
        response = self.llm_provider.generate_text(prompt, **kwargs)
        return response.content
    
    def generate_with_images(self, prompt: str, images, **kwargs) -> str:
        """Generate response from prompt and images."""
        response = self.llm_provider.generate_with_images(prompt, images, **kwargs)
        return response.content
    
    def generate_with_documents(self, prompt: str, documents, **kwargs) -> str:
        """Generate response from prompt and documents."""
        response = self.llm_provider.generate_with_documents(prompt, documents, **kwargs)
        return response.content
    
    def parse_pdf(self, pdf_path: str, prompt: str, use_images: bool = True, **kwargs) -> str:
        """Parse PDF with LLM."""
        response = self.pdf_parser.parse_with_llm(pdf_path, prompt, use_images, **kwargs)
        return response.content
    
    def parse_pdf_text(self, pdf_path: str, prompt: str, **kwargs) -> str:
        """Parse PDF text only."""
        response = self.pdf_parser.parse_text_only(pdf_path, prompt, **kwargs)
        return response.content
    
    def parse_pdf_images(self, pdf_path: str, prompt: str, **kwargs) -> str:
        """Parse PDF images only."""
        response = self.pdf_parser.parse_images_only(pdf_path, prompt, **kwargs)
        return response.content
    
    def analyze_image(self, image, prompt: str, **kwargs) -> str:
        """Analyze image with LLM."""
        response = self.image_parser.analyze_image(image, prompt, **kwargs)
        return response.content
    
    def analyze_images(self, images, prompt: str, **kwargs) -> str:
        """Analyze multiple images with LLM."""
        response = self.image_parser.analyze_images(images, prompt, **kwargs)
        return response.content
    
    def extract_text_from_image(self, image, **kwargs) -> str:
        """Extract text from image using OCR."""
        response = self.image_parser.extract_text_from_image(image, **kwargs)
        return response.content
    
    def describe_image(self, image, **kwargs) -> str:
        """Generate description of image."""
        response = self.image_parser.describe_image(image, **kwargs)
        return response.content
    
    def classify_image(self, image, categories=None, **kwargs) -> str:
        """Classify image into categories."""
        response = self.image_parser.classify_image(image, categories, **kwargs)
        return response.content
    
    def compare_images(self, images, prompt=None, **kwargs) -> str:
        """Compare multiple images."""
        response = self.image_parser.compare_images(images, prompt, **kwargs)
        return response.content
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get PDF metadata and page count."""
        metadata = self.pdf_parser.extract_metadata(pdf_path)
        page_count = self.pdf_parser.get_page_count(pdf_path)
        return {**metadata, "page_count": page_count}
    
    def search_pdf(self, pdf_path: str, search_term: str, case_sensitive: bool = False) -> list:
        """Search for text in PDF."""
        return self.pdf_parser.search_text(pdf_path, search_term, case_sensitive)
    
    def get_image_info(self, image) -> Dict[str, Any]:
        """Get image information."""
        return self.image_parser.get_image_info(image)
    
    def switch_provider(self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        """Switch to a different LLM provider."""
        self.llm_provider = self._create_provider(provider, api_key, model)
        self.pdf_parser = PDFParser(self.llm_provider)
        self.image_parser = ImageParser(self.llm_provider)
        self.provider_name = provider.lower()
    
    @property
    def current_provider(self) -> str:
        """Get current provider name."""
        return self.provider_name
