"""
Parser Shadai - A Python package for parsing PDFs and images using various LLM providers.
"""

from .llm_providers.base import BaseLLMProvider
from .llm_providers.gemini import GeminiProvider
from .llm_providers.anthropic import AnthropicProvider
from .llm_providers.openai import OpenAIProvider
from .parsers.pdf_parser import PDFParser
from .parsers.image_parser import ImageParser
from .llm_client import LLMClient

# Import agents
from .agents.document_agent import DocumentAgent
from .agents.image_agent import ImageAgent
from .agents.main_agent import MainProcessingAgent, AgentConfig
from .agents.metadata_schemas import MetadataSchema, ChunkNode
from .agents.text_chunker import TextChunker, SmartChunker

__version__ = "0.1.0"
__all__ = [
    # Core LLM providers
    "BaseLLMProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    # Parsers
    "PDFParser",
    "ImageParser",
    # Main client
    "LLMClient",
    # Agents
    "DocumentAgent",
    "ImageAgent",
    "MainProcessingAgent",
    "AgentConfig",
    # Schemas and types
    "DocumentType",
    "MetadataSchema",
    "ChunkNode",
    # Utilities
    "TextChunker",
    "SmartChunker",
]
