"""
Infrastructure layer for parser-shadai.

This layer contains:
- Factories: Object creation logic
- Container: Dependency injection container
- Adapters: Adapt external libraries to domain interfaces
- Implementations: Concrete implementations of domain interfaces

Following clean architecture:
- Depends on domain layer (interfaces)
- Implements domain interfaces
- Handles external dependencies (LLMs, file I/O, etc.)
"""

from .container import Container, ParserConfiguration
from .factories import ChunkerFactory, LLMProviderFactory, ParserFactory

__all__ = [
    # Container
    "Container",
    "ParserConfiguration",
    # Factories
    "LLMProviderFactory",
    "ParserFactory",
    "ChunkerFactory",
]
