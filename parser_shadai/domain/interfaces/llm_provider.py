"""
LLM Provider interface (Protocol).

Defines the contract for LLM providers without forcing inheritance.
Following clean architecture and SOLID principles.
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class TokenUsage:
    """
    Standardized token usage across all LLM providers.

    Enforces Liskov Substitution Principle - all providers
    must return the same token usage structure.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        """Validate token counts."""
        if self.input_tokens < 0:
            raise ValueError(f"input_tokens must be >= 0, got {self.input_tokens}")
        if self.output_tokens < 0:
            raise ValueError(f"output_tokens must be >= 0, got {self.output_tokens}")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"input_tokens ({self.input_tokens}) + "
                f"output_tokens ({self.output_tokens})"
            )


@dataclass(frozen=True)
class LLMResponse:
    """
    Standardized LLM response across all providers.

    Enforces Liskov Substitution Principle - all providers
    must return the same response structure.
    """

    content: str
    usage: TokenUsage
    model: str
    finish_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate response."""
        if not self.content:
            raise ValueError("LLM response content cannot be empty")
        if not self.model:
            raise ValueError("Model name must be specified")


@runtime_checkable
class LLMProvider(Protocol):
    """
    LLM Provider protocol (interface).

    Defines the contract all LLM providers must implement.
    Using Protocol instead of ABC allows structural subtyping
    without requiring explicit inheritance.

    This follows:
    - Dependency Inversion Principle (depend on abstraction)
    - Interface Segregation Principle (minimal interface)
    - Liskov Substitution Principle (standardized responses)
    """

    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text completion from prompt.

        Args:
            prompt: Text prompt for the LLM
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Standardized LLMResponse

        Raises:
            LLMProviderError: If generation fails
        """
        ...


@runtime_checkable
class VisionLLMProvider(Protocol):
    """
    Vision-capable LLM Provider protocol.

    Separate interface following Interface Segregation Principle.
    Not all providers support vision - those that don't shouldn't
    be forced to implement this method.

    Providers can implement both LLMProvider and VisionLLMProvider
    if they support vision capabilities.
    """

    def generate_with_images(
        self, prompt: str, images: List[Image.Image], **kwargs
    ) -> LLMResponse:
        """
        Generate text completion from prompt and images.

        Args:
            prompt: Text prompt for the LLM
            images: List of PIL images to analyze
            **kwargs: Provider-specific parameters

        Returns:
            Standardized LLMResponse

        Raises:
            LLMProviderError: If generation fails
        """
        ...
