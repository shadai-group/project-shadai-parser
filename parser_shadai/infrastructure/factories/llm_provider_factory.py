"""
Factory for creating LLM providers.

Centralizes provider creation logic following the Factory Pattern.
Enables easy switching between providers and testing with mocks.
"""

import logging
from typing import Any, Dict, Union

from parser_shadai.domain.interfaces import LLMProvider
from parser_shadai.llm_providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    GeminiProvider,
    OpenAIProvider,
)
from parser_shadai.shared.errors import ConfigurationError, LLMProviderError

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Follows Factory Pattern + Strategy Pattern:
    - Centralizes object creation logic
    - Hides provider-specific initialization
    - Returns domain interface (LLMProvider)
    - Enables dependency injection
    """

    # Supported provider names
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"

    # Provider class mapping
    _PROVIDERS = {
        GEMINI: GeminiProvider,
        OPENAI: OpenAIProvider,
        ANTHROPIC: AnthropicProvider,
        AZURE: AzureOpenAIProvider,
        BEDROCK: BedrockProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        credentials: Union[str, Dict[str, Any]],
        model: str,
        **kwargs,
    ) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_name: Name of provider (gemini, openai, anthropic, azure, bedrock)
            credentials: API key (str) or credentials dict (for Azure/Bedrock)
            model: Model name (e.g., "gemini-2.0-flash-exp", "gpt-4", etc.)
            **kwargs: Additional provider-specific configuration

        Returns:
            LLM provider instance implementing LLMProvider protocol

        Raises:
            ConfigurationError: If provider_name is unsupported
            LLMProviderError: If provider initialization fails

        Example:
            >>> factory = LLMProviderFactory()
            >>> provider = factory.create(
            ...     provider_name="gemini",
            ...     credentials="api-key-here",
            ...     model="gemini-2.0-flash-exp"
            ... )
            >>> response = provider.generate_text("Hello, world!")
        """
        provider_name_lower = provider_name.lower().strip()

        # Validate provider name
        if provider_name_lower not in cls._PROVIDERS:
            raise ConfigurationError(
                message=f"Unsupported LLM provider: {provider_name}",
                config_key="provider_name",
                config_value=provider_name,
                expected_type=f"One of: {', '.join(cls._PROVIDERS.keys())}",
            )

        provider_class = cls._PROVIDERS[provider_name_lower]

        try:
            logger.info(
                f"Creating LLM provider: {provider_name_lower} (model: {model})"
            )

            # Create provider instance
            if provider_name_lower == cls.AZURE:
                # Azure requires special handling (deployment-based)
                if not isinstance(credentials, dict):
                    raise ConfigurationError(
                        message="Azure provider requires credentials dict",
                        config_key="credentials",
                        config_value=type(credentials).__name__,
                        expected_type="dict with api_key, azure_endpoint, azure_deployment",
                    )

                provider = provider_class(
                    api_key=credentials.get("api_key"),
                    azure_endpoint=credentials.get("azure_endpoint"),
                    azure_deployment=credentials.get("azure_deployment"),
                    api_version=credentials.get("api_version", "2024-08-01-preview"),
                )

            elif provider_name_lower == cls.BEDROCK:
                # Bedrock requires AWS credentials
                if not isinstance(credentials, dict):
                    raise ConfigurationError(
                        message="Bedrock provider requires credentials dict",
                        config_key="credentials",
                        config_value=type(credentials).__name__,
                        expected_type="dict with access_key, secret_key, region",
                    )

                provider = provider_class(
                    model=model,
                    aws_access_key=credentials.get("access_key"),
                    aws_secret_key=credentials.get("secret_key"),
                    region_name=credentials.get("region", "us-east-1"),
                )

            else:
                # Standard providers (Gemini, OpenAI, Anthropic)
                if isinstance(credentials, dict):
                    api_key = credentials.get("api_key")
                else:
                    api_key = credentials

                provider = provider_class(api_key=api_key, model=model)

            logger.info(f"✓ Successfully created {provider_name_lower} provider")
            return provider

        except Exception as e:
            logger.exception(f"Failed to create {provider_name_lower} provider: {e}")
            raise LLMProviderError(
                message=f"Failed to initialize {provider_name} provider",
                provider_name=provider_name_lower,
                model_name=model,
                cause=e,
            )

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """
        Get list of supported provider names.

        Returns:
            List of provider names
        """
        return list(cls._PROVIDERS.keys())

    @classmethod
    def is_provider_supported(cls, provider_name: str) -> bool:
        """
        Check if provider is supported.

        Args:
            provider_name: Provider name to check

        Returns:
            True if supported, False otherwise
        """
        return provider_name.lower().strip() in cls._PROVIDERS
