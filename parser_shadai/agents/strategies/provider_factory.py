"""
Provider factory implementation (Factory Pattern + DIP).

Creates LLM provider instances based on provider name and credentials.

Follows:
- Factory Pattern: Centralized object creation
- Dependency Inversion Principle: Depend on abstraction (IProviderFactory)
- Open/Closed Principle: Easy to add new providers
"""

from typing import Any, Union

from parser_shadai import (
    AnthropicProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    GeminiProvider,
    OpenAIProvider,
)
from parser_shadai.agents.interfaces import IProviderFactory
from parser_shadai.llm_providers.base import BaseLLMProvider


class AWSCredentials:
    """AWS credentials for Bedrock provider."""

    def __init__(self, access_key: str, secret_key: str, region: str):
        """
        Initialize AWS credentials.

        Args:
            access_key: AWS access key
            secret_key: AWS secret key
            region: AWS region
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region


class AzureCredentials:
    """Azure credentials for Azure OpenAI provider."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
    ):
        """
        Initialize Azure credentials.

        Args:
            api_key: Azure API key
            azure_endpoint: Azure endpoint URL
            azure_deployment: Azure deployment name
            api_version: Azure API version
        """
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version


class ProviderFactory(IProviderFactory):
    """
    Factory for creating LLM provider instances.

    Supports:
    - Gemini (Google)
    - Anthropic (Claude)
    - OpenAI (GPT)
    - Azure OpenAI
    - Bedrock (AWS) - placeholder for future implementation

    Benefits:
    - Centralized provider creation logic
    - Easy to add new providers
    - Type-safe with proper abstractions
    - Consistent error handling
    """

    # Provider name constants
    GOOGLE = "google"
    GEMINI = "gemini"
    GOOGLE_GENAI = "google_genai"  # Alias for Google/Gemini (Django API)
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    BEDROCK = "bedrock"
    BEDROCK_CONVERSE = "bedrock_converse"  # Alias for Bedrock (Django API)

    def create_provider(
        self, provider_name: str, credentials: Any, **kwargs: Any
    ) -> BaseLLMProvider:
        """
        Create LLM provider instance based on provider name.

        Args:
            provider_name: Name of provider (google, google_genai, gemini, anthropic, openai, azure, bedrock, bedrock_converse)
            credentials: API key string, AWSCredentials, or AzureCredentials object
            **kwargs: Additional configuration (model, timeout, etc.)

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is unsupported or credentials are invalid
        """
        provider_name_lower = provider_name.lower()

        # Handle AWS Bedrock (requires special credentials)
        if provider_name_lower in (self.BEDROCK, self.BEDROCK_CONVERSE):
            return self._create_bedrock_provider(credentials=credentials, **kwargs)

        # Handle Azure OpenAI (requires special credentials)
        if provider_name_lower == self.AZURE:
            return self._create_azure_provider(credentials=credentials, **kwargs)

        # Handle API key-based providers
        if not isinstance(credentials, str):
            raise ValueError(
                f"Expected API key string for {provider_name}, "
                f"got {type(credentials).__name__}"
            )

        # Create provider based on name
        if provider_name_lower in (self.GOOGLE, self.GEMINI, self.GOOGLE_GENAI):
            return self._create_gemini_provider(api_key=credentials, **kwargs)
        elif provider_name_lower == self.ANTHROPIC:
            return self._create_anthropic_provider(api_key=credentials, **kwargs)
        elif provider_name_lower == self.OPENAI:
            return self._create_openai_provider(api_key=credentials, **kwargs)
        else:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: {self.GOOGLE}, {self.GOOGLE_GENAI}, {self.GEMINI}, "
                f"{self.ANTHROPIC}, {self.OPENAI}, {self.AZURE}, {self.BEDROCK}, {self.BEDROCK_CONVERSE}"
            )

    def _create_gemini_provider(self, api_key: str, **kwargs: Any) -> GeminiProvider:
        """
        Create Gemini provider instance.

        Args:
            api_key: Google API key
            **kwargs: Additional configuration

        Returns:
            GeminiProvider instance
        """
        return GeminiProvider(api_key=api_key, **kwargs)

    def _create_anthropic_provider(
        self, api_key: str, **kwargs: Any
    ) -> AnthropicProvider:
        """
        Create Anthropic provider instance.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration

        Returns:
            AnthropicProvider instance
        """
        return AnthropicProvider(api_key=api_key, **kwargs)

    def _create_openai_provider(self, api_key: str, **kwargs: Any) -> OpenAIProvider:
        """
        Create OpenAI provider instance.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration

        Returns:
            OpenAIProvider instance
        """
        return OpenAIProvider(api_key=api_key, **kwargs)

    def _create_azure_provider(
        self, credentials: Union[AzureCredentials, Any], **kwargs: Any
    ) -> AzureOpenAIProvider:
        """
        Create Azure OpenAI provider instance.

        Args:
            credentials: Azure credentials object with api_key, azure_endpoint, and azure_deployment
            **kwargs: Additional configuration

        Returns:
            AzureOpenAIProvider instance

        Raises:
            ValueError: If credentials are not AzureCredentials
        """
        if not isinstance(credentials, AzureCredentials):
            raise ValueError(
                f"Expected AzureCredentials for Azure provider, "
                f"got {type(credentials).__name__}"
            )

        return AzureOpenAIProvider(
            api_key=credentials.api_key,
            azure_endpoint=credentials.azure_endpoint,
            azure_deployment=credentials.azure_deployment,
            api_version=credentials.api_version,
            **kwargs,
        )

    def _create_bedrock_provider(
        self, credentials: Union[AWSCredentials, Any], **kwargs: Any
    ) -> BedrockProvider:
        """
        Create Bedrock provider instance.

        Args:
            credentials: AWS credentials object with access_key, secret_key, and region
            **kwargs: Additional configuration (model, etc.)

        Returns:
            Bedrock provider instance

        Raises:
            ValueError: If credentials are not AWSCredentials
        """
        if not isinstance(credentials, AWSCredentials):
            raise ValueError(
                f"Expected AWSCredentials for Bedrock provider, "
                f"got {type(credentials).__name__}"
            )

        return BedrockProvider(
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            region=credentials.region,
            **kwargs,
        )

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """
        Get list of supported provider names.

        Returns:
            List of provider names
        """
        return [
            cls.GOOGLE,
            cls.GEMINI,
            cls.GOOGLE_GENAI,
            cls.ANTHROPIC,
            cls.OPENAI,
            cls.AZURE,
            cls.BEDROCK,
            cls.BEDROCK_CONVERSE,
        ]

    @classmethod
    def is_supported(cls, provider_name: str) -> bool:
        """
        Check if provider is supported.

        Args:
            provider_name: Provider name to check

        Returns:
            True if supported, False otherwise
        """
        return provider_name.lower() in [
            p.lower() for p in cls.get_supported_providers()
        ]
