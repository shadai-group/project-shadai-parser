"""
Azure OpenAI LLM Provider implementation.
"""

import base64
from io import BytesIO
from typing import List, Union

import openai
from PIL import Image

from parser_shadai.llm_providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    MessageRole,
)


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI LLM Provider."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        model: str = "gpt-4o-mini",
        api_version: str = "2024-02-15-preview",
    ):
        """
        Initialize Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure deployment name
            model: Model name for display (default: gpt-4o-mini)
            api_version: Azure API version (default: 2024-02-15-preview)
        """
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
        )
        super().__init__(api_key=api_key, model=model)

    def _validate_api_key(self) -> None:
        """Validate the API key by making a test call."""
        try:
            # Simple test to validate API key
            self.client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
        except Exception as e:
            raise ValueError(f"Invalid Azure OpenAI API key or configuration: {str(e)}")

    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text response from a text prompt.

        Args:
            prompt: Text prompt to send to Azure OpenAI
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse object with the generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            raise RuntimeError(f"Error generating text with Azure OpenAI: {str(e)}")

    def generate_with_images(
        self, prompt: str, images: List[Union[str, Image.Image]], **kwargs
    ) -> LLMResponse:
        """
        Generate response from text prompt and images.

        Args:
            prompt: Text prompt to send to Azure OpenAI
            images: List of images (file paths, PIL Images, or base64 strings)
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text
        """
        try:
            # Prepare images for Azure OpenAI (same format as OpenAI)
            prepared_images = []
            for img in images:
                if isinstance(img, str):
                    # Check if it's base64 or file path
                    try:
                        base64.b64decode(img)
                        # It's base64
                        prepared_images.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img}"},
                            }
                        )
                    except:
                        # It's a file path
                        with open(img, "rb") as f:
                            img_data = f.read()
                            img_b64 = base64.b64encode(img_data).decode("utf-8")
                            # Try to detect media type
                            media_type = "image/png"
                            if img.lower().endswith((".jpg", ".jpeg")):
                                media_type = "image/jpeg"
                            elif img.lower().endswith(".gif"):
                                media_type = "image/gif"
                            elif img.lower().endswith(".webp"):
                                media_type = "image/webp"

                            prepared_images.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{img_b64}"
                                    },
                                }
                            )
                elif isinstance(img, Image.Image):
                    # Convert PIL Image to base64
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    prepared_images.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        }
                    )
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

            # Create content list with text and images
            content = [{"type": "text", "text": prompt}]
            content.extend(prepared_images)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "images_processed": len(prepared_images),
                },
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error generating with images using Azure OpenAI: {str(e)}"
            )

    def generate_with_documents(
        self, prompt: str, documents: List[str], **kwargs
    ) -> LLMResponse:
        """
        Generate response from text prompt and documents.

        Args:
            prompt: Text prompt to send to Azure OpenAI
            documents: List of document file paths or content
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text
        """
        try:
            # For documents, we'll read them as text and include in the prompt
            document_content = []
            for doc in documents:
                try:
                    with open(doc, "r", encoding="utf-8") as f:
                        content = f.read()
                        document_content.append(f"Document: {doc}\n{content}")
                except:
                    # Assume it's already content
                    document_content.append(f"Document content:\n{doc}")

            full_prompt = f"{prompt}\n\nDocuments:\n" + "\n\n".join(document_content)

            return self.generate_text(full_prompt, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Error generating with documents using Azure OpenAI: {str(e)}"
            )

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate response from a conversation.

        Args:
            messages: List of Message objects representing the conversation
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text
        """
        try:
            # Convert messages to Azure OpenAI format (same as OpenAI)
            openai_messages = []

            for msg in messages:
                if msg.role == MessageRole.USER:
                    content = [{"type": "text", "text": msg.content}]

                    # Add images if present
                    if msg.images:
                        for img_b64 in msg.images:
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                }
                            )

                    openai_messages.append({"role": "user", "content": content})
                elif msg.role == MessageRole.ASSISTANT:
                    openai_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
                elif msg.role == MessageRole.SYSTEM:
                    openai_messages.append({"role": "system", "content": msg.content})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "messages": len(messages),
                },
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            raise RuntimeError(f"Error in chat with Azure OpenAI: {str(e)}")
