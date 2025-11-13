"""
AWS Bedrock LLM Provider implementation.

Uses boto3 to interact with AWS Bedrock's Converse API.
"""

import base64
from io import BytesIO
from typing import Any, Dict, List, Union

import boto3
from PIL import Image

from parser_shadai.llm_providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    MessageRole,
)


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock LLM Provider using the Converse API."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        region: str,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ):
        """
        Initialize Bedrock provider.

        Args:
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region (e.g., "us-east-1")
            model: Bedrock model ID (default: Claude 3.5 Sonnet)
                   Common models:
                   - anthropic.claude-3-5-sonnet-20241022-v2:0
                   - anthropic.claude-3-haiku-20240307-v1:0
                   - anthropic.claude-3-opus-20240229-v1:0
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

        # Initialize boto3 Bedrock client
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        # Initialize with model (api_key is not used for Bedrock, but required by base class)
        super().__init__(api_key=access_key, model=model)

    def _validate_api_key(self) -> None:
        """Validate AWS credentials by making a test call."""
        try:
            # Simple test to validate credentials
            self.client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": "test"}]}],
                inferenceConfig={"maxTokens": 10, "temperature": 0.5},
            )
        except Exception as e:
            raise ValueError(f"Invalid AWS Bedrock credentials: {str(e)}")

    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text response from a text prompt.

        Args:
            prompt: Text prompt to send to Bedrock
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse object with the generated text
        """
        try:
            response = self.client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                    "topP": kwargs.get("top_p", 1.0),
                },
            )

            # Extract response content
            output_message = response["output"]["message"]
            content_blocks = output_message.get("content", [])
            text_content = " ".join(
                [block.get("text", "") for block in content_blocks if "text" in block]
            )

            # Extract usage stats
            usage = response.get("usage", {})

            return LLMResponse(
                content=text_content,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": usage.get("inputTokens", 0),
                    "completion_tokens": usage.get("outputTokens", 0),
                },
                finish_reason=response.get("stopReason", "stop"),
            )
        except Exception as e:
            raise RuntimeError(f"Error generating text with Bedrock: {str(e)}")

    def generate_with_images(
        self, prompt: str, images: List[Union[str, Image.Image]], **kwargs
    ) -> LLMResponse:
        """
        Generate response from text prompt and images.

        Args:
            prompt: Text prompt to send to Bedrock
            images: List of images (file paths, PIL Images, or base64 strings)
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text
        """
        try:
            # Prepare content blocks with images
            content_blocks = []

            # Add images first
            for img in images:
                img_data = self._prepare_image_for_bedrock(img)
                content_blocks.append(
                    {
                        "image": {
                            "format": img_data["format"],
                            "source": {"bytes": img_data["bytes"]},
                        }
                    }
                )

            # Add text prompt
            content_blocks.append({"text": prompt})

            response = self.client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": content_blocks}],
                inferenceConfig={
                    "maxTokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                    "topP": kwargs.get("top_p", 1.0),
                },
            )

            # Extract response content
            output_message = response["output"]["message"]
            content_blocks_response = output_message.get("content", [])
            text_content = " ".join(
                [
                    block.get("text", "")
                    for block in content_blocks_response
                    if "text" in block
                ]
            )

            # Extract usage stats
            usage = response.get("usage", {})

            return LLMResponse(
                content=text_content,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": usage.get("inputTokens", 0),
                    "completion_tokens": usage.get("outputTokens", 0),
                },
                finish_reason=response.get("stopReason", "stop"),
            )
        except Exception as e:
            raise RuntimeError(f"Error generating with images (Bedrock): {str(e)}")

    def generate_with_documents(
        self, prompt: str, documents: List[str], **kwargs
    ) -> LLMResponse:
        """
        Generate response from text prompt and documents.

        For Bedrock, we concatenate documents with the prompt.

        Args:
            prompt: Text prompt to send to Bedrock
            documents: List of document texts
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text
        """
        # Concatenate documents with prompt
        combined_prompt = "\n\n".join(documents) + "\n\n" + prompt
        return self.generate_text(combined_prompt, **kwargs)

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
            # Convert messages to Bedrock format
            bedrock_messages = []
            for msg in messages:
                role = "user" if msg.role == MessageRole.USER else "assistant"

                # Build content blocks
                content_blocks = []

                # Add images if present
                if msg.images:
                    for img_b64 in msg.images:
                        # Decode base64 to bytes
                        img_bytes = base64.b64decode(img_b64)
                        content_blocks.append(
                            {
                                "image": {
                                    "format": "png",  # Assume PNG for base64 images
                                    "source": {"bytes": img_bytes},
                                }
                            }
                        )

                # Add text content
                content_blocks.append({"text": msg.content})

                bedrock_messages.append({"role": role, "content": content_blocks})

            response = self.client.converse(
                modelId=self.model,
                messages=bedrock_messages,
                inferenceConfig={
                    "maxTokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                    "topP": kwargs.get("top_p", 1.0),
                },
            )

            # Extract response content
            output_message = response["output"]["message"]
            content_blocks_response = output_message.get("content", [])
            text_content = " ".join(
                [
                    block.get("text", "")
                    for block in content_blocks_response
                    if "text" in block
                ]
            )

            # Extract usage stats
            usage = response.get("usage", {})

            return LLMResponse(
                content=text_content,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": usage.get("inputTokens", 0),
                    "completion_tokens": usage.get("outputTokens", 0),
                },
                finish_reason=response.get("stopReason", "stop"),
            )
        except Exception as e:
            raise RuntimeError(f"Error in chat (Bedrock): {str(e)}")

    def _prepare_image_for_bedrock(
        self, image: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """
        Prepare image for Bedrock API format.

        Args:
            image: Image as file path, PIL Image, or base64 string

        Returns:
            Dictionary with 'format' and 'bytes' keys
        """
        if isinstance(image, str):
            # Check if it's base64
            try:
                img_bytes = base64.b64decode(image)
                # Assume PNG for base64 (could be improved with format detection)
                return {"format": "png", "bytes": img_bytes}
            except Exception:
                # It's a file path
                with open(image, "rb") as f:
                    img_bytes = f.read()

                # Detect format from extension
                img_format = "png"
                if image.lower().endswith((".jpg", ".jpeg")):
                    img_format = "jpeg"
                elif image.lower().endswith(".gif"):
                    img_format = "gif"
                elif image.lower().endswith(".webp"):
                    img_format = "webp"

                return {"format": img_format, "bytes": img_bytes}

        elif isinstance(image, Image.Image):
            # Convert PIL Image to bytes
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return {"format": "png", "bytes": buffer.getvalue()}

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
