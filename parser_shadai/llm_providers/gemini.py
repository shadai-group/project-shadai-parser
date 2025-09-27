"""
Google Gemini LLM Provider implementation.
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

from google import genai
from google.genai import types
from PIL import Image

from parser_shadai.llm_providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    MessageRole,
)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI API key
            model: Gemini model name (default: gemini-2.0-flash)
        """
        self.client = genai.Client(api_key=api_key)
        super().__init__(api_key=api_key, model=model)
        self._image_mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }

    def _validate_api_key(self) -> str:
        """Validate the API key by making a test call.

        Returns:
            Response text

        Raises:
            ValueError: If the API key is invalid
        """
        try:
            response = self.client.models.generate_content(
                model=self.model, contents="Why is the sky blue?"
            )
            return response.text
        except Exception as e:
            raise ValueError(f"Invalid Gemini API key: {str(e)}")

    def _prepare_images(
        self, images: List[Union[str, Image.Image]]
    ) -> List[types.Part]:
        """Prepare images for Gemini API format.

        Args:
            images: List of images (file paths, PIL Images, or base64 strings)

        Returns:
            List of images in Gemini API format

        Raises:
            ValueError: If the image type is unsupported
        """
        return [self._process_single_image(img=img) for img in images]

    def _process_single_image(self, img: Union[str, Image.Image]) -> types.Part:
        """Process a single image to Gemini API format.

        Args:
            img: Image

        Returns:
            Image in Gemini API format
        """
        if isinstance(img, Image.Image):
            return self._convert_pil_image(pil_img=img)
        elif isinstance(img, str):
            return self._process_string_image(img_str=img)
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _process_string_image(self, img_str: str) -> types.Part:
        """Process string image (base64 or file path).

        Args:
            img_str: String image

        Returns:
            Image in Gemini API format
        """
        if self._is_base64_image(img_str):
            return self._convert_base64_image(img_str=img_str)
        return self._convert_file_path_image(img_str=img_str)

    def _is_base64_image(self, img_str: str) -> bool:
        """Check if string is base64 encoded image.

        Args:
            img_str: String image

        Returns:
            True if the string is base64 encoded image, False otherwise
        """
        try:
            base64.b64decode(img_str, validate=True)
            return True
        except Exception as e:
            print(f"Error checking if string is base64 encoded image: {str(e)}")
            return False

    def _convert_base64_image(self, img_str: str) -> types.Part:
        """Convert base64 string to Gemini format.

        Args:
            img_str: Base64 string

        Returns:
            Image in Gemini API format
        """
        img_data = base64.b64decode(img_str)
        pil_img = Image.open(BytesIO(img_data))
        return self._convert_pil_image(pil_img=pil_img)

    def _convert_file_path_image(self, file_path: str) -> types.Part:
        """Convert file path image to Gemini format.

        Args:
            file_path: File path to the image

        Returns:
            Image in Gemini API format
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        extension = path.suffix.lower().lstrip(".")
        mime_type = self._image_mime_types.get(extension, "image/png")

        with open(path, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=mime_type)

    def _convert_pil_image(self, pil_img: Image.Image) -> types.Part:
        """Convert PIL Image to Gemini format.

        Args:
            pil_img: PIL Image

        Returns:
            Image in Gemini API format
        """
        buffer = BytesIO()
        format_type = getattr(pil_img, "format", "PNG") or "PNG"
        pil_img.save(buffer, format=format_type)
        buffer.seek(0)

        mime_type = self._image_mime_types.get(format_type.lower(), "image/png")
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type=mime_type)

    def _convert_messages_to_contents(
        self, messages: List[Message]
    ) -> Tuple[List[dict], Optional[str]]:
        """Convert Message objects to Gemini API format.

        Args:
            messages: List of Message objects

        Returns:
            List of messages in Gemini API format, Optional[str] system message
        """
        contents = []
        system_message = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
                continue

            content_dict = self._create_message_content(
                msg=msg,
            )
            contents.append(content_dict)
            system_message = None

        return contents, system_message

    def _create_message_content(self, msg: Message) -> dict:
        """Create content dictionary for a single message.

        Args:
            msg: Message object

        Returns:
            Content dictionary in Gemini API format
        """
        if msg.role == MessageRole.USER:
            text_content = msg.content

            parts = [types.Part.from_text(text=text_content)]
            if msg.images:
                prepared_images = self._prepare_images(images=msg.images)
                parts.extend(prepared_images)

            return {"role": "user", "parts": parts}

        elif msg.role == MessageRole.ASSISTANT:
            return {"role": "model", "parts": [types.Part.from_text(text=msg.content)]}

        else:
            raise ValueError(f"Unsupported message role: {msg.role}")

    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text response from a text prompt.

        Args:
            prompt: Text prompt to send to Gemini
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse object with the generated text
        """
        try:
            response: types.GenerateContentResponse = (
                self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=kwargs.get("max_tokens", 8192),
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.8),
                        top_k=kwargs.get("top_k", 40),
                    ),
                )
            )

            return LLMResponse(
                content=response.text,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                },
            )
        except Exception as e:
            raise RuntimeError(f"Error generating text with Gemini: {str(e)}")

    def generate_with_images(
        self, prompt: str, images: List[Union[str, Image.Image]], **kwargs
    ) -> LLMResponse:
        """
        Generate response from text prompt and images.

        Args:
            prompt: Text prompt to send to Gemini
            images: List of images (file paths, PIL Images, or base64 strings)
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text
        """
        try:
            prepared_images = self._prepare_images(images=images)

            content = [types.Part.from_text(text=prompt)]
            content.extend(prepared_images)

            response: types.GenerateContentResponse = (
                self.client.models.generate_content(
                    model=self.model,
                    contents=content,
                    config=types.GenerateContentConfig(
                        max_output_tokens=kwargs.get("max_tokens", 8192),
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.8),
                        top_k=kwargs.get("top_k", 40),
                    ),
                )
            )

            return LLMResponse(
                content=response.text,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "images_processed": len(prepared_images),
                },
            )
        except Exception as e:
            raise RuntimeError(f"Error generating with images using Gemini: {str(e)}")

    def generate_with_documents(
        self, prompt: str, documents: List[str], **kwargs
    ) -> LLMResponse:
        """
        Generate response from text prompt and documents.

        Args:
            prompt: Text prompt to send to Gemini
            documents: List of document file paths or content strings
            **kwargs: Additional parameters

        Returns:
            LLMResponse object with the generated text

        Raises:
            RuntimeError: If document processing or generation fails
            FileNotFoundError: If a document file doesn't exist
        """
        try:
            document_contents = self._process_documents(documents=documents)
            full_prompt = self._build_prompt_with_documents(
                prompt=prompt, document_contents=document_contents
            )
            return self.generate_text(prompt=full_prompt, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Error generating with documents using Gemini: {str(e)}"
            )

    def _process_documents(self, documents: List[str]) -> List[str]:
        """Process document inputs (file paths or content strings).

        Args:
            documents: List of document file paths or content strings

        Returns:
            List of processed document content strings
        """
        processed_docs = []
        for i, doc in enumerate(documents):
            try:
                processed_content = self._process_single_document(doc=doc, index=i)
                processed_docs.append(processed_content)
            except Exception as e:
                print(f"Warning: Failed to process document {i}: {e}")
                processed_docs.append(f"Document {i + 1}: [Error loading document]")
        return processed_docs

    def _process_single_document(self, doc: str, index: int) -> str:
        """Process a single document (file path or content string).

        Args:
            doc: Document file path or content string
            index: Document index for labeling

        Returns:
            Formatted document content string
        """
        if self._is_file_path(doc=doc):
            return self._read_document_file(file_path=doc, index=index)
        return f"Document {index + 1}:\n{doc}"

    def _is_file_path(self, doc: str) -> bool:
        """Determine if string is a file path or content using built-in methods.

        Args:
            doc: String to check

        Returns:
            True if it appears to be a file path, False otherwise
        """
        try:
            path = Path(doc)

            if not path.parts:
                return False

            if path.exists():
                return True

            if path.parent.exists():
                return True

            return path.suffix != "" or os.sep in doc or "/" in doc or "\\" in doc

        except (ValueError, OSError):
            return False

    def _read_document_file(self, file_path: str, index: int) -> str:
        """Read content from a document file.

        Args:
            file_path: Path to the document file
            index: Document index for labeling

        Returns:
            Formatted document content

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                return f"Document {index + 1} ({path.name}):\n{content}"
        except UnicodeDecodeError:
            for encoding in ["latin1", "cp1252"]:
                try:
                    with open(path, "r", encoding=encoding) as f:
                        content = f.read()
                        return f"Document {index + 1} ({path.name}):\n{content}"
                except UnicodeDecodeError:
                    continue
            raise IOError(
                f"Could not decode file {file_path} with any supported encoding"
            )

    def _build_prompt_with_documents(
        self, prompt: str, document_contents: List[str]
    ) -> str:
        """Build the final prompt with document contents.

        Args:
            prompt: Original user prompt
            document_contents: List of processed document content strings

        Returns:
            Complete prompt string with documents
        """
        if not document_contents:
            return prompt

        documents_section = "\n\n".join(document_contents)
        return f"{prompt}\n\nDocuments:\n{documents_section}"

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
            contents, system_message = self._convert_messages_to_contents(
                messages=messages
            )

            response: types.GenerateContentResponse = (
                self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_message,
                        max_output_tokens=kwargs.get("max_tokens", 8192),
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.8),
                        top_k=kwargs.get("top_k", 40),
                    ),
                )
            )

            return LLMResponse(
                content=response.text,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "messages": len(messages),
                },
            )
        except Exception as e:
            raise RuntimeError(f"Error in chat with Gemini: {str(e)}")
