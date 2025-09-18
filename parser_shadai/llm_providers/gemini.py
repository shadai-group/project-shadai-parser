"""
Google Gemini LLM Provider implementation.
"""

import google.generativeai as genai
from typing import List, Union
from PIL import Image
import base64
from io import BytesIO

from parser_shadai.llm_providers.base import BaseLLMProvider, LLMResponse, Message, MessageRole


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google AI API key
            model: Gemini model name (default: gemini-1.5-flash)
        """
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    def _validate_api_key(self) -> None:
        """Validate the API key by making a test call."""
        try:
            # Simple test to validate API key
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            model.generate_content("test")
        except Exception as e:
            raise ValueError(f"Invalid Gemini API key: {str(e)}")
    
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
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_output_tokens=kwargs.get('max_tokens', 8192),
                top_p=kwargs.get('top_p', 0.8),
                top_k=kwargs.get('top_k', 40)
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                content=response.text,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.text.split()) if response.text else 0
                }
            )
        except Exception as e:
            raise RuntimeError(f"Error generating text with Gemini: {str(e)}")
    
    def generate_with_images(self, prompt: str, images: List[Union[str, Image.Image]], **kwargs) -> LLMResponse:
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
            # Prepare images for Gemini
            prepared_images = []
            for img in images:
                if isinstance(img, str):
                    # Check if it's base64 or file path
                    try:
                        base64.b64decode(img)
                        # It's base64, convert to PIL Image
                        img_data = base64.b64decode(img)
                        pil_img = Image.open(BytesIO(img_data))
                        prepared_images.append(pil_img)
                    except:
                        # It's a file path
                        prepared_images.append(Image.open(img))
                elif isinstance(img, Image.Image):
                    prepared_images.append(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Create content list with text and images
            content = [prompt]
            for img in prepared_images:
                content.append(img)
            
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_output_tokens=kwargs.get('max_tokens', 8192),
                top_p=kwargs.get('top_p', 0.8),
                top_k=kwargs.get('top_k', 40)
            )
            
            response = self.client.generate_content(
                content,
                generation_config=generation_config
            )
            
            return LLMResponse(
                content=response.text,
                model=self._get_model_name(),
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.text.split()) if response.text else 0,
                    "images_processed": len(prepared_images)
                }
            )
        except Exception as e:
            raise RuntimeError(f"Error generating with images using Gemini: {str(e)}")
    
    def generate_with_documents(self, prompt: str, documents: List[str], **kwargs) -> LLMResponse:
        """
        Generate response from text prompt and documents.
        
        Args:
            prompt: Text prompt to send to Gemini
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
                    with open(doc, 'r', encoding='utf-8') as f:
                        content = f.read()
                        document_content.append(f"Document: {doc}\n{content}")
                except:
                    # Assume it's already content
                    document_content.append(f"Document content:\n{doc}")
            
            full_prompt = f"{prompt}\n\nDocuments:\n" + "\n\n".join(document_content)
            
            return self.generate_text(full_prompt, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error generating with documents using Gemini: {str(e)}")
    
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
            # Convert messages to Gemini format
            chat_history = []
            for msg in messages:
                if msg.role == MessageRole.USER:
                    if msg.images:
                        # For user messages with images, create content list
                        content = [msg.content]
                        for img_b64 in msg.images:
                            img_data = base64.b64decode(img_b64)
                            pil_img = Image.open(BytesIO(img_data))
                            content.append(pil_img)
                        chat_history.append({"role": "user", "parts": content})
                    else:
                        chat_history.append({"role": "user", "parts": [msg.content]})
                elif msg.role == MessageRole.ASSISTANT:
                    chat_history.append({"role": "model", "parts": [msg.content]})
                elif msg.role == MessageRole.SYSTEM:
                    # Gemini doesn't have system messages, prepend to first user message
                    if chat_history and chat_history[0]["role"] == "user":
                        chat_history[0]["parts"][0] = f"System: {msg.content}\n\n{chat_history[0]['parts'][0]}"
                    else:
                        # If no user message yet, create one
                        chat_history.append({"role": "user", "parts": [f"System: {msg.content}"]})
            
            # Start a chat session
            chat = self.client.start_chat(history=chat_history[:-1] if len(chat_history) > 1 else [])
            
            # Send the last message
            last_message = chat_history[-1]
            if last_message["role"] == "user":
                if len(last_message["parts"]) > 1:
                    # Has images
                    response = chat.send_message(last_message["parts"])
                else:
                    # Text only
                    response = chat.send_message(last_message["parts"][0])
            else:
                raise ValueError("Last message must be from user")
            
            return LLMResponse(
                content=response.text,
                model=self._get_model_name(),
                usage={
                    "messages": len(messages),
                    "completion_tokens": len(response.text.split()) if response.text else 0
                }
            )
        except Exception as e:
            raise RuntimeError(f"Error in chat with Gemini: {str(e)}")
