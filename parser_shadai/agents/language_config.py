"""
Language configuration for document and image processing.
"""

from enum import Enum
from typing import Dict, List


class Language(Enum):
    """Enumeration of supported languages for processing."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"


# Language names mapping
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
}

# Language prompts for LLM processing
LANGUAGE_PROMPTS: Dict[str, str] = {
    "en": "Please respond in English.",
    "es": "Por favor responde en español.",
    "fr": "Veuillez répondre en français.",
    "it": "Si prega di rispondere in italiano.",
    "pt": "Por favor, responda em português.",
    "ja": "日本語で回答してください。",
}


def get_language_prompt(language: str) -> str:
    """
    Get the language prompt for LLM processing.

    Args:
        language: Language code (e.g., 'en', 'es', 'fr')

    Returns:
        Language-specific prompt string
    """
    return LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])


def get_language_name(language: str) -> str:
    """
    Get the full language name from language code.

    Args:
        language: Language code (e.g., 'en', 'es', 'fr')

    Returns:
        Full language name
    """
    return LANGUAGE_NAMES.get(language, "English")


def get_supported_languages() -> List[str]:
    """
    Get list of all supported language codes.

    Returns:
        List of language codes
    """
    return list(LANGUAGE_NAMES.keys())
