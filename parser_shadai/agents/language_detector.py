"""
Language detection utilities for document processing.
"""

import re

from parser_shadai.agents.language_config import Language, get_supported_languages


class LanguageDetector:
    """Utility class for detecting document language."""

    @classmethod
    def detect_language_with_llm(cls, text: str, llm_provider):
        """
        Detect language using LLM for more accurate detection.

        Args:
            text: Text content to analyze
            llm_provider: LLM provider instance (optional)
            llm_provider: LLM provider instance

        Returns:
            Tuple of (Language code, usage dict)
        """
        if not text or len(text.strip()) < 10:
            return "en", None

        # Take first 500 characters for language detection
        sample_text = text[:500]

        prompt = f"""Analyze the following text and determine its language. Return only the language code.

Supported languages: {", ".join(get_supported_languages())}

Text sample:
{sample_text}

Language code:"""

        try:
            response = llm_provider.generate_text(
                prompt, temperature=0.1, max_tokens=10
            )
            detected_lang = response.content.strip().lower()
            # Validate the detected language
            if detected_lang in get_supported_languages():
                return detected_lang, response.usage
            else:
                # Fallback to pattern-based detection
                return Language.ENGLISH.value, response.usage

        except Exception as e:
            print(f"Warning: LLM language detection failed: {e}")
            # Fallback to pattern-based detection
            return Language.ENGLISH.value, None

    @classmethod
    def get_language_confidence(cls, text: str, language: str) -> float:
        """
        Get confidence score for a specific language in the text.

        Args:
            text: Text content to analyze
            llm_provider: LLM provider instance (optional)
            language: Language code to check

        Returns:
            Confidence score between 0 and 1
        """
        if language not in cls.LANGUAGE_PATTERNS:
            return 0.0

        # Fallback to pattern-based detection
        text_lower = text.lower()
        patterns = cls.LANGUAGE_PATTERNS[language]
        total_matches = 0

        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            total_matches += matches

        # Normalize by text length
        confidence = total_matches / len(text.split()) if text.split() else 0.0
        return min(confidence, 1.0)  # Cap at 1.0
