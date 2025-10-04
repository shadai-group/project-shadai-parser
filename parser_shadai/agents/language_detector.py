"""
Language detection utilities for document processing.
"""

import re

from parser_shadai.agents.language_config import get_supported_languages


class LanguageDetector:
    """Utility class for detecting document language."""

    @classmethod
    def detect_language_with_llm(cls, text: str, llm_provider):
        """
        Detect language using LLM for more accurate detection.

        Args:
            text: Text content to analyze
            llm_provider: LLM provider instance

        Returns:
            Tuple of (Language code, usage dict)
        """
        # Take first 500 characters for language detection
        sample_text = text[:500]

        prompt = f"""
        You are a professional language detection system. Your task is to identify the primary language of the given text sample with high accuracy.

        INSTRUCTIONS:
        1. Analyze the text carefully for linguistic patterns, vocabulary, and structure
        2. Return ONLY the ISO language code (e.g., 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ar')
        3. Be precise - choose the single most dominant language
        4. If multiple languages are present, identify the PRIMARY language (most prevalent)
        5. Consider context clues like proper nouns, technical terms, and sentence structure

        SUPPORTED LANGUAGES: {", ".join(get_supported_languages())}

        TEXT TO ANALYZE:
        {sample_text}

        LANGUAGE CODE:
        """

        try:
            response = llm_provider.generate_text(
                prompt, temperature=0.1, max_tokens=10
            )
            detected_lang = response.content.strip().lower()
            # Validate the detected language
            if detected_lang in get_supported_languages():
                return detected_lang, response.usage
            else:
                # If unsupported language detected, raise an error instead of fallback
                raise ValueError(
                    f"Detected language '{detected_lang}' is not supported. Supported languages: {get_supported_languages()}"
                )

        except Exception as e:
            print(f"Error: Language detection failed: {e}")
            # Instead of falling back to English, raise the error
            raise RuntimeError(f"Language detection is required but failed: {e}")

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
