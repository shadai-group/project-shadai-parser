"""
Constants for parser-shadai.

This module contains all magic numbers and configuration constants
extracted from the codebase to improve maintainability and readability.

Following clean code principles, all hardcoded values are defined here
with clear, descriptive names and documentation.
"""

# ============================================================================
# TEXT SAMPLING CONSTANTS
# ============================================================================

LANGUAGE_DETECTION_SAMPLE_SIZE: int = 1000
"""
Number of characters to sample for language detection (cached).

Used in: document_agent.py:121
Rationale: 1000 chars is sufficient for accurate language detection
           while minimizing LLM tokens used.
"""

LANGUAGE_SAMPLE_SIZE: int = 500
"""
Number of characters sent to LLM for language detection.

Used in: language_detector.py:26
Rationale: LLM only needs 500 chars to detect language accurately,
           reducing token usage by 50% vs full sample.
"""

DOCUMENT_TYPE_SAMPLE_SIZE: int = 5000
"""
Number of characters to sample for document type classification (cached).

Used in: document_agent.py:235
Rationale: Document type requires more context (headers, structure)
           than language detection. 5000 chars covers most doc headers.
"""

# ============================================================================
# TEXT CHUNKING CONSTANTS
# ============================================================================

CHUNK_SIZE_DEFAULT: int = 4000
"""
Default chunk size for text splitting (Phase 1 optimization).

Used in: document_agent.py:43, services.py:419
Rationale: Increased from 1000 to 4000 for 75% fewer LLM calls
           while maintaining semantic coherence.
"""

CHUNK_SIZE_MIN: int = 100
"""
Minimum chunk size for validation.

Rationale: Chunks smaller than 100 chars lack sufficient context
           for meaningful metadata extraction.
"""

CHUNK_OVERLAP_SIZE_DEFAULT: int = 400
"""
Default overlap between chunks (10% of chunk size).

Used in: document_agent.py:44, services.py:420
Rationale: Proportional to chunk_size. Ensures context continuity
           across chunk boundaries for better semantic coherence.
"""

CHUNK_MAX_CONCURRENT: int = 10
"""
Maximum number of chunks processed concurrently (Phase 2 optimization).

Used in: document_agent.py:55, services.py:438
Rationale: Balance between throughput and API rate limits.
           10 concurrent requests avoids throttling while maximizing speed.
"""

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

MIN_EXTRACTED_TEXT_LENGTH: int = 50
"""
Minimum text length to consider extraction successful.

Used in: pdf_parser.py:92
Rationale: If native PDF extraction returns <50 chars, likely a
           scanned/image-based PDF requiring OCR fallback.
"""

# ============================================================================
# RETRY CONFIGURATION
# ============================================================================

MAX_RETRIES: int = 2
"""
Maximum number of retries for failed chunk processing.

Used in: async_processor.py:117
Rationale: 2 retries balances fault tolerance with processing time.
           Most transient errors (network, rate limits) resolve in 1-2 retries.
"""

RETRY_BACKOFF_MULTIPLIER: int = 2
"""
Exponential backoff multiplier for retries.

Rationale: Each retry waits RETRY_BACKOFF_MULTIPLIER^attempt seconds.
           Retry 1: 2 seconds, Retry 2: 4 seconds.
           Prevents hammering API during transient failures.
"""

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

TEMPERATURE_DEFAULT: float = 0.2
"""
Default temperature for LLM calls (low for consistency).

Used in: document_agent.py:48, services.py:428
Rationale: Lower temperature (0.2 vs 0.7) produces more consistent,
           deterministic metadata extraction. Critical for production.
"""

# ============================================================================
# OCR CONFIGURATION
# ============================================================================

OCR_DEFAULT_DPI: int = 200
"""
Default DPI for PDF-to-image conversion in OCR.

Used in: pdf_parser.py:35, 118, 135
Rationale: 200 DPI balances quality and performance:
           - 150 DPI: Faster but lower quality (misses small text)
           - 300 DPI: Higher quality but 2x slower
           - 200 DPI: Sweet spot for medical prescriptions
"""

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_DEFAULT_TTL_SECONDS: int = 86400
"""
Default cache TTL (Time To Live) in seconds (24 hours).

Used in: document_agent.py:58, services.py:441
Rationale: 24 hours is long enough to benefit repeat uploads
           (same document in different sessions) while preventing
           stale data (language/type may change with edits).
"""

# ============================================================================
# IMAGE PROCESSING CONFIGURATION
# ============================================================================

IMAGE_CONSECUTIVE_FAILURE_THRESHOLD: int = 5
"""
Number of consecutive image processing failures before stopping.

Used in: main_agent.py:188-197
Rationale: If 5+ images fail in a row, likely a systemic issue
           (corrupted PDF, unsupported format). Stop processing
           to avoid wasting resources.
"""

# ============================================================================
# SUPPORTED LANGUAGES
# ============================================================================

SUPPORTED_LANGUAGES: list[str] = ["en", "es", "fr", "it", "pt", "ja"]
"""
List of ISO language codes supported by language detection.

Used in: language_config.py:67-74
Rationale: Limited to languages with high-quality LLM support
           and tested prompt templates.
"""

# ============================================================================
# DEFAULT LANGUAGE
# ============================================================================

DEFAULT_LANGUAGE: str = "multilingual"
"""
Default language when detection is disabled or fails.

Used in: document_agent.py:49, services.py:435
Rationale: "multilingual" is production-safe fallback that works
           for all documents regardless of language. Prevents
           quality degradation from incorrect language assumptions.
"""
