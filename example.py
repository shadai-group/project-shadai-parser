"""
Example script for testing the ShadAI Parser with optimized configuration.

This script demonstrates how to use the parser with optimized settings
to process PDF files from the data folder.

Usage:
    # Set API key first
    export GEMINI_API_KEY="your-key-here"
    # OR
    export ANTHROPIC_API_KEY="your-key-here"
    # OR
    export OPENAI_API_KEY="your-key-here"

    # Run the example
    python example.py

Requirements:
    - python-dotenv (optional, for loading from .env file)
    - One of: google-generativeai, anthropic, or openai
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from parser_shadai import (
    AgentConfig,
    AnthropicProvider,
    GeminiProvider,
    MainProcessingAgent,
    OpenAIProvider,
)

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly


# Configuration
DATA_FOLDER = Path(__file__).parent / "data"
OUTPUT_FOLDER = Path(__file__).parent / "output"


def get_llm_provider():
    """
    Get LLM provider from environment variables.

    Priority order:
    1. GEMINI_API_KEY / GOOGLE_API_KEY
    2. ANTHROPIC_API_KEY
    3. OPENAI_API_KEY

    Returns:
        LLM provider instance

    Raises:
        ValueError: If no API key is found
    """
    # Check Gemini/Google
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        print("✓ Using Gemini provider")
        return GeminiProvider(api_key=gemini_key)

    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("✓ Using Anthropic provider")
        return AnthropicProvider(api_key=anthropic_key)

    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✓ Using OpenAI provider")
        return OpenAIProvider(api_key=openai_key)

    # No API key found
    raise ValueError(
        "No API key found. Set one of:\n"
        "  - GEMINI_API_KEY or GOOGLE_API_KEY\n"
        "  - ANTHROPIC_API_KEY\n"
        "  - OPENAI_API_KEY"
    )


def create_optimized_config() -> AgentConfig:
    """
    Create optimized configuration for fast parsing.

    Optimizations applied (Phase 1 + Phase 2 + Phase 3):
    ✅ Larger chunk size: 4000 (vs 1000) = 75% fewer chunks
    ✅ Auto language detection: detect document language (with caching)
    ✅ Image processing disabled: text-only parsing
    ✅ Low temperature: 0.2 for consistent results
    ✅ Document-level metadata: extract once instead of per-chunk
    ✅ Async batch processing: concurrent chunk processing
    ✅ Smart caching: cache expensive LLM operations

    Returns:
        Optimized AgentConfig
    """
    return AgentConfig(
        # Chunking config (OPTIMIZED - Phase 1)
        chunk_size=4000,  # Increased from 1000 (75% fewer LLM calls!)
        overlap_size=400,  # Proportional overlap
        use_smart_chunking=True,  # Smart chunking for better context
        # Image processing (DISABLED for speed)
        extract_images=False,
        extract_text_from_images=False,
        describe_images=False,
        classify_images=False,
        # Language config (Auto-detection with caching)
        language="en",  # Default if auto-detection disabled
        auto_detect_language=True,  # ENABLED: Auto-detect language (with caching)
        # Document type detection
        auto_detect_document_type=True,  # Keep for metadata quality
        # LLM config
        temperature=0.2,  # Low for consistency
        # Processing config
        parallel_processing=False,
        folder_processing=False,
        # Phase 2 optimization: document-level metadata + async batch processing
        use_optimized_extraction=True,  # Enable Phase 2 optimizations
        max_concurrent_chunks=10,  # Max concurrent chunk processing
        # Phase 3 optimization: smart caching
        enable_caching=True,  # Enable caching for expensive operations
        cache_ttl_seconds=86400,  # Cache TTL: 24 hours
    )


def print_separator(title: str = "", width: int = 80):
    """Print formatted separator."""
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}\n")
    else:
        print(f"{'=' * width}\n")


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def print_results_summary(result: Dict[str, Any], processing_time: float):
    """Print formatted summary of processing results."""
    doc_info = result.get("document_info", {})
    chunk_stats = result.get("chunk_statistics", {})
    usage = result.get("usage", {})

    print_separator("📊 Processing Results")

    # Document info
    print(f"📄 File: {doc_info.get('file_name', 'Unknown')}")
    print(f"📑 Document Type: {doc_info.get('document_type', 'general').upper()}")
    print(f"📖 Pages: {doc_info.get('page_count', 0)}")
    print(f"🔢 Total Chunks: {result.get('total_chunks', 0)}")
    print(f"⏱️  Processing Time: {processing_time:.2f}s")

    # Chunk statistics
    if chunk_stats:
        print("\n📈 Chunk Statistics:")
        print(
            f"   • Average size: {chunk_stats.get('avg_chunk_size', 0):.0f} characters"
        )
        print(f"   • Total characters: {chunk_stats.get('total_chars', 0):,}")
        print(f"   • Total words: {chunk_stats.get('total_words', 0):,}")
        print(f"   • Min chunk: {chunk_stats.get('min_chunk_size', 0):.0f} characters")
        print(f"   • Max chunk: {chunk_stats.get('max_chunk_size', 0):.0f} characters")

    # Token usage
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        print("\n💰 Token Usage:")
        print(f"   • Prompt tokens: {prompt_tokens:,}")
        print(f"   • Completion tokens: {completion_tokens:,}")
        print(f"   • Total tokens: {total_tokens:,}")

        # Rough cost estimate (adjust based on provider)
        cost_per_1m_tokens = 0.50  # Average estimate
        estimated_cost = total_tokens / 1_000_000 * cost_per_1m_tokens
        print(f"   • Estimated cost: ${estimated_cost:.4f}")

    # Document summary
    summary = result.get("document_summary", "")
    if summary:
        print("\n📝 Document Summary:")
        summary_preview = summary[:300] + "..." if len(summary) > 300 else summary
        print(f"   {summary_preview}")

    # Sample chunk
    chunks = result.get("chunks", [])
    if chunks:
        print("\n📄 Sample Chunk (first chunk):")
        first_chunk = chunks[0]
        content_preview = (
            first_chunk.get("content", "")[:200] + "..."
            if len(first_chunk.get("content", "")) > 200
            else first_chunk.get("content", "")
        )
        print(f"   Content: {content_preview}")

        metadata = first_chunk.get("metadata", {})
        if metadata.get("summary"):
            print(f"   Summary: {metadata.get('summary', '')[:200]}...")

    print_separator()


def save_results_to_file(result: Dict[str, Any], filename: str):
    """
    Save processing results to JSON file.

    Args:
        result: Processing results dictionary
        filename: Output filename (without extension)
    """
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    output_path = OUTPUT_FOLDER / f"{filename}_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Full results saved to: {output_path}")
    print(f"  File size: {format_bytes(output_path.stat().st_size)}")


def process_single_file(file_path: Path, agent: MainProcessingAgent):
    """
    Process a single PDF file.

    Args:
        file_path: Path to PDF file
        agent: MainProcessingAgent instance

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If processing fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print_separator(f"📂 Processing: {file_path.name}")
    print(f"📦 File size: {format_bytes(file_path.stat().st_size)}")
    print(f"📁 Full path: {file_path}")

    start_time = time.time()

    try:
        print("\n⏳ Parsing document...")
        result = agent.process_file(file_path=str(file_path), auto_detect_type=True)

        processing_time = time.time() - start_time

        print(f"✅ Processing completed in {processing_time:.2f}s\n")

        # Print summary
        print_results_summary(result=result, processing_time=processing_time)

        # Save results
        save_results_to_file(result=result, filename=file_path.stem)

        return result

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"\n❌ Error after {processing_time:.2f}s: {e}")
        raise


def process_all_files(agent: MainProcessingAgent):
    """
    Process all PDF files in the data folder.

    Args:
        agent: MainProcessingAgent instance
    """
    if not DATA_FOLDER.exists():
        print(f"❌ Data folder not found: {DATA_FOLDER}")
        return

    pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {DATA_FOLDER}")
        print(f"\nPlease add PDF files to: {DATA_FOLDER.absolute()}")
        return

    print(f"\n📚 Found {len(pdf_files)} PDF file(s) to process:")
    total_size = sum(f.stat().st_size for f in pdf_files)
    for pdf_file in pdf_files:
        print(f"   • {pdf_file.name} ({format_bytes(pdf_file.stat().st_size)})")
    print(f"\n📦 Total size: {format_bytes(total_size)}")

    total_start_time = time.time()
    results = []
    errors = []

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n\n{'=' * 80}")
        print(f"[{i}/{len(pdf_files)}] Processing file {i} of {len(pdf_files)}")
        print(f"{'=' * 80}")

        try:
            result = process_single_file(file_path=pdf_file, agent=agent)
            results.append(
                {"file": pdf_file.name, "status": "success", "result": result}
            )
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
            errors.append({"file": pdf_file.name, "error": str(e)})
            continue

    total_time = time.time() - total_start_time

    # Print final summary
    print_separator("🎉 Batch Processing Complete")
    print(f"✅ Processed: {len(results)} file(s)")
    if errors:
        print(f"❌ Failed: {len(errors)} file(s)")
        for error in errors:
            print(f"   • {error['file']}: {error['error']}")

    print(f"\n⏱️  Total time: {total_time:.2f}s")
    if results:
        print(f"⏱️  Average time per file: {total_time / len(results):.2f}s")

    # Calculate total tokens
    if results:
        total_tokens = sum(
            r["result"].get("usage", {}).get("prompt_tokens", 0)
            + r["result"].get("usage", {}).get("completion_tokens", 0)
            for r in results
        )
        print(f"💰 Total tokens used: {total_tokens:,}")

    print_separator()


def main():
    """Main entry point."""
    print_separator("🚀 ShadAI Parser - Optimized Example")

    print("📌 Parser Optimizations (Phase 1 + Phase 2 + Phase 3):")
    print("   ✅ Larger chunk size (4000 vs 1000) = 75% fewer chunks")
    print("   ✅ Auto language detection = metadata in document's language (cached)")
    print("   ✅ Image extraction disabled = text-only parsing")
    print("   ✅ Low temperature (0.2) = consistent results")
    print("   ✅ Document-level metadata = extract once instead of per-chunk")
    print("   ✅ Async batch processing = concurrent chunk processing")
    print("   ✅ Smart caching = cache expensive LLM operations (24h TTL)")

    # Get LLM provider
    try:
        llm_provider = get_llm_provider()
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nExample setup:")
        print("   export GEMINI_API_KEY='your-key-here'")
        print("   # OR")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("   # OR")
        print("   export OPENAI_API_KEY='your-key-here'")
        return 1

    # Create optimized config
    config = create_optimized_config()

    print("\n⚙️  Configuration:")
    print(f"   • Chunk size: {config.chunk_size} characters")
    print(f"   • Overlap: {config.overlap_size} characters")
    print(f"   • Language: {config.language}")
    print(f"   • Auto-detect language: {config.auto_detect_language}")
    print(f"   • Temperature: {config.temperature}")

    # Create agent
    agent = MainProcessingAgent(llm_provider=llm_provider, config=config)

    # Process all files
    process_all_files(agent=agent)

    return 0


if __name__ == "__main__":
    exit(main())
