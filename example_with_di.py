"""
Example: Using parser-shadai with Dependency Injection (Phase 2).

This example demonstrates the new clean architecture with:
- Factory Pattern for object creation
- Dependency Injection for loose coupling
- Configuration Object for centralized config
- Service Locator pattern for dependency management

This is more testable, maintainable, and follows SOLID principles.
"""

import os
from parser_shadai.infrastructure.container import Container, ParserConfiguration


def example_with_dependency_injection():
    """
    Example using DI container (NEW - Phase 2).

    Benefits:
    - All dependencies injected (testable)
    - Centralized configuration
    - Easy to swap implementations
    - Follows Dependency Inversion Principle
    """
    print("=" * 70)
    print("Example: Parser with Dependency Injection (Phase 2)")
    print("=" * 70)

    # Step 1: Create configuration object
    config = ParserConfiguration(
        # LLM Provider
        llm_provider_name="gemini",
        llm_credentials=os.getenv("GEMINI_API_KEY", ""),
        llm_model="gemini-2.0-flash-exp",
        temperature=0.2,
        # Chunking
        chunker_type="smart",
        chunk_size=4000,
        chunk_overlap=400,
        # Language detection
        auto_detect_language=False,  # Disabled for production
        language="multilingual",
        # OCR
        extract_text_from_images=True,
    )

    print("\n1. Configuration created:")
    print(f"   - Provider: {config.llm_provider_name}")
    print(f"   - Model: {config.llm_model}")
    print(f"   - Chunk size: {config.chunk_size}")
    print(f"   - Language detection: {config.auto_detect_language}")

    # Step 2: Create DI container
    container = Container(config)

    print("\n2. DI container created")

    # Step 3: Get dependencies from container (singletons)
    llm_provider = container.get_llm_provider()
    print(f"\n3. LLM Provider created: {type(llm_provider).__name__}")

    chunker = container.get_chunker()
    print(f"4. Text Chunker created: {type(chunker).__name__}")

    # Step 4: Process a document
    file_path = "example.pdf"  # Replace with actual file

    if os.path.exists(file_path):
        print("\n5. Creating DocumentAgent with DI...")
        agent = container.create_document_agent(file_path)

        print(f"6. Processing document: {file_path}")
        result = agent.process_document(file_path, auto_detect_type=True)

        print("\n✓ Document processed successfully!")
        print(f"   - Chunks: {result.get('total_chunks', 0)}")
        print(
            f"   - Document type: {result.get('document_info', {}).get('type', 'unknown')}"
        )
        print(
            f"   - Language: {result.get('document_info', {}).get('language', 'unknown')}"
        )

    else:
        print(f"\n⚠️  File not found: {file_path}")
        print("   Skipping document processing demo")

    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)


def example_manual_factory_usage():
    """
    Example using factories manually (without container).

    Shows how factories can be used independently.
    """
    from parser_shadai.infrastructure.factories import (
        ChunkerFactory,
        LLMProviderFactory,
        ParserFactory,
    )

    print("\n\n" + "=" * 70)
    print("Example: Manual Factory Usage (Phase 2)")
    print("=" * 70)

    # Create LLM provider using factory
    print("\n1. Creating LLM provider with factory...")
    llm_provider = LLMProviderFactory.create(
        provider_name="gemini",
        credentials=os.getenv("GEMINI_API_KEY", ""),
        model="gemini-2.0-flash-exp",
    )
    print(f"   ✓ Created: {type(llm_provider).__name__}")

    # Create parser using factory
    print("\n2. Creating PDF parser with factory...")
    pdf_parser = ParserFactory.create_pdf_parser(llm_provider)
    print(f"   ✓ Created: {type(pdf_parser).__name__}")

    # Create chunker using factory
    print("\n3. Creating chunker with factory...")
    chunker = ChunkerFactory.create_smart_chunker(
        chunk_size=4000,
        overlap_size=400,
    )
    print(f"   ✓ Created: {type(chunker).__name__}")

    # Use them independently
    print("\n4. Factories can be used independently without container")
    print("   Benefits:")
    print("   - Centralized object creation logic")
    print("   - Easy to switch implementations")
    print("   - Testable (can mock factories)")

    print("\n" + "=" * 70)


def example_old_vs_new():
    """
    Comparison: Old approach vs New approach.
    """
    print("\n\n" + "=" * 70)
    print("Comparison: Old vs New Architecture")
    print("=" * 70)

    print("\n❌ OLD APPROACH (Before Phase 2):")
    print("-" * 70)
    print("""
    from parser_shadai.agents.document_agent import DocumentAgent
    from parser_shadai.llm_providers import GeminiProvider

    # Manual object creation (tight coupling)
    llm_provider = GeminiProvider(
        api_key="api-key",
        model="gemini-2.0-flash-exp"
    )

    # DocumentAgent creates dependencies internally
    agent = DocumentAgent(llm_provider, config)

    # Problems:
    # - DocumentAgent creates PDFParser internally (can't inject mock)
    # - DocumentAgent creates Chunker internally (can't swap)
    # - Hard to test in isolation
    # - Violates Dependency Inversion Principle
    """)

    print("\n✅ NEW APPROACH (After Phase 2):")
    print("-" * 70)
    print("""
    from parser_shadai.infrastructure.container import (
        Container,
        ParserConfiguration
    )

    # Centralized configuration
    config = ParserConfiguration(
        llm_provider_name="gemini",
        llm_credentials="api-key",
        llm_model="gemini-2.0-flash-exp"
    )

    # DI container manages dependencies
    container = Container(config)

    # All dependencies injected
    agent = container.create_document_agent("file.pdf")

    # Benefits:
    # - Loose coupling (all deps injected)
    # - Easy to test (inject mocks)
    # - Easy to swap implementations
    # - Follows SOLID principles
    # - Centralized configuration
    """)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run examples
    example_old_vs_new()
    example_manual_factory_usage()
    example_with_dependency_injection()

    print("\n\n✨ All examples completed!")
    print("\nKey Takeaways:")
    print("1. ✅ Use factories for object creation")
    print("2. ✅ Use DI container for wiring dependencies")
    print("3. ✅ Use Configuration object for centralized config")
    print("4. ✅ Inject dependencies instead of creating them")
    print("5. ✅ Test with mocks (inject fake implementations)")
