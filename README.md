# Parser Shadai

A powerful Python package for parsing PDFs and images using various Large Language Model (LLM) providers including Google Gemini, Anthropic Claude, and OpenAI.

## Features

- **Multi-LLM Support**: Works with Google Gemini, Anthropic Claude, and OpenAI
- **Document Processing**: Extract text and metadata from PDFs
- **Image Analysis**: OCR, description, classification, and metadata extraction from images
- **Intelligent Chunking**: Smart text chunking with context preservation
- **Language Detection**: Automatic language detection with LLM support
- **Structured Metadata**: Document-type specific metadata extraction
- **Vector Search**: ChromaDB integration for similarity search

## Installation

```bash
pip install parser-shadai
```

## Quick Start

```python
from parser_shadai import MainProcessingAgent, GeminiProvider, AgentConfig, DocumentType

# Configure the agent
config = AgentConfig(
    chunk_size=1000,
    overlap_size=150,
    temperature=0.2,
    language="en",
    auto_detect_language=True
    extract_text_from_images=True
)

# Initialize with your preferred LLM provider
agent = MainProcessingAgent(
    GeminiProvider(api_key="your-api-key"),
    config
)

# Process a PDF document
result = agent.process_file("document.pdf", DocumentType.LEGAL)

# Process an image
result = agent.process_file("image.png")

```

## Supported LLM Providers

- **Google Gemini**: `GeminiProvider`
- **Anthropic Claude**: `AnthropicProvider`
- **OpenAI**: `OpenAIProvider`

## Document Types (to improve parsing for specifics types)

- Legal documents
- Medical documents
- Financial documents
- Technical documents
- Academic documents
- Business documents
- General documents

## You can define new types with their metadata

## Language Support

Supports 6 languages with automatic detection:

- English (en)
- Spanish (es)
- French (fr)
- Italian (it)
- Portuguese (pt)
- Japanese (ja)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
