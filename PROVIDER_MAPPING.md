# Provider Name Mapping

This document maps provider names between the Django API and the Parser ProviderFactory.

## Supported Providers

| Django API (`apps/llms/choices.py`) | Parser Factory | Status | Notes |
|--------------------------------------|----------------|--------|-------|
| `openai` | `openai` | ✅ Supported | Direct match |
| `anthropic` | `anthropic` | ✅ Supported | Direct match |
| `google_genai` | `google_genai`, `google`, `gemini` | ✅ Supported | Alias added for Django API compatibility |
| `bedrock_converse` | `bedrock_converse`, `bedrock` | ⚠️ Placeholder | Not yet implemented (returns NotImplementedError) |
| `cohere` | N/A | ❌ Not Supported | Not implemented in parser |
| `amazon` | N/A | ❌ Not Supported | Not implemented in parser |

## Provider Factory Constants

**Location**: `parser_shadai/agents/strategies/provider_factory.py`

```python
class ProviderFactory:
    # Provider name constants
    GOOGLE = "google"                        # Original alias
    GEMINI = "gemini"                        # Original alias
    GOOGLE_GENAI = "google_genai"            # Django API alias (added 2025-10-13)
    ANTHROPIC = "anthropic"                  # Matches Django API
    OPENAI = "openai"                        # Matches Django API
    BEDROCK = "bedrock"                      # Original alias
    BEDROCK_CONVERSE = "bedrock_converse"    # Django API alias (added 2025-10-13)
```

## Usage in Django API

**Location**: `apps/rag/tasks/ingestion/services.py`

```python
from parser_shadai.agents.strategies import ProviderFactory

factory = ProviderFactory()
provider = factory.create_provider(
    provider_name="google_genai",  # From Django ProviderNames.GOOGLE
    credentials=api_key
)
```

### Provider Name Flow

1. **Database**: Provider.name = `"google_genai"` (from `ProviderNames.GOOGLE`)
2. **Ingestion Service**: Passes `"google_genai"` to ProviderFactory
3. **ProviderFactory**: Recognizes `"google_genai"` as alias for Gemini
4. **Creates**: `GeminiProvider(api_key=...)`

## Implementation Details

### Google/Gemini Provider

The factory accepts multiple aliases for Google's LLM provider:
- `"google"` - Original parser alias
- `"gemini"` - Alternative parser alias
- `"google_genai"` - Django API alias (matches database values)

All three resolve to the same `GeminiProvider` instance.

### Bedrock Provider

The factory accepts multiple aliases for AWS Bedrock:
- `"bedrock"` - Original parser alias
- `"bedrock_converse"` - Django API alias (matches database values)

**Note**: Bedrock provider is not yet implemented. The factory will raise `NotImplementedError` when attempting to create a Bedrock provider.

### Error Handling

If an unsupported provider name is provided, the factory raises:

```python
ValueError: Unsupported provider: {name}. Supported providers: google, google_genai, gemini, anthropic, openai, bedrock, bedrock_converse
```

## Future Providers

To add new providers to the parser:

1. **Add constant** to `ProviderFactory`:
   ```python
   COHERE = "cohere"
   ```

2. **Add to supported list**:
   ```python
   if provider_name_lower == cls.COHERE:
       return self._create_cohere_provider(api_key=credentials, **kwargs)
   ```

3. **Implement provider class**:
   ```python
   class CohereProvider(BaseLLMProvider):
       # Implementation
   ```

4. **Update `get_supported_providers()`** to include new provider

## Testing

Test provider name resolution:

```python
from parser_shadai.agents.strategies import ProviderFactory

factory = ProviderFactory()

# Test Django API names
assert factory.is_supported("google_genai")  # True
assert factory.is_supported("bedrock_converse")  # True
assert factory.is_supported("openai")  # True
assert factory.is_supported("anthropic")  # True

# Test original parser aliases
assert factory.is_supported("google")  # True
assert factory.is_supported("gemini")  # True
assert factory.is_supported("bedrock")  # True

# Test unsupported
assert not factory.is_supported("cohere")  # False
assert not factory.is_supported("amazon")  # False
```

## Compatibility Matrix

| Parser Version | Django API Version | google_genai | bedrock_converse |
|----------------|-------------------|--------------|------------------|
| < Phase 4 | Any | ❌ Not supported | ❌ Not supported |
| Phase 4 (2025-10-13+) | Current | ✅ Supported | ⚠️ Placeholder |

## Migration Notes

**No migration required** for existing code:
- Original aliases (`google`, `gemini`, `anthropic`, `openai`, `bedrock`) still work
- New aliases added without breaking changes
- Backward compatible with all existing parser usage

---

*Last Updated: 2025-10-13*
*Version: Phase 4 with Django API Compatibility*
