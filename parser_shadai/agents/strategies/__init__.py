"""
Strategy pattern implementations for SOLID refactoring.

Contains concrete implementations of abstract interfaces:
- Metadata extraction strategies
- Document type detection strategies
- Provider factories
"""

from .metadata_extractors import (
    LegacyMetadataExtractor,
    OptimizedMetadataStrategy,
)
from .provider_factory import AWSCredentials, AzureCredentials, ProviderFactory

__all__ = [
    "LegacyMetadataExtractor",
    "OptimizedMetadataStrategy",
    "ProviderFactory",
    "AWSCredentials",
    "AzureCredentials",
]
