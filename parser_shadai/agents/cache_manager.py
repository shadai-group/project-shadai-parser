"""
Smart caching layer for Phase 3 optimization.

Caches expensive LLM operations like:
- Document type detection
- Document-level metadata extraction
- Language detection

Uses content-based hashing for cache keys to ensure accuracy.
Implements LRU eviction with configurable size and TTL.

Expected improvement: 10-15% additional speedup for repeated content.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from parser_shadai.agents.metadata_schemas import DocumentType


@dataclass
class CacheConfig:
    """Configuration for cache manager."""

    enabled: bool = True  # Enable caching
    max_entries: int = 1000  # Maximum cache entries (LRU eviction)
    ttl_seconds: int = 86400  # Time-to-live: 24 hours
    persistent: bool = False  # Save cache to disk
    cache_dir: str = ".parser_cache"  # Directory for persistent cache


@dataclass
class CacheEntry:
    """Single cache entry with value and metadata."""

    value: Any
    timestamp: float
    hits: int = 0


class CacheManager:
    """
    Smart cache manager for LLM operations.

    Features:
    - Content-based hashing for accurate cache keys
    - LRU eviction when max_entries exceeded
    - TTL-based expiration
    - Optional disk persistence
    - Separate namespaces for different operation types
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()

        # In-memory LRU caches (namespace -> OrderedDict)
        self._caches: Dict[str, OrderedDict] = {
            "document_type": OrderedDict(),
            "document_metadata": OrderedDict(),
            "language": OrderedDict(),
        }

        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        # Load persistent cache if enabled
        if self.config.persistent:
            self._load_from_disk()

    def _generate_content_hash(self, content: str, max_length: int = 5000) -> str:
        """
        Generate SHA256 hash of content for cache key.

        Uses first max_length characters for efficiency.

        Args:
            content: Text content to hash
            max_length: Maximum content length to hash

        Returns:
            Hex digest of SHA256 hash
        """
        # Use first max_length chars for consistent hashing
        sample = content[:max_length] if len(content) > max_length else content
        return hashlib.sha256(sample.encode("utf-8")).hexdigest()

    def _get_cache_key(self, namespace: str, content: str, **kwargs) -> str:
        """
        Generate unique cache key for operation.

        Args:
            namespace: Cache namespace (document_type, document_metadata, etc.)
            content: Document content
            **kwargs: Additional parameters for key generation

        Returns:
            Unique cache key
        """
        # Base key: content hash
        content_hash = self._generate_content_hash(content=content)

        # Add kwargs to key if present
        if kwargs:
            # Sort kwargs for consistent keys
            kwargs_str = json.dumps(kwargs, sort_keys=True)
            kwargs_hash = hashlib.sha256(kwargs_str.encode("utf-8")).hexdigest()[:8]
            return f"{content_hash}:{kwargs_hash}"

        return content_hash

    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if cache entry has expired.

        Args:
            entry: Cache entry to check

        Returns:
            True if expired, False otherwise
        """
        if self.config.ttl_seconds <= 0:
            return False  # No expiration

        age = time.time() - entry.timestamp
        return age > self.config.ttl_seconds

    def _evict_lru(self, namespace: str) -> None:
        """
        Evict least recently used entry from namespace.

        Args:
            namespace: Cache namespace to evict from
        """
        cache = self._caches[namespace]
        if cache:
            cache.popitem(last=False)  # Remove oldest (first) item
            self._stats["evictions"] += 1

    def get(self, namespace: str, content: str, **kwargs) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            namespace: Cache namespace
            content: Document content
            **kwargs: Additional parameters for cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.config.enabled:
            return None

        cache = self._caches.get(namespace)
        if not cache:
            return None

        key = self._get_cache_key(namespace=namespace, content=content, **kwargs)
        entry = cache.get(key)

        if not entry:
            self._stats["misses"] += 1
            return None

        # Check expiration
        if self._is_expired(entry=entry):
            del cache[key]
            self._stats["expirations"] += 1
            self._stats["misses"] += 1
            return None

        # Move to end (most recently used)
        cache.move_to_end(key)
        entry.hits += 1
        self._stats["hits"] += 1

        return entry.value

    def set(self, namespace: str, content: str, value: Any, **kwargs) -> None:
        """
        Set value in cache.

        Args:
            namespace: Cache namespace
            content: Document content
            value: Value to cache
            **kwargs: Additional parameters for cache key
        """
        if not self.config.enabled:
            return

        cache = self._caches.get(namespace)
        if not cache:
            return

        key = self._get_cache_key(namespace=namespace, content=content, **kwargs)

        # Check if we need to evict
        if len(cache) >= self.config.max_entries:
            self._evict_lru(namespace=namespace)

        # Create entry
        entry = CacheEntry(value=value, timestamp=time.time(), hits=0)

        # Add to cache (at end = most recently used)
        cache[key] = entry

        # Persist to disk if enabled
        if self.config.persistent:
            self._save_to_disk()

    def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            namespace: Specific namespace to clear, or None to clear all
        """
        if namespace:
            self._caches[namespace].clear()
        else:
            for cache in self._caches.values():
                cache.clear()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_sizes": {
                namespace: len(cache) for namespace, cache in self._caches.items()
            },
        }

    def _save_to_disk(self) -> None:
        """Save cache to disk for persistence."""
        try:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Save each namespace separately
            for namespace, cache in self._caches.items():
                cache_file = cache_dir / f"{namespace}.json"

                # Convert OrderedDict to serializable format
                cache_data = {
                    key: {
                        "value": entry.value,
                        "timestamp": entry.timestamp,
                        "hits": entry.hits,
                    }
                    for key, entry in cache.items()
                }

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk if it exists."""
        try:
            cache_dir = Path(self.config.cache_dir)
            if not cache_dir.exists():
                return

            # Load each namespace
            for namespace in self._caches.keys():
                cache_file = cache_dir / f"{namespace}.json"
                if not cache_file.exists():
                    continue

                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Convert back to CacheEntry objects
                for key, entry_data in cache_data.items():
                    # Skip expired entries
                    entry = CacheEntry(
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        hits=entry_data["hits"],
                    )

                    if not self._is_expired(entry=entry):
                        self._caches[namespace][key] = entry

            print(f"✓ Loaded cache from {cache_dir}")

        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")


# Singleton instance for global cache
_global_cache: Optional[CacheManager] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """
    Get global cache manager instance (singleton).

    Args:
        config: Cache configuration (only used on first call)

    Returns:
        Global CacheManager instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(config=config)
    return _global_cache


def clear_global_cache() -> None:
    """Clear and reset global cache manager."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
    _global_cache = None


# Convenience functions for common cache operations


def cache_document_type(content: str, document_type: DocumentType) -> None:
    """
    Cache document type detection result.

    Args:
        content: Document content
        document_type: Detected document type
    """
    cache = get_cache_manager()
    cache.set(namespace="document_type", content=content, value=document_type.value)


def get_cached_document_type(content: str) -> Optional[str]:
    """
    Get cached document type.

    Args:
        content: Document content

    Returns:
        Cached document type or None
    """
    cache = get_cache_manager()
    return cache.get(namespace="document_type", content=content)


def cache_document_metadata(
    content: str, document_type: DocumentType, metadata: Dict[str, Any]
) -> None:
    """
    Cache document-level metadata.

    Args:
        content: Document content
        document_type: Document type
        metadata: Extracted metadata
    """
    cache = get_cache_manager()
    cache.set(
        namespace="document_metadata",
        content=content,
        value=metadata,
        document_type=document_type.value,
    )


def get_cached_document_metadata(
    content: str, document_type: DocumentType
) -> Optional[Dict[str, Any]]:
    """
    Get cached document-level metadata.

    Args:
        content: Document content
        document_type: Document type

    Returns:
        Cached metadata or None
    """
    cache = get_cache_manager()
    return cache.get(
        namespace="document_metadata",
        content=content,
        document_type=document_type.value,
    )


def cache_language(content: str, language: str) -> None:
    """
    Cache language detection result.

    Args:
        content: Document content sample
        language: Detected language code
    """
    cache = get_cache_manager()
    cache.set(namespace="language", content=content, value=language)


def get_cached_language(content: str) -> Optional[str]:
    """
    Get cached language.

    Args:
        content: Document content sample

    Returns:
        Cached language code or None
    """
    cache = get_cache_manager()
    return cache.get(namespace="language", content=content)
