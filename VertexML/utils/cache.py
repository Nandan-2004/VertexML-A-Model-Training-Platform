"""Caching utilities for VertexML."""

_cache = {}


def get(key):
    """Get cached value."""
    return _cache.get(key)


def set(key, value):
    """Set cached value."""
    _cache[key] = value
