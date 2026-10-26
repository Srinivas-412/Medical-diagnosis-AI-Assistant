"""Disk-based prompt cache with TTL using diskcache."""
import hashlib
import logging
from typing import Any
import diskcache
import config

logger = logging.getLogger(__name__)

_cache = diskcache.Cache(config.CACHE_DIR)


def _make_key(prompt: str, session_id: str = "") -> str:
    """Create a stable cache key from prompt text and optional session context."""
    raw = f"{session_id}::{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(prompt: str, session_id: str = "") -> Any | None:
    """Return cached LLM response, or None on miss."""
    key = _make_key(prompt, session_id)
    value = _cache.get(key)
    if value is not None:
        logger.debug("Cache HIT for key %s", key[:12])
    return value


def set_cached(prompt: str, response: Any, session_id: str = "") -> None:
    """Store LLM response in cache with configured TTL."""
    key = _make_key(prompt, session_id)
    _cache.set(key, response, expire=config.CACHE_TTL_SECONDS)
    logger.debug("Cache SET for key %s (TTL=%ds)", key[:12], config.CACHE_TTL_SECONDS)


def invalidate(prompt: str, session_id: str = "") -> None:
    key = _make_key(prompt, session_id)
    _cache.delete(key)


def clear_all() -> None:
    _cache.clear()
    logger.info("Prompt cache cleared.")
