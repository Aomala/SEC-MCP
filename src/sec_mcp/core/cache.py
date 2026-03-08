"""Unified multi-tier cache layer with automatic promotion across L1/L2/L3.

Architecture:
  L1: In-memory dict with TTL (default 5 min) — fastest, per-process
  L2: Disk-based via sec_mcp.disk_cache (default 1 hr) — persistent, per-machine
  L3: MongoDB via sec_mcp.db (default 24 hr) — distributed, shared across processes
  
On cache hit: automatically promotes value to higher tiers.
Graceful degradation: if MongoDB unavailable, uses L1+L2 only.
Thread-safe with threading.Lock across all tiers.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

# Attempt imports for L2 and L3 caching; set availability flags
try:
    from sec_mcp import disk_cache
    DISK_CACHE_AVAILABLE = True
except ImportError:
    DISK_CACHE_AVAILABLE = False

try:
    from sec_mcp import db
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Module logger for all warnings/errors
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Lazy Singleton Pattern (follows sec_client.py convention)
# ═══════════════════════════════════════════════════════════════════════════

# Global CacheLayer instance holder — initially None
_cache_instance: CacheLayer | None = None

# Lock for thread-safe singleton initialization
_cache_lock_init = threading.Lock()


def get_cache() -> CacheLayer:
    """Return or create the global CacheLayer singleton instance.
    
    Thread-safe lazy initialization using double-checked locking.
    First call creates and caches instance; subsequent calls return cached instance.
    """
    # Check global _cache_instance without lock first (fast path)
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    
    # Acquire lock and check again (standard double-checked locking pattern)
    with _cache_lock_init:
        if _cache_instance is None:
            # Create new instance and store in global _cache_instance
            _cache_instance = CacheLayer()
        return _cache_instance


# ═══════════════════════════════════════════════════════════════════════════
#  Cache Entry Data Structure (for L1 in-memory tier)
# ═══════════════════════════════════════════════════════════════════════════

class _L1Entry:
    """L1 in-memory cache entry with TTL tracking.
    
    Attributes:
        value: The cached data (any Python object)
        timestamp: When this entry was created (datetime UTC)
        ttl_seconds: Time to live before expiration
    """
    
    def __init__(self, value: Any, ttl_seconds: int):
        """Initialize L1 cache entry.
        
        Args:
            value: The data to cache
            ttl_seconds: Seconds until this entry expires
        """
        # Store the actual cached value
        self.value = value
        # Capture creation time in UTC
        self.timestamp = datetime.now(timezone.utc)
        # Store TTL in seconds
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL.
        
        Returns:
            True if current time >= timestamp + ttl_seconds, False otherwise
        """
        # Calculate expiration time
        expiration = self.timestamp + timedelta(seconds=self.ttl_seconds)
        # Return True if current UTC time is past expiration
        return datetime.now(timezone.utc) >= expiration


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-Tier Cache Layer Class
# ═══════════════════════════════════════════════════════════════════════════

class CacheLayer:
    """Unified multi-tier cache with L1 (memory), L2 (disk), L3 (MongoDB).
    
    Features:
      - get(key, tier="l1") → checks L1, then L2, then L3; promotes on hit
      - set(key, value, tier="l1") → writes to specified tier + all tiers above
      - invalidate(key) → removes from all tiers
      - Thread-safe: all operations protected by threading.Lock
      - Graceful degradation: works without L2 or L3 if unavailable
      - Automatic promotion: cache hits are re-stored in higher tiers for faster access
    
    Default TTLs:
      - L1 (memory): 5 minutes
      - L2 (disk): 1 hour
      - L3 (MongoDB): 24 hours
    
    Pattern: tier="l1" for frequently-accessed data (prices, tickers)
             tier="l2" for medium-term caching (financials, filings)
             tier="l3" for long-term distributed cache (comparison results)
    """
    
    # Default TTL settings (in seconds) for each cache tier
    L1_TTL = 300  # 5 minutes (L1 = in-memory, fastest)
    L2_TTL = 3600  # 1 hour (L2 = disk, persistent)
    L3_TTL = 86400  # 24 hours (L3 = MongoDB, distributed)
    
    def __init__(self):
        """Initialize CacheLayer with empty L1 dict and availability flags.
        
        Sets up:
          - _l1_cache: dict mapping key → _L1Entry (in-memory, with TTL)
          - _lock: threading.Lock for thread-safe access to all tiers
          - Flags for L2 and L3 availability (set during __init__)
        """
        # L1 (in-memory) cache: key → _L1Entry
        self._l1_cache: dict[str, _L1Entry] = {}
        
        # Lock protecting all cache operations (L1, L2, L3)
        # Ensures thread-safe access across tiers
        self._lock = threading.Lock()
        
        # L2 availability: will be set based on import success
        self._l2_available = DISK_CACHE_AVAILABLE
        # L3 availability: will be set based on import and connectivity
        self._l3_available = MONGODB_AVAILABLE
        
        # Log initialization with tier availability
        msg_parts = ["CacheLayer initialized (L1: memory"]
        if self._l2_available:
            msg_parts.append(", L2: disk")
        if self._l3_available:
            msg_parts.append(", L3: MongoDB")
        msg_parts.append(")")
        log.info("".join(msg_parts))
    
    def get(self, key: str, tier: str = "l1") -> Any | None:
        """Retrieve value from cache, checking tiers in order.
        
        Checks from highest tier down to L1:
          1. Specified tier (if L2 or L3)
          2. Previous tiers down to L1
          Promotes value to higher tiers on hit for faster access next time.
        
        Args:
            key: Cache key (string identifier, e.g., "AAPL_financials_2024")
            tier: Starting tier to check ("l1", "l2", or "l3")
                  Default "l1" means check L1 → L2 → L3
                  If "l3", checks L3 → L2 → L1
                  If "l2", checks L2 → L1
        
        Returns:
            Cached value if found and not expired, None otherwise.
            No exceptions raised — returns None on any error.
        
        Raises:
            No exceptions — logs warnings and returns None on failure.
        """
        # Acquire lock for thread-safe cache access
        with self._lock:
            # ─────────────────────────────────────────────────────────────
            # Check L1 (in-memory) first — fastest
            # ─────────────────────────────────────────────────────────────
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                # Check if entry still valid (not expired)
                if not entry.is_expired():
                    log.debug("Cache hit L1 for key '%s'", key)
                    # Return the cached value to caller
                    return entry.value
                else:
                    # Entry expired, remove it
                    del self._l1_cache[key]
            
            # ─────────────────────────────────────────────────────────────
            # Check L2 (disk) if available and not a L1-only request
            # ─────────────────────────────────────────────────────────────
            if self._l2_available and tier in ["l2", "l3"]:
                try:
                    # Try to retrieve from disk cache
                    l2_value = disk_cache.get(key)
                    
                    # If L2 has the value, promote to L1 and return
                    if l2_value is not None:
                        log.debug("Cache hit L2 for key '%s', promoting to L1", key)
                        # Store in L1 for faster access next time
                        self._l1_cache[key] = _L1Entry(l2_value, self.L1_TTL)
                        # Return value to caller
                        return l2_value
                except Exception as e:
                    # L2 access failed (disk I/O error, corruption, etc.)
                    log.warning("L2 cache access failed for key '%s': %s", key, e)
            
            # ─────────────────────────────────────────────────────────────
            # Check L3 (MongoDB) if available and explicitly requested
            # ─────────────────────────────────────────────────────────────
            if self._l3_available and tier == "l3":
                try:
                    # Try to retrieve from MongoDB cache
                    l3_value = db.get_cached(key)
                    
                    # If L3 has the value, promote through L2→L1 and return
                    if l3_value is not None:
                        log.debug("Cache hit L3 for key '%s', promoting to L1/L2", key)
                        # Promote to L2 (disk)
                        if self._l2_available:
                            try:
                                disk_cache.put(key, l3_value)
                            except Exception as e2:
                                log.warning("Failed to promote L3→L2: %s", e2)
                        # Promote to L1 (memory)
                        self._l1_cache[key] = _L1Entry(l3_value, self.L1_TTL)
                        # Return value to caller
                        return l3_value
                except Exception as e:
                    # L3 access failed (network, MongoDB unavailable, etc.)
                    log.warning("L3 cache access failed for key '%s': %s", key, e)
        
        # No value found in any tier
        log.debug("Cache miss for key '%s'", key)
        return None
    
    def set(self, key: str, value: Any, tier: str = "l1") -> None:
        """Store value to specified tier and all higher tiers.
        
        Writes to specified tier plus all tiers above it:
          - tier="l1": writes to L1 only
          - tier="l2": writes to L1 and L2
          - tier="l3": writes to L1, L2, and L3
        
        Args:
            key: Cache key (string identifier, e.g., "AAPL_financials_2024")
            value: Value to cache (any Python object, should be serializable for L2/L3)
            tier: Target tier ("l1", "l2", or "l3")
                  Default "l1" = only in-memory
        
        Returns:
            None
        
        Raises:
            No exceptions — logs warnings if L2/L3 writes fail, but still returns normally.
        """
        # Acquire lock for thread-safe write access
        with self._lock:
            # ─────────────────────────────────────────────────────────────
            # Always write to L1 (in-memory) — fastest, always available
            # ─────────────────────────────────────────────────────────────
            # Store in L1 with TTL
            self._l1_cache[key] = _L1Entry(value, self.L1_TTL)
            log.debug("Set cache L1 for key '%s'", key)
            
            # ─────────────────────────────────────────────────────────────
            # Write to L2 (disk) if tier >= "l2" and L2 available
            # ─────────────────────────────────────────────────────────────
            if tier in ["l2", "l3"] and self._l2_available:
                try:
                    # Write to disk cache via disk_cache module
                    disk_cache.put(key, value)
                    log.debug("Set cache L2 for key '%s'", key)
                except Exception as e:
                    # L2 write failed; log warning but don't fail the operation
                    log.warning("Failed to write to L2 cache for key '%s': %s", key, e)
            
            # ─────────────────────────────────────────────────────────────
            # Write to L3 (MongoDB) if tier == "l3" and L3 available
            # ─────────────────────────────────────────────────────────────
            if tier == "l3" and self._l3_available:
                try:
                    # Write to MongoDB cache via db module
                    db.put_cached(key, value)
                    log.debug("Set cache L3 for key '%s'", key)
                except Exception as e:
                    # L3 write failed; log warning but don't fail the operation
                    log.warning("Failed to write to L3 cache for key '%s': %s", key, e)
    
    def invalidate(self, key: str | None = None) -> None:
        """Remove entry from all cache tiers.
        
        Args:
            key: Specific key to invalidate, or None to clear all caches entirely.
                 If provided, removes only that key from L1, L2, and L3.
                 If None, clears all entries from all tiers (useful for testing).
        
        Returns:
            None
        
        Raises:
            No exceptions — logs warnings if L2/L3 deletions fail.
        """
        # Acquire lock for thread-safe invalidation
        with self._lock:
            # ─────────────────────────────────────────────────────────────
            # Case 1: Invalidate specific key
            # ─────────────────────────────────────────────────────────────
            if key is not None:
                # Remove from L1 if present
                if key in self._l1_cache:
                    del self._l1_cache[key]
                    log.debug("Invalidated L1 for key '%s'", key)
                
                # Remove from L2 if available
                if self._l2_available:
                    try:
                        disk_cache.delete(key)
                        log.debug("Invalidated L2 for key '%s'", key)
                    except Exception as e:
                        log.warning("Failed to invalidate L2 for key '%s': %s", key, e)
                
                # Remove from L3 if available
                if self._l3_available:
                    try:
                        db.delete_cached(key)
                        log.debug("Invalidated L3 for key '%s'", key)
                    except Exception as e:
                        log.warning("Failed to invalidate L3 for key '%s': %s", key, e)
            
            # ─────────────────────────────────────────────────────────────
            # Case 2: Clear all caches (key is None)
            # ─────────────────────────────────────────────────────────────
            else:
                # Clear L1 (in-memory)
                self._l1_cache.clear()
                log.info("Cleared all L1 cache entries")
                
                # Clear L2 (disk) if available
                if self._l2_available:
                    try:
                        disk_cache.clear()
                        log.info("Cleared all L2 cache entries")
                    except Exception as e:
                        log.warning("Failed to clear L2 cache: %s", e)
                
                # Clear L3 (MongoDB) if available
                if self._l3_available:
                    try:
                        db.clear_cache()
                        log.info("Cleared all L3 cache entries")
                    except Exception as e:
                        log.warning("Failed to clear L3 cache: %s", e)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring and debugging.
        
        Returns:
            Dict with cache stats:
            {
              "l1": {
                "available": True,
                "entries": 42,
                "ttl_seconds": 300,
              },
              "l2": {
                "available": True,
                "ttl_seconds": 3600,
              },
              "l3": {
                "available": True,
                "ttl_seconds": 86400,
              },
            }
        
        Raises:
            No exceptions — returns best-effort stats.
        """
        # Acquire lock for consistent snapshot
        with self._lock:
            # Build stats dict
            stats = {
                "l1": {
                    "available": True,
                    "entries": len(self._l1_cache),
                    "ttl_seconds": self.L1_TTL,
                },
                "l2": {
                    "available": self._l2_available,
                    "ttl_seconds": self.L2_TTL,
                },
                "l3": {
                    "available": self._l3_available,
                    "ttl_seconds": self.L3_TTL,
                },
            }
            return stats
