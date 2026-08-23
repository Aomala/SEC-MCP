"""Single-flight background refresh — the machinery behind stale-while-revalidate.

Why this exists
---------------
SEC fundamentals are near-immutable: a filed 10-K never changes, and the only
thing a cache refresh buys is picking up the NEXT filing. But `financial_cache`
rows carry a hard `expires_at`, and expiry flipped a 0.4s response into a 45s
one — measured on prod 2026-08-23, where a cold /api/metrics?period=ttm took
40-45s while the warm twin took 0.36s. Worse, the cache had decayed to 42 live
`financials` rows out of 9,663, so ~every request paid the cold price.

Hard expiry is the wrong model for this data. Instead:

  fresh   → serve it
  stale   → serve it ANYWAY (it is a filed document, not a quote) and kick off
            one background rebuild so the next caller gets the new one
  missing → build inline, the only case that may legitimately be slow

Two guarantees this module provides:

1. **Single-flight.** N concurrent callers wanting the same key trigger exactly
   ONE rebuild. Without this, a popular stale ticker summons a thundering herd
   of identical 45s extractions that hammer SEC EDGAR's 8 req/s budget and
   starve the threadpool.
2. **Bounded fan-out.** Background rebuilds are capped, so a burst of stale keys
   can never consume every worker and stall the requests being served.

Callers never wait on a background refresh; `spawn()` returns immediately.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

# Ceiling on concurrent background rebuilds. Foreground requests share the same
# anyio threadpool, so an unbounded background fan-out would starve them.
MAX_CONCURRENT_REFRESH = 4

# A key that just failed is not retried for this long — a ticker whose
# extraction genuinely crashes must not respawn a rebuild on every request.
_FAILURE_BACKOFF_S = 900.0

_lock = threading.Lock()
_inflight: set[str] = set()
_recent_failures: dict[str, float] = {}
_stats = {"spawned": 0, "collapsed": 0, "rejected_busy": 0, "failed": 0, "succeeded": 0}


def _release(key: str, ok: bool) -> None:
    with _lock:
        _inflight.discard(key)
        if ok:
            _stats["succeeded"] += 1
            _recent_failures.pop(key, None)
        else:
            _stats["failed"] += 1
            _recent_failures[key] = time.time()


def spawn(key: str, fn, *args, **kwargs) -> bool:
    """Run `fn(*args, **kwargs)` in the background, at most once per `key`.

    Returns True if this call started the work, False if it was collapsed into
    an in-flight rebuild, refused because the fan-out cap is full, or suppressed
    by the failure backoff. Never raises and never blocks the caller.
    """
    now = time.time()
    with _lock:
        if key in _inflight:
            _stats["collapsed"] += 1
            return False
        failed_at = _recent_failures.get(key)
        if failed_at is not None:
            if now - failed_at < _FAILURE_BACKOFF_S:
                return False
            del _recent_failures[key]
        if len(_inflight) >= MAX_CONCURRENT_REFRESH:
            _stats["rejected_busy"] += 1
            return False
        _inflight.add(key)
        _stats["spawned"] += 1

    def _run() -> None:
        ok = False
        started = time.time()
        try:
            fn(*args, **kwargs)
            ok = True
            log.info("Background refresh done: %s (%.1fs)", key, time.time() - started)
        except Exception as exc:  # noqa: BLE001 — a background failure must stay background
            log.warning("Background refresh failed for %s: %s", key, exc)
        finally:
            _release(key, ok)

    threading.Thread(target=_run, name=f"refresh:{key}"[:60], daemon=True).start()
    return True


def stats() -> dict:
    """Counters for /api/cache/stats — how the stale-while-revalidate path is doing."""
    with _lock:
        return {
            **_stats,
            "inflight": len(_inflight),
            "inflight_keys": sorted(_inflight),
            "backoff_keys": len(_recent_failures),
            "max_concurrent": MAX_CONCURRENT_REFRESH,
        }
