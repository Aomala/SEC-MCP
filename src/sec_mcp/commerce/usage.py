"""Usage metering + quota enforcement.

The constraint that shapes this module: we just spent the whole latency budget
getting responses to ~0.3s. A metering design that writes a row to Supabase on
every request would put a network round trip back in front of every call and
undo that. So:

  - counters live in memory and are the authority for enforcement (O(1), no IO)
  - a background thread flushes DELTAS to Postgres every FLUSH_INTERVAL seconds
  - on boot we read today's totals back, so a redeploy does not hand every key
    a fresh quota

That trades exactness for speed in a bounded way: a crash can lose up to one
flush interval of counts, and with multiple uvicorn workers each process holds
its own counters, so effective limits scale with worker count. Both are
acceptable for usage-based *quota* (we are protecting capacity, not billing to
the cent) and both are documented rather than hidden. If usage ever becomes the
billing basis, move the daily counter to a Postgres atomic increment or Redis.

Rate limiting is a fixed-window counter per rolling minute — same tradeoff,
same reasoning.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

log = logging.getLogger(__name__)

FLUSH_INTERVAL = 20.0

_lock = threading.Lock()

# key_id → calls today (UTC). The quota authority.
_daily: dict[str, int] = defaultdict(int)
# (key_id, endpoint) → unflushed delta {calls, errors, total_ms}
_pending: dict[tuple[str, str], dict] = {}
# key_id → (minute_bucket, count). The burst authority.
_minute: dict[str, tuple[int, int]] = {}

_day: str = ""
_flusher_started = False


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _roll_day_locked() -> None:
    """Reset daily counters when UTC midnight passes. Caller holds _lock."""
    global _day
    today = _today()
    if today != _day:
        _day = today
        _daily.clear()
        _minute.clear()


def _client():
    from sec_mcp.supabase_cache import _get_client
    return _get_client()


def load_today() -> None:
    """Seed in-memory daily counters from Postgres so a restart does not reset
    every customer's quota. Best-effort: failure just means a generous day."""
    client = _client()
    if not client:
        return
    try:
        res = (
            client.table("api_usage")
            .select("key_id, calls")
            .eq("day", _today())
            .execute()
        )
        with _lock:
            _roll_day_locked()
            for row in res.data or []:
                _daily[row["key_id"]] += int(row.get("calls") or 0)
        log.info("Usage: restored today's counters for %d keys", len(res.data or []))
    except Exception as exc:
        log.warning("Usage: could not restore today's counters: %s", exc)


def check(key_id: str, plan: dict) -> dict:
    """Would this call be allowed? Read-only — does NOT consume quota.

    Returns {allowed, reason, limit, remaining, reset_epoch, rpm_remaining}.
    """
    daily_limit = int(plan.get("daily_calls") or 0)
    rpm_limit = int(plan.get("rpm") or 0)
    now = time.time()
    minute_bucket = int(now // 60)

    with _lock:
        _roll_day_locked()
        used = _daily.get(key_id, 0)
        bucket, count = _minute.get(key_id, (minute_bucket, 0))
        if bucket != minute_bucket:
            count = 0

    # Seconds until UTC midnight — what Retry-After should say for a quota stop.
    now_dt = datetime.now(timezone.utc)
    reset_epoch = int(now_dt.timestamp()) + int(
        86400 - (now_dt.hour * 3600 + now_dt.minute * 60 + now_dt.second))

    if rpm_limit and count >= rpm_limit:
        return {
            "allowed": False, "reason": "rate_limited",
            "limit": daily_limit, "remaining": max(daily_limit - used, 0),
            "reset_epoch": (minute_bucket + 1) * 60,
            "rpm_limit": rpm_limit, "rpm_remaining": 0,
            "retry_after": max(int((minute_bucket + 1) * 60 - now), 1),
        }
    if daily_limit and used >= daily_limit:
        return {
            "allowed": False, "reason": "quota_exceeded",
            "limit": daily_limit, "remaining": 0, "reset_epoch": reset_epoch,
            "rpm_limit": rpm_limit, "rpm_remaining": max(rpm_limit - count, 0),
            "retry_after": max(reset_epoch - int(now), 1),
        }
    return {
        "allowed": True, "reason": None,
        "limit": daily_limit, "remaining": max(daily_limit - used - 1, 0),
        "reset_epoch": reset_epoch,
        "rpm_limit": rpm_limit, "rpm_remaining": max(rpm_limit - count - 1, 0),
        "retry_after": 0,
    }


def record(key_id: str, endpoint: str, *, duration_ms: float = 0.0,
           error: bool = False, consume: bool = True) -> None:
    """Count one call. `consume=False` records for analytics without spending
    quota (used for rejected calls, which should not also eat the budget)."""
    now = time.time()
    minute_bucket = int(now // 60)
    with _lock:
        _roll_day_locked()
        if consume:
            _daily[key_id] += 1
            bucket, count = _minute.get(key_id, (minute_bucket, 0))
            _minute[key_id] = ((minute_bucket, 1) if bucket != minute_bucket
                               else (bucket, count + 1))
        slot = _pending.setdefault((key_id, endpoint),
                                   {"calls": 0, "errors": 0, "total_ms": 0})
        slot["calls"] += 1
        if error:
            slot["errors"] += 1
        slot["total_ms"] += int(duration_ms)


def flush() -> int:
    """Push pending deltas to Postgres. Returns the number of rows written."""
    with _lock:
        if not _pending:
            return 0
        batch = dict(_pending)
        _pending.clear()
        day = _day or _today()

    client = _client()
    if not client:
        return 0

    written = 0
    touched_keys = set()
    for (key_id, endpoint), delta in batch.items():
        try:
            # increment_api_usage is an upsert-and-add in SQL — doing it
            # client-side would race between workers and lose counts.
            client.rpc("increment_api_usage", {
                "p_key_id": key_id,
                "p_day": day,
                "p_endpoint": endpoint,
                "p_calls": delta["calls"],
                "p_errors": delta["errors"],
                "p_total_ms": delta["total_ms"],
            }).execute()
            written += 1
            touched_keys.add(key_id)
        except Exception as exc:
            log.warning("Usage flush failed for %s %s: %s", key_id, endpoint, exc)
            # Put it back so the next flush retries rather than dropping counts.
            with _lock:
                slot = _pending.setdefault((key_id, endpoint),
                                           {"calls": 0, "errors": 0, "total_ms": 0})
                for k in ("calls", "errors", "total_ms"):
                    slot[k] += delta[k]

    from sec_mcp.commerce import keys as _keys
    for key_id in touched_keys:
        _keys.touch_last_used(key_id)
    return written


def start_flusher() -> None:
    """Start the background flush loop (idempotent)."""
    global _flusher_started
    with _lock:
        if _flusher_started:
            return
        _flusher_started = True

    def _loop() -> None:
        while True:
            time.sleep(FLUSH_INTERVAL)
            try:
                flush()
            except Exception as exc:  # noqa: BLE001 — the loop must never die
                log.warning("Usage flusher error: %s", exc)

    threading.Thread(target=_loop, name="usage-flusher", daemon=True).start()
    log.info("Usage flusher started (every %.0fs)", FLUSH_INTERVAL)


def summary_for_key(key_id: str, days: int = 30) -> dict:
    """Usage history for a key — what the customer dashboard reads."""
    client = _client()
    if not client:
        return {"days": [], "totalCalls": 0}
    try:
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        res = (
            client.table("api_usage")
            .select("day, endpoint, calls, errors, total_ms")
            .eq("key_id", key_id)
            .gte("day", since)
            .order("day", desc=True)
            .execute()
        )
        rows = res.data or []
        by_day: dict[str, int] = defaultdict(int)
        by_endpoint: dict[str, int] = defaultdict(int)
        for r in rows:
            by_day[r["day"]] += int(r.get("calls") or 0)
            by_endpoint[r["endpoint"]] += int(r.get("calls") or 0)
        with _lock:
            live_today = _daily.get(key_id, 0)
        return {
            "days": [{"day": d, "calls": c} for d, c in sorted(by_day.items(), reverse=True)],
            "byEndpoint": [{"endpoint": e, "calls": c}
                           for e, c in sorted(by_endpoint.items(), key=lambda x: -x[1])],
            "totalCalls": sum(by_day.values()),
            "todayCalls": max(live_today, by_day.get(_today(), 0)),
        }
    except Exception as exc:
        log.warning("Usage summary failed: %s", exc)
        return {"days": [], "totalCalls": 0}


def stats() -> dict:
    """Process-local metering state — for /health and debugging."""
    with _lock:
        return {
            "day": _day,
            "keys_active_today": len(_daily),
            "calls_today": sum(_daily.values()),
            "pending_flush_rows": len(_pending),
            "flush_interval_s": FLUSH_INTERVAL,
        }
