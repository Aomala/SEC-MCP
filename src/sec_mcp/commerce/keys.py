"""API key issuance, hashing, and lookup.

Storage rule: the database NEVER holds a usable key. We store a SHA-256 of the
full secret plus a short non-secret prefix. The prefix is what a customer sees
in a dashboard ("fin_live_7Kq2…") and what we put in logs; the secret itself is
shown exactly once, at creation. A database leak therefore leaks no credentials.

Lookup is by hash, which is the indexed unique column — so verification is one
equality probe, not a scan-and-compare over every row.

Reads are memoized for KEY_CACHE_TTL seconds. Auth runs on every single request
and Supabase is a network hop away; without this, adding auth would have undone
the latency work. Revocation therefore takes up to KEY_CACHE_TTL to propagate,
which is the deliberate trade — call `invalidate()` after a revoke to make it
immediate in the process that performed it.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time

log = logging.getLogger(__name__)

KEY_CACHE_TTL = 60.0

# Live keys and test keys are visually distinct so nobody pastes the wrong one.
_PREFIXES = {"live": "fin_live_", "test": "fin_test_"}

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict | None]] = {}


def _client():
    from sec_mcp.supabase_cache import _get_client
    return _get_client()


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate(mode: str = "live") -> tuple[str, str, str]:
    """Mint a key. Returns (full_secret, visible_prefix, hash).

    The secret is 32 url-safe bytes (~256 bits) — brute force is not a threat
    model we need to think about again.
    """
    body = secrets.token_urlsafe(32)
    prefix_tag = _PREFIXES.get(mode, _PREFIXES["live"])
    full = f"{prefix_tag}{body}"
    # Visible prefix = tag + first 8 chars: enough to identify a key in a list
    # without being enough to use it.
    visible = f"{prefix_tag}{body[:8]}"
    return full, visible, hash_key(full)


def create(owner_email: str, plan: str = "free", name: str = "",
           mode: str = "live", metadata: dict | None = None) -> dict | None:
    """Issue and persist a key. The returned dict is the ONLY time `key` appears."""
    client = _client()
    if not client:
        log.error("Cannot create API key — Supabase unavailable")
        return None
    full, visible, digest = generate(mode)
    row = {
        "key_prefix": visible,
        "key_hash": digest,
        "name": name or "default",
        "owner_email": owner_email.strip().lower(),
        "plan": plan,
        "status": "active",
        "metadata": metadata or {},
    }
    try:
        res = client.table("api_keys").insert(row).execute()
    except Exception as exc:
        log.error("API key insert failed: %s", exc)
        return None
    created = (res.data or [{}])[0]
    return {
        "key": full,                       # show once, never retrievable again
        "keyPrefix": visible,
        "plan": plan,
        "ownerEmail": row["owner_email"],
        "id": created.get("id"),
        "createdAt": created.get("created_at"),
    }


def lookup(raw_key: str) -> dict | None:
    """Resolve a presented key to its active record, or None.

    Returns None for unknown AND for revoked keys — a caller cannot tell the
    difference, which is what we want.
    """
    if not raw_key:
        return None
    digest = hash_key(raw_key)
    now = time.time()

    with _lock:
        hit = _cache.get(digest)
        if hit and (now - hit[0]) < KEY_CACHE_TTL:
            return hit[1]

    record = None
    client = _client()
    if client:
        try:
            res = (
                client.table("api_keys")
                .select("id, key_prefix, owner_email, plan, status, name, metadata")
                .eq("key_hash", digest)
                .eq("status", "active")
                .limit(1)
                .execute()
            )
            if res.data:
                record = res.data[0]
        except Exception as exc:
            # A Supabase blip must not 500 every authed request. Serve the last
            # known answer if we have one; otherwise treat as unauthenticated
            # and let the anon tier apply.
            log.warning("API key lookup failed: %s", exc)
            with _lock:
                stale = _cache.get(digest)
            return stale[1] if stale else None

    with _lock:
        _cache[digest] = (now, record)
    return record


def touch_last_used(key_id: str) -> None:
    """Best-effort last-seen stamp. Called from the metering flush, not the
    request path — a write per request would reintroduce the latency we removed."""
    client = _client()
    if not client or not key_id:
        return
    try:
        from datetime import datetime, timezone
        client.table("api_keys").update(
            {"last_used_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", key_id).execute()
    except Exception as exc:
        log.debug("last_used_at update failed: %s", exc)


def revoke(key_prefix: str) -> bool:
    client = _client()
    if not client:
        return False
    try:
        from datetime import datetime, timezone
        client.table("api_keys").update({
            "status": "revoked",
            "revoked_at": datetime.now(timezone.utc).isoformat(),
        }).eq("key_prefix", key_prefix).execute()
        invalidate()
        return True
    except Exception as exc:
        log.error("API key revoke failed: %s", exc)
        return False


def set_plan(key_prefix: str, plan: str, **extra) -> bool:
    """Move a key to another plan — what the Stripe webhook calls on upgrade,
    downgrade, and cancellation."""
    client = _client()
    if not client:
        return False
    try:
        client.table("api_keys").update({"plan": plan, **extra}).eq(
            "key_prefix", key_prefix).execute()
        invalidate()
        return True
    except Exception as exc:
        log.error("API key plan update failed: %s", exc)
        return False


def list_for_owner(owner_email: str) -> list[dict]:
    client = _client()
    if not client:
        return []
    try:
        res = (
            client.table("api_keys")
            .select("key_prefix, name, plan, status, created_at, last_used_at")
            .eq("owner_email", owner_email.strip().lower())
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        log.error("API key list failed: %s", exc)
        return []


def invalidate() -> None:
    """Drop the lookup cache — call after any mutation for same-process immediacy."""
    with _lock:
        _cache.clear()
