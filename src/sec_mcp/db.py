"""Supabase persistence layer for SEC Terminal.

Stores extracted filing data so subsequent requests are instant.
Falls back GRACEFULLY when:
  - SUPABASE_URL / SUPABASE_KEY are not set
  - Supabase is unreachable

Every public function catches exceptions and returns safe defaults
so the app NEVER crashes from a database problem.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_client: Any = None
_available: bool | None = None


def _get_client():
    """Lazy-init Supabase connection. Returns None if unavailable."""
    global _client, _available
    if _available is False:
        return None
    if _client is not None:
        return _client

    try:
        from sec_mcp.config import get_config
        cfg = get_config()
        if not cfg.supabase_url or not cfg.supabase_key:
            log.info("SUPABASE_URL/SUPABASE_KEY not set — running without persistent cache")
            _available = False
            return None

        from supabase import create_client
        _client = create_client(cfg.supabase_url, cfg.supabase_key)

        # Verify connectivity with a simple query
        _client.table("companies").select("ticker").limit(1).execute()

        _available = True
        log.info("Supabase connected")
        return _client

    except Exception as exc:
        log.warning("Supabase unavailable: %s", exc)
        _available = False
        _client = None
        return None


def is_available() -> bool:
    """Check if Supabase is connected."""
    _get_client()
    return _available is True


# ── Companies ──────────────────────────────────────────────────────────

def upsert_company(ticker: str, data: dict):
    """Save company info. Silently fails if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return
        row = {
            "ticker": ticker.upper(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Pull known columns out of data, put the rest in jsonb
        for col in ("name", "cik", "industry", "sic_code", "website", "exchange"):
            if col in data:
                row[col] = data[col]
        extra = {k: v for k, v in data.items() if k not in row and k != "ticker"}
        if extra:
            row["data"] = extra
        client.table("companies").upsert(row, on_conflict="ticker").execute()
    except Exception as exc:
        log.debug("upsert_company failed: %s", exc)


def get_company(ticker: str) -> dict | None:
    """Get company info. Returns None if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return None
        resp = client.table("companies").select("*").eq("ticker", ticker.upper()).limit(1).execute()
        if resp.data:
            return resp.data[0]
        return None
    except Exception as exc:
        log.debug("get_company failed: %s", exc)
        return None


# ── Filings ────────────────────────────────────────────────────────────

def upsert_filing(ticker: str, accession: str, data: dict):
    """Save a filing record. Silently fails if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return
        row = {
            "ticker": ticker.upper(),
            "accession": accession,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        for col in ("form_type", "filing_date"):
            if col in data:
                row[col] = data[col]
        extra = {k: v for k, v in data.items() if k not in row and k not in ("ticker", "accession")}
        if extra:
            row["data"] = extra
        client.table("filings").upsert(row, on_conflict="ticker,accession").execute()
    except Exception as exc:
        log.debug("upsert_filing failed: %s", exc)


def get_filings(ticker: str, form_type: str | None = None, limit: int = 500) -> list[dict]:
    """List filings for a ticker. Returns [] if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return []
        query = client.table("filings").select("*").eq("ticker", ticker.upper())
        if form_type:
            query = query.eq("form_type", form_type)
        resp = query.order("filing_date", desc=True).limit(limit).execute()
        return resp.data or []
    except Exception as exc:
        log.debug("get_filings failed: %s", exc)
        return []


def get_filing(ticker: str, accession: str) -> dict | None:
    """Get a single filing. Returns None if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return None
        resp = (
            client.table("filings")
            .select("*")
            .eq("ticker", ticker.upper())
            .eq("accession", accession)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
        return None
    except Exception as exc:
        log.debug("get_filing failed: %s", exc)
        return None


def count_filings(ticker: str) -> int:
    """Count filings for a ticker. Returns 0 if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return 0
        resp = (
            client.table("filings")
            .select("id", count="exact")
            .eq("ticker", ticker.upper())
            .execute()
        )
        return resp.count or 0
    except Exception as exc:
        log.debug("count_filings failed: %s", exc)
        return 0


# ── Extraction jobs ────────────────────────────────────────────────────

def set_job(ticker: str, status: str, progress: int = 0, total: int = 0, detail: str = ""):
    """Update extraction job status. Silently fails if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return
        row = {
            "ticker": ticker.upper(),
            "status": status,
            "progress": progress,
            "total": total,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        client.table("jobs").upsert(row, on_conflict="ticker").execute()
    except Exception as exc:
        log.debug("set_job failed: %s", exc)


def get_job(ticker: str) -> dict | None:
    """Get extraction job status. Returns None if Supabase unavailable."""
    try:
        client = _get_client()
        if client is None:
            return None
        resp = client.table("jobs").select("*").eq("ticker", ticker.upper()).limit(1).execute()
        if resp.data:
            return resp.data[0]
        return None
    except Exception as exc:
        log.debug("get_job failed: %s", exc)
        return None
