"""Supabase persistence layer for SEC Terminal.

Uses httpx + PostgREST API directly (no supabase SDK needed).
Stores extracted filing data so subsequent requests are instant.
Falls back GRACEFULLY when:
  - SUPABASE_URL / SUPABASE_KEY are not set
  - Supabase is unreachable

Every public function catches exceptions and returns safe defaults
so the app NEVER crashes from a database problem.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

_headers: dict[str, str] | None = None
_base_url: str = ""
_available: bool | None = None


def _init() -> bool:
    """Lazy-init connection config. Returns True if available."""
    global _headers, _base_url, _available
    if _available is False:
        return False
    if _headers is not None:
        return True

    try:
        from sec_mcp.config import get_config
        cfg = get_config()
        if not cfg.supabase_url or not cfg.supabase_key:
            log.info("SUPABASE_URL/SUPABASE_KEY not set — running without persistent cache")
            _available = False
            return False

        _base_url = cfg.supabase_url.rstrip("/") + "/rest/v1"
        _headers = {
            "apikey": cfg.supabase_key,
            "Authorization": f"Bearer {cfg.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        # Verify connectivity
        resp = httpx.get(
            f"{_base_url}/companies?select=ticker&limit=1",
            headers=_headers,
            timeout=5,
        )
        resp.raise_for_status()

        _available = True
        log.info("Supabase connected (PostgREST)")
        return True

    except Exception as exc:
        log.warning("Supabase unavailable: %s", exc)
        _available = False
        _headers = None
        return False


def is_available() -> bool:
    """Check if Supabase is connected."""
    _init()
    return _available is True


def _post(table: str, data: dict, on_conflict: str = "") -> Any:
    """Upsert a row via PostgREST."""
    headers = {**_headers, "Prefer": "resolution=merge-duplicates,return=representation"}
    url = f"{_base_url}/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    resp = httpx.post(url, headers=headers, json=data, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get(table: str, params: str) -> list[dict]:
    """Query rows via PostgREST."""
    resp = httpx.get(f"{_base_url}/{table}?{params}", headers=_headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Companies ──────────────────────────────────────────────────────────

def upsert_company(ticker: str, data: dict):
    """Save company info. Silently fails if Supabase unavailable."""
    try:
        if not _init():
            return
        row = {
            "ticker": ticker.upper(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for col in ("name", "cik", "industry", "sic_code", "website", "exchange"):
            if col in data:
                row[col] = data[col]
        extra = {k: v for k, v in data.items() if k not in row and k != "ticker"}
        if extra:
            row["data"] = extra
        _post("companies", row, on_conflict="ticker")
    except Exception as exc:
        log.debug("upsert_company failed: %s", exc)


def get_company(ticker: str) -> dict | None:
    """Get company info. Returns None if Supabase unavailable."""
    try:
        if not _init():
            return None
        rows = _get("companies", f"select=*&ticker=eq.{ticker.upper()}&limit=1")
        return rows[0] if rows else None
    except Exception as exc:
        log.debug("get_company failed: %s", exc)
        return None


# ── Filings ────────────────────────────────────────────────────────────

def upsert_filing(ticker: str, accession: str, data: dict):
    """Save a filing record. Silently fails if Supabase unavailable."""
    try:
        if not _init():
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
        _post("filings", row, on_conflict="ticker,accession")
    except Exception as exc:
        log.debug("upsert_filing failed: %s", exc)


def get_filings(ticker: str, form_type: str | None = None, limit: int = 500) -> list[dict]:
    """List filings for a ticker. Returns [] if Supabase unavailable."""
    try:
        if not _init():
            return []
        params = f"select=*&ticker=eq.{ticker.upper()}&order=filing_date.desc&limit={limit}"
        if form_type:
            params += f"&form_type=eq.{form_type}"
        return _get("filings", params)
    except Exception as exc:
        log.debug("get_filings failed: %s", exc)
        return []


def get_filing(ticker: str, accession: str) -> dict | None:
    """Get a single filing. Returns None if Supabase unavailable."""
    try:
        if not _init():
            return None
        rows = _get("filings", f"select=*&ticker=eq.{ticker.upper()}&accession=eq.{accession}&limit=1")
        return rows[0] if rows else None
    except Exception as exc:
        log.debug("get_filing failed: %s", exc)
        return None


def count_filings(ticker: str) -> int:
    """Count filings for a ticker. Returns 0 if Supabase unavailable."""
    try:
        if not _init():
            return 0
        headers = {**_headers, "Prefer": "count=exact"}
        resp = httpx.get(
            f"{_base_url}/filings?select=id&ticker=eq.{ticker.upper()}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        # PostgREST returns count in content-range header
        cr = resp.headers.get("content-range", "")
        # Format: "0-N/total" or "*/total"
        if "/" in cr:
            return int(cr.split("/")[-1])
        return len(resp.json())
    except Exception as exc:
        log.debug("count_filings failed: %s", exc)
        return 0


# ── Extraction jobs ────────────────────────────────────────────────────

def set_job(ticker: str, status: str, progress: int = 0, total: int = 0, detail: str = ""):
    """Update extraction job status. Silently fails if Supabase unavailable."""
    try:
        if not _init():
            return
        row = {
            "ticker": ticker.upper(),
            "status": status,
            "progress": progress,
            "total": total,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _post("jobs", row, on_conflict="ticker")
    except Exception as exc:
        log.debug("set_job failed: %s", exc)


def get_job(ticker: str) -> dict | None:
    """Get extraction job status. Returns None if Supabase unavailable."""
    try:
        if not _init():
            return None
        rows = _get("jobs", f"select=*&ticker=eq.{ticker.upper()}&limit=1")
        return rows[0] if rows else None
    except Exception as exc:
        log.debug("get_job failed: %s", exc)
        return None
