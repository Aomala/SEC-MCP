"""Persistent disk cache for SEC filing extraction results.

Caches the output of extract_financials() to disk so results survive
server restarts and don't require re-hitting SEC EDGAR on repeat visits.

Storage: ~/.sec_mcp_cache/{TICKER}/{safe_accession}.json
TTL: 7 days (SEC filings are immutable once filed)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".sec_mcp_cache"
CACHE_TTL = 86400 * 7  # 7 days


def _path(ticker: str, accession: str) -> Path:
    safe = accession.replace("-", "_").replace("/", "_")
    return CACHE_DIR / ticker.upper() / f"{safe}.json"


def get(ticker: str, accession: str) -> dict | None:
    """Return cached result or None if missing/expired."""
    p = _path(ticker, accession)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - raw.get("cached_at", 0) > CACHE_TTL:
            p.unlink(missing_ok=True)
            return None
        log.debug("Disk cache hit: %s / %s", ticker, accession)
        return raw  # {"data": ..., "summary": ..., "cached_at": ...}
    except Exception as exc:
        log.warning("Disk cache read error for %s/%s: %s", ticker, accession, exc)
        return None


def put(ticker: str, accession: str, data: dict, summary: str) -> None:
    """Write result to disk cache."""
    p = _path(ticker, accession)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "data": data,
            "summary": summary,
            "cached_at": time.time(),
        }
        p.write_text(json.dumps(payload, default=str, ensure_ascii=False), encoding="utf-8")
        log.debug("Disk cache write: %s / %s", ticker, accession)
    except Exception as exc:
        log.warning("Disk cache write error for %s/%s: %s", ticker, accession, exc)


def stats() -> dict:
    """Return cache statistics."""
    if not CACHE_DIR.exists():
        return {"tickers": 0, "entries": 0, "size_mb": 0.0, "cache_dir": str(CACHE_DIR)}
    entries = list(CACHE_DIR.glob("**/*.json"))
    tickers = {p.parent.name for p in entries}
    total_bytes = sum(p.stat().st_size for p in entries if p.exists())
    return {
        "tickers": len(tickers),
        "entries": len(entries),
        "size_mb": round(total_bytes / 1_000_000, 2),
        "cache_dir": str(CACHE_DIR),
    }


def clear(ticker: str | None = None) -> int:
    """Delete cache entries. If ticker given, only clear that ticker."""
    count = 0
    if ticker:
        d = CACHE_DIR / ticker.upper()
        if d.exists():
            for f in d.glob("*.json"):
                f.unlink(missing_ok=True)
                count += 1
    else:
        for f in CACHE_DIR.glob("**/*.json"):
            f.unlink(missing_ok=True)
            count += 1
    return count


def list_tickers() -> list[dict]:
    """List all cached tickers with entry count and latest cached_at."""
    if not CACHE_DIR.exists():
        return []
    result = []
    for ticker_dir in sorted(CACHE_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        entries = list(ticker_dir.glob("*.json"))
        latest = 0.0
        for e in entries:
            try:
                raw = json.loads(e.read_text(encoding="utf-8"))
                ts = raw.get("cached_at", 0)
                if ts > latest:
                    latest = ts
            except Exception:
                pass
        if entries:
            result.append({
                "ticker": ticker_dir.name,
                "filings": len(entries),
                "last_cached": latest,
            })
    return result
