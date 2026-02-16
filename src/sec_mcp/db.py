"""MongoDB persistence layer for SEC Terminal.

Stores extracted filing data so subsequent requests are instant.
Falls back GRACEFULLY when:
  - MONGODB_URI is not set
  - MongoDB is unreachable
  - User lacks permissions on sec_terminal database (Atlas auth issue)

Every public function catches exceptions and returns safe defaults
so the app NEVER crashes from a MongoDB problem.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_client: Any = None
_db: Any = None
_available: bool | None = None


def _get_db():
    """Lazy-init MongoDB connection. Returns None if unavailable or unauthorized."""
    global _client, _db, _available
    if _available is False:
        return None
    if _db is not None:
        return _db

    try:
        from sec_mcp.config import get_config
        uri = get_config().mongodb_uri
        if not uri:
            log.info("MONGODB_URI not set — running without persistent cache")
            _available = False
            return None

        from pymongo import MongoClient
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Ping confirms connectivity, but NOT database-level authorization
        _client.admin.command("ping")

        _db = _client.sec_terminal

        # Test an actual read on our database to verify permissions
        # If the user doesn't have readWrite on sec_terminal, this throws
        _db.companies.find_one({}, {"_id": 1})

        # Create indexes (only runs once per connection)
        _db.filings.create_index([("ticker", 1), ("accession", 1)], unique=True)
        _db.filings.create_index([("ticker", 1), ("form_type", 1), ("filing_date", -1)])
        _db.companies.create_index("ticker", unique=True)
        _db.jobs.create_index("ticker", unique=True)

        _available = True
        log.info("MongoDB connected and authorized on sec_terminal")
        return _db

    except Exception as exc:
        log.warning("MongoDB unavailable: %s", exc)
        _available = False
        _db = None
        _client = None
        return None


def is_available() -> bool:
    """Check if MongoDB is connected and authorized."""
    _get_db()
    return _available is True


# ── Companies ──────────────────────────────────────────────────────────

def upsert_company(ticker: str, data: dict):
    """Save company info. Silently fails if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return
        data["ticker"] = ticker.upper()
        data["updated_at"] = datetime.now(timezone.utc)
        db.companies.update_one({"ticker": ticker.upper()}, {"$set": data}, upsert=True)
    except Exception as exc:
        log.debug("upsert_company failed: %s", exc)


def get_company(ticker: str) -> dict | None:
    """Get company info. Returns None if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return None
        return db.companies.find_one({"ticker": ticker.upper()}, {"_id": 0})
    except Exception as exc:
        log.debug("get_company failed: %s", exc)
        return None


# ── Filings ────────────────────────────────────────────────────────────

def upsert_filing(ticker: str, accession: str, data: dict):
    """Save a filing record. Silently fails if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return
        data["ticker"] = ticker.upper()
        data["accession"] = accession
        data["processed_at"] = datetime.now(timezone.utc).isoformat()
        db.filings.update_one(
            {"ticker": ticker.upper(), "accession": accession},
            {"$set": data}, upsert=True,
        )
    except Exception as exc:
        log.debug("upsert_filing failed: %s", exc)


def get_filings(ticker: str, form_type: str | None = None, limit: int = 500) -> list[dict]:
    """List filings for a ticker. Returns [] if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return []
        query: dict = {"ticker": ticker.upper()}
        if form_type:
            query["form_type"] = form_type
        return list(db.filings.find(query, {"_id": 0}).sort("filing_date", -1).limit(limit))
    except Exception as exc:
        log.debug("get_filings failed: %s", exc)
        return []


def get_filing(ticker: str, accession: str) -> dict | None:
    """Get a single filing. Returns None if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return None
        return db.filings.find_one(
            {"ticker": ticker.upper(), "accession": accession}, {"_id": 0},
        )
    except Exception as exc:
        log.debug("get_filing failed: %s", exc)
        return None


def count_filings(ticker: str) -> int:
    """Count filings for a ticker. Returns 0 if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return 0
        return db.filings.count_documents({"ticker": ticker.upper()})
    except Exception as exc:
        log.debug("count_filings failed: %s", exc)
        return 0


# ── Extraction jobs ────────────────────────────────────────────────────

def set_job(ticker: str, status: str, progress: int = 0, total: int = 0, detail: str = ""):
    """Update extraction job status. Silently fails if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return
        db.jobs.update_one(
            {"ticker": ticker.upper()},
            {"$set": {
                "ticker": ticker.upper(),
                "status": status,
                "progress": progress,
                "total": total,
                "detail": detail,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )
    except Exception as exc:
        log.debug("set_job failed: %s", exc)


def get_job(ticker: str) -> dict | None:
    """Get extraction job status. Returns None if MongoDB unavailable."""
    try:
        db = _get_db()
        if db is None:
            return None
        return db.jobs.find_one({"ticker": ticker.upper()}, {"_id": 0})
    except Exception as exc:
        log.debug("get_job failed: %s", exc)
        return None
