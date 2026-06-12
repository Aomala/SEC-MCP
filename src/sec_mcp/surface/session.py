"""US market-session + EDGAR-business-hours clock.

All time logic for the tool surface lives here so tests can freeze one
function (`_now`) and prove 24/7 behavior at a simulated Sunday 3am.
"""

from __future__ import annotations

# stdlib datetime + IANA tz database — no external deps needed
from datetime import datetime, timezone
from datetime import time as dtime
from zoneinfo import ZoneInfo

# Both NYSE/Nasdaq sessions and EDGAR acceptance windows run on US/Eastern
ET = ZoneInfo("America/New_York")

# NYSE full-day market holidays, 2025-2027 (source: nyse.com holiday calendar).
# Static table — revisit annually; an unknown future year simply means
# holiday detection degrades to weekday/weekend logic (safe, never an error).
_NYSE_HOLIDAYS: set[str] = {
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def _now() -> datetime:
    """Single source of wall-clock truth — tests monkeypatch THIS function."""
    return datetime.now(timezone.utc)


def _to_et(now: datetime | None) -> datetime:
    """Normalize any datetime (or None = real now) to US/Eastern."""
    base = now or _now()
    # Treat naive datetimes as UTC so callers can pass plain datetimes safely
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.astimezone(ET)


def is_market_holiday(now: datetime | None = None) -> bool:
    """True when the ET calendar date is a full-day NYSE holiday."""
    return _to_et(now).strftime("%Y-%m-%d") in _NYSE_HOLIDAYS


def market_session(now: datetime | None = None) -> str:
    """Classify the current US equity session.

    Returns one of: "pre" | "regular" | "after" | "closed".
    Weekends and NYSE holidays are always "closed".
    """
    et = _to_et(now)
    # Saturday=5 / Sunday=6 → no session at all
    if et.weekday() >= 5:
        return "closed"
    # Full-day exchange holidays → closed regardless of hour
    if is_market_holiday(et):
        return "closed"
    t = et.time()
    # Pre-market tape runs 04:00–09:30 ET
    if dtime(4, 0) <= t < dtime(9, 30):
        return "pre"
    # Regular session 09:30–16:00 ET
    if dtime(9, 30) <= t < dtime(16, 0):
        return "regular"
    # After-hours tape 16:00–20:00 ET
    if dtime(16, 0) <= t < dtime(20, 0):
        return "after"
    # Overnight → closed
    return "closed"


def edgar_business_hours(now: datetime | None = None) -> bool:
    """True during EDGAR's live acceptance window (06:00–22:00 ET weekdays).

    EDGAR accepts filings 06:00–22:00 ET Mon-Fri; outside that window the
    filings index cannot change, so we can poll it far less often.
    """
    et = _to_et(now)
    # No filings are accepted on weekends
    if et.weekday() >= 5:
        return False
    # Federal holidays close EDGAR too — reuse the NYSE table as a proxy
    if is_market_holiday(et):
        return False
    # Inside the dissemination window?
    return dtime(6, 0) <= et.time() < dtime(22, 0)


def filings_index_ttl(now: datetime | None = None) -> int:
    """Cache TTL (seconds) for the filings index.

    60s while EDGAR can accept new filings (near-real-time discovery),
    600s off-hours (the index is frozen — no reason to poll faster).
    """
    return 60 if edgar_business_hours(now) else 600


def quote_ttl(session: str | None = None, now: datetime | None = None) -> int:
    """Session-aware quote cache TTL (seconds).

    regular: 30s (prices move tick-by-tick)
    pre/after: 120s (thin tape, slower movement)
    closed: 3600s (last close cannot change until the next session)
    """
    s = session or market_session(now)
    if s == "regular":
        return 30
    if s in ("pre", "after"):
        return 120
    return 3600
