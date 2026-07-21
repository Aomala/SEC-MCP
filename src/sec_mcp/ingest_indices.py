"""Scheduled worker: market breadth + S&P 500 cap-weighted sector rollup.

The heavy half of the market dashboard. `get_market_overview` reads what this
writes — it must NEVER run on the request path.

One Polygon grouped-daily call returns every US stock's OHLC for a date, so a
`lookback` window of history costs one call PER DAY (not per ticker). From that
window we compute, over the S&P 500 constituent set:

  breadth  — advancers/decliners (last close vs prior close), advance/decline
             ratio, new highs/lows over the window, % above the 50-day SMA.
  sectors  — cap-weighted rollup by coarse sector (market cap = last close ×
             shares outstanding), with a cap-weighted average day change.

Output: ~/.sec_mcp_cache/_market/overview.json (+ Supabase financial_cache when
configured). Coverage is always reported — partial data is labeled, never
silently presented as complete.

Run directly:  python -m sec_mcp.ingest_indices
Or POST /api/ingest/indices (chat_app) triggers run_ingest() for the cron.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sec_mcp import polygon_client
from sec_mcp.classify import SECTOR_ETF, classify
from sec_mcp.surface.company_search import _load_sp500
from sec_mcp.surface.indices import MARKET_CACHE

log = logging.getLogger(__name__)

# Shares-outstanding cache — shares change rarely, so we persist them and only
# refetch when missing/stale, keeping the per-run reference-call count low.
_SHARES_CACHE = Path.home() / ".sec_mcp_cache" / "_market" / "shares.json"
_SHARES_TTL = 14 * 86400  # 2 weeks

DEFAULT_LOOKBACK = 60  # trading-day window for breadth SMAs / new-highs


def _trading_dates(lookback: int) -> list[str]:
    """Most-recent `lookback` weekday ISO dates (holidays tolerated as empty pulls)."""
    dates: list[str] = []
    d = datetime.now(timezone.utc).date()
    # Reach back over enough calendar days to net `lookback` weekdays.
    for _ in range(lookback * 2):
        if len(dates) >= lookback:
            break
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(dates))  # oldest → newest


def _load_shares() -> dict[str, dict]:
    if not _SHARES_CACHE.exists():
        return {}
    try:
        return json.loads(_SHARES_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_shares(cache: dict[str, dict]) -> None:
    try:
        _SHARES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _SHARES_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as exc:
        log.warning("shares cache write failed: %s", exc)


def _shares_and_sector(ticker: str, cache: dict[str, dict]) -> tuple[float | None, str | None]:
    """Shares outstanding + canonical GICS sector for a ticker.

    Only the expensive Polygon data (shares + SIC) is cached; the sector is
    recomputed fresh from the canonical classifier each run, so a stale disk
    cache can't leave old coarse sector names mixed with GICS names.
    """
    ent = cache.get(ticker)
    if ent and (time.time() - ent.get("fetched_at", 0) < _SHARES_TTL) and "sic_code" in ent:
        shares, sic = ent.get("shares"), ent.get("sic_code")
    else:
        details = polygon_client.get_ticker_details(ticker)
        shares = sic = None
        if details:
            shares = details.get("share_class_shares_outstanding") or details.get("weighted_shares_outstanding")
            sic = details.get("sic_code")
        cache[ticker] = {"shares": shares, "sic_code": sic, "fetched_at": time.time()}
    cls = classify(sic_code=sic, ticker=ticker)
    sector = None if cls.sector == "Other" else cls.sector
    return shares, sector


def _sma(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_ingest(lookback: int = DEFAULT_LOOKBACK, max_constituents: int | None = None) -> dict:
    """Compute breadth + sector rollup and write the market cache blob.

    Returns the blob (also persisted). Gracefully partial: whatever couldn't be
    fetched is reflected in the coverage counts rather than failing the run.
    """
    t0 = time.time()
    constituents = sorted(_load_sp500())
    if max_constituents:
        constituents = constituents[:max_constituents]
    if not constituents:
        raise RuntimeError("S&P 500 constituent set unavailable — cannot ingest.")

    # ── one grouped-daily pull per trading day → full-market history ─────────
    dates = _trading_dates(lookback)
    # Per-ticker close series (oldest → newest), constituents only.
    series: dict[str, list[float]] = {t: [] for t in constituents}
    covered_days = 0
    for d in dates:
        grouped = polygon_client.get_grouped_daily(d)
        if not grouped:
            continue  # holiday / not-yet-settled — skip, don't fail
        covered_days += 1
        for tk in constituents:
            row = grouped.get(tk)
            if row and row.get("c") is not None:
                series[tk].append(row["c"])

    # ── breadth ──────────────────────────────────────────────────────────────
    advancers = decliners = new_highs = new_lows = above_50dma = with_signal = 0
    for closes in series.values():
        if len(closes) < 2:
            continue
        with_signal += 1
        last, prev = closes[-1], closes[-2]
        if last > prev:
            advancers += 1
        elif last < prev:
            decliners += 1
        window_hi, window_lo = max(closes), min(closes)
        if last >= window_hi:
            new_highs += 1
        if last <= window_lo:
            new_lows += 1
        sma50 = _sma(closes[-50:])
        if sma50 and last > sma50:
            above_50dma += 1

    breadth = {
        "advancers": advancers,
        "decliners": decliners,
        "advDecRatio": round(advancers / decliners, 2) if decliners else None,
        "newHighs": new_highs,
        "newLows": new_lows,
        "pctAbove50dma": round(100 * above_50dma / with_signal, 1) if with_signal else None,
        "window": f"{covered_days}D",
        "coverage": round(with_signal / len(constituents), 3),
    } if with_signal else None

    # ── cap-weighted sector rollup ────────────────────────────────────────────
    shares_cache = _load_shares()
    sector_cap: dict[str, float] = {}
    sector_count: dict[str, int] = {}
    sector_capchg: dict[str, float] = {}  # Σ(cap × dayChangePct)
    sector_members: dict[str, list[tuple[float, str]]] = {}  # (cap, ticker) per sector
    total_cap = 0.0
    weighted = 0
    for tk, closes in series.items():
        if not closes:
            continue
        shares, sector = _shares_and_sector(tk, shares_cache)
        if not shares or not sector:
            continue
        cap = closes[-1] * shares
        day_chg = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 and closes[-2] else 0.0
        sector_cap[sector] = sector_cap.get(sector, 0.0) + cap
        sector_count[sector] = sector_count.get(sector, 0) + 1
        sector_capchg[sector] = sector_capchg.get(sector, 0.0) + cap * day_chg
        sector_members.setdefault(sector, []).append((cap, tk))
        total_cap += cap
        weighted += 1
    _save_shares(shares_cache)

    sectors = None
    if total_cap > 0:
        sectors = sorted(
            (
                {
                    "sector": s,
                    "etf": SECTOR_ETF.get(s),                # SPDR tile ETF (XLK, …)
                    "weightPct": round(100 * cap / total_cap, 2),
                    "count": sector_count[s],
                    "avgChangePct": round(sector_capchg[s] / cap, 2) if cap else None,
                    # Top constituents by market cap — feeds the tile tap-through
                    "constituents": [
                        t for _, t in sorted(sector_members.get(s, []), reverse=True)[:15]
                    ],
                }
                for s, cap in sector_cap.items()
            ),
            key=lambda r: r["weightPct"],
            reverse=True,
        )

    now = datetime.now(timezone.utc)
    blob = {
        "breadth": breadth,
        "sectors": sectors,
        "coverage": {
            "constituents": len(constituents),
            "priced": weighted,
            "tradingDays": covered_days,
        },
        "cached_at": time.time(),
        "cached_at_iso": now.isoformat(timespec="seconds"),
        "elapsedMs": int((time.time() - t0) * 1000),
    }

    # ── persist: disk (source of truth for get_market_overview) + Supabase ───
    try:
        MARKET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MARKET_CACHE.write_text(json.dumps(blob), encoding="utf-8")
    except Exception as exc:
        log.warning("market cache write failed: %s", exc)
    try:
        from sec_mcp import supabase_cache
        supabase_cache.set_cached("__MARKET__", "market_overview", blob, ttl=6 * 3600)
    except Exception as exc:
        log.debug("supabase market cache write skipped: %s", exc)

    log.info(
        "index ingest done: %d/%d priced, breadth over %dD, %dms",
        weighted, len(constituents), covered_days, blob["elapsedMs"],
    )
    return blob


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_ingest()
    print(json.dumps({k: v for k, v in result.items() if k != "sectors"}, indent=2))
    print("sectors:", json.dumps(result.get("sectors"), indent=2))
