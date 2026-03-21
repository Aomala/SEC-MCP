"""Ingest ALL SEC EDGAR public filers into Supabase company_directory.

Phase 1: Download company tickers list (10K+ companies) → save to company_directory
Phase 2: For each company, check if they file 10-K/10-Q/20-F → update flags
Phase 3: Extract financials for top companies by market cap → cache in financial_cache

Run:
  python -m sec_mcp.ingest_all --phase 1        # Download all tickers (fast, 30s)
  python -m sec_mcp.ingest_all --phase 2        # Check filing types (slow, ~2hrs)
  python -m sec_mcp.ingest_all --phase 3        # Extract top N financials
  python -m sec_mcp.ingest_all --phase 3 --top 500  # Extract top 500 by revenue
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

import requests

from sec_mcp.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

_HEADERS = {}


def _get_headers():
    global _HEADERS
    if not _HEADERS:
        cfg = get_config()
        _HEADERS = {"User-Agent": cfg.edgar_identity, "Accept": "application/json"}
    return _HEADERS


def _get_sb():
    """Get Supabase client."""
    from sec_mcp.config import get_config
    cfg = get_config()
    if not cfg.supabase_url or not cfg.supabase_key:
        log.error("SUPABASE_URL and SUPABASE_KEY required")
        sys.exit(1)
    from supabase import create_client
    return create_client(cfg.supabase_url, cfg.supabase_key)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: Download all tickers from SEC
# ═══════════════════════════════════════════════════════════════════════════

def phase1_download_tickers():
    """Download all company tickers from SEC EDGAR and save to Supabase."""
    log.info("Phase 1: Downloading ALL SEC EDGAR company tickers...")

    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_get_headers(),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    companies = []
    for entry in data.values():
        cik = str(entry["cik_str"])
        ticker = entry.get("ticker", "")
        name = entry.get("title", "")
        companies.append({
            "cik": cik,
            "ticker": ticker,
            "name": name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    log.info("Downloaded %d companies from SEC EDGAR", len(companies))

    # Batch upsert to Supabase (chunks of 500)
    sb = _get_sb()
    chunk_size = 500
    for i in range(0, len(companies), chunk_size):
        chunk = companies[i:i + chunk_size]
        try:
            sb.table("company_directory").upsert(chunk, on_conflict="cik").execute()
            log.info("  Upserted %d-%d / %d", i + 1, min(i + chunk_size, len(companies)), len(companies))
        except Exception as exc:
            log.error("  Upsert failed at %d: %s", i, exc)

    log.info("Phase 1 complete: %d companies saved to company_directory", len(companies))
    return len(companies)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: Check filing types for each company
# ═══════════════════════════════════════════════════════════════════════════

def phase2_check_filings(limit: int = 0):
    """Check what filing types each company has (10-K, 10-Q, 20-F).

    Uses the SEC submissions API to check recent filings.
    Rate limited to ~5 req/sec to respect SEC limits.
    """
    sb = _get_sb()

    # Get companies that haven't been checked yet
    query = sb.table("company_directory").select("cik, ticker, name").eq("has_10k", False).eq("has_20f", False)
    if limit > 0:
        query = query.limit(limit)
    result = query.execute()
    companies = result.data or []

    log.info("Phase 2: Checking filing types for %d companies...", len(companies))

    checked = 0
    has_filings = 0

    for i, company in enumerate(companies):
        cik = company["cik"]
        ticker = company.get("ticker", "?")

        try:
            cik_padded = cik.zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            r = requests.get(url, headers=_get_headers(), timeout=10)

            if r.status_code != 200:
                time.sleep(0.2)
                continue

            sub = r.json()
            recent = sub.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])

            form_set = set(forms)
            has_10k = "10-K" in form_set or "10-K/A" in form_set
            has_10q = "10-Q" in form_set or "10-Q/A" in form_set
            has_20f = "20-F" in form_set or "20-F/A" in form_set

            # Get latest filing date
            latest_date = dates[0] if dates else None

            # Get SIC code and other metadata
            sic = sub.get("sic", "")
            sic_desc = sub.get("sicDescription", "")
            state = sub.get("stateOfIncorporation", "")
            exchanges = sub.get("exchanges", [])
            exchange = exchanges[0] if exchanges else ""
            category = sub.get("category", "")

            update = {
                "has_10k": has_10k,
                "has_10q": has_10q,
                "has_20f": has_20f,
                "latest_filing_date": latest_date,
                "sic_code": sic,
                "sic_description": sic_desc,
                "state": state,
                "exchange": exchange,
                "category": category,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            sb.table("company_directory").update(update).eq("cik", cik).execute()

            if has_10k or has_20f:
                has_filings += 1

            checked += 1
            if checked % 100 == 0:
                log.info("  Checked %d/%d — %d have 10-K/20-F", checked, len(companies), has_filings)

        except Exception as exc:
            log.debug("  %s (%s): check failed — %s", ticker, cik, exc)

        # Rate limit
        time.sleep(0.15)

    log.info("Phase 2 complete: checked %d, %d have annual filings (10-K/20-F)", checked, has_filings)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: Extract financials for top companies
# ═══════════════════════════════════════════════════════════════════════════

def phase3_extract(top: int = 500):
    """Extract financials for top companies and cache in Supabase."""
    from sec_mcp.financials import extract_financials
    from sec_mcp import supabase_cache

    sb = _get_sb()

    # Get companies with 10-K/20-F that aren't cached yet
    result = (
        sb.table("company_directory")
        .select("cik, ticker, name")
        .or_("has_10k.eq.true,has_20f.eq.true")
        .eq("cached", False)
        .limit(top)
        .execute()
    )
    companies = result.data or []

    log.info("Phase 3: Extracting financials for %d companies...", len(companies))

    extracted = 0
    failed = 0

    for i, company in enumerate(companies):
        ticker = company.get("ticker", "")
        if not ticker:
            continue

        prefix = f"[{i+1}/{len(companies)}]"

        # Skip if already in financial_cache
        cached = supabase_cache.get_cached(ticker, "financials", "10-K|latest")
        if cached and isinstance(cached, dict) and cached.get("metrics"):
            log.debug("%s %s — already cached", prefix, ticker)
            sb.table("company_directory").update({"cached": True}).eq("cik", company["cik"]).execute()
            continue

        log.info("%s %s (%s) — extracting...", prefix, ticker, company.get("name", "")[:30])

        try:
            data = extract_financials(ticker, include_statements=True, include_segments=True)
            if data and data.get("metrics"):
                supabase_cache.set_cached(ticker, "financials", data, "10-K|latest")

                # Update company_directory with revenue/net_income for screener
                m = data.get("metrics", {})
                sb.table("company_directory").update({
                    "cached": True,
                    "revenue": m.get("revenue"),
                    "net_income": m.get("net_income"),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("cik", company["cik"]).execute()

                log.info("%s %s — OK (rev=%s)", prefix, ticker,
                         f"${m.get('revenue', 0)/1e9:.1f}B" if m.get("revenue") and m["revenue"] > 1e9
                         else f"${m.get('revenue', 0)/1e6:.0f}M" if m.get("revenue") else "N/A")
                extracted += 1
            else:
                failed += 1
        except Exception as exc:
            log.warning("%s %s — failed: %s", prefix, ticker, exc)
            failed += 1

        time.sleep(0.3)

    log.info("Phase 3 complete: extracted %d, failed %d", extracted, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest all SEC EDGAR companies")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="1=download tickers, 2=check filings, 3=extract financials")
    parser.add_argument("--top", type=int, default=500, help="Phase 3: how many to extract")
    parser.add_argument("--limit", type=int, default=0, help="Phase 2: limit companies to check")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_download_tickers()
    elif args.phase == 2:
        phase2_check_filings(limit=args.limit)
    elif args.phase == 3:
        phase3_extract(top=args.top)
