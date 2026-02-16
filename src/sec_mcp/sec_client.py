"""Direct SEC EDGAR API client — replaces edgartools entirely.

Uses only public SEC endpoints (no API key needed, just User-Agent header):
  - company_tickers.json  — ticker→CIK resolution
  - submissions/CIK{cik}.json  — company info + filing list
  - api/xbrl/companyfacts/CIK{cik}.json  — ALL XBRL facts for a company
  - api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json  — single concept
  - Archives/edgar/data/{cik}/{acc}/{doc}  — filing document HTML/text

Rate limited to 8 req/sec per SEC guidelines.
In-memory caching for frequently-accessed data (tickers list, company facts).
"""

from __future__ import annotations

import logging
import re
import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import requests

from sec_mcp.models import CompanyInfo, FilingMetadata

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# SEC EDGAR public API base URLs
SEC_BASE = "https://www.sec.gov"
DATA_BASE = "https://data.sec.gov"
TICKERS_URL = f"{SEC_BASE}/files/company_tickers.json"
TICKERS_EXCHANGE_URL = f"{SEC_BASE}/files/company_tickers_exchange.json"
SUBMISSIONS_URL = f"{DATA_BASE}/submissions/CIK{{cik}}.json"
COMPANY_FACTS_URL = f"{DATA_BASE}/api/xbrl/companyfacts/CIK{{cik}}.json"
COMPANY_CONCEPT_URL = f"{DATA_BASE}/api/xbrl/companyconcept/CIK{{cik}}/{{taxonomy}}/{{tag}}.json"

# SEC requires a descriptive User-Agent with contact email
DEFAULT_USER_AGENT = "SEC-MCP sec-mcp@example.com"

# Rate limiting: SEC allows up to 10 req/s; we use 8 to stay safe
MAX_REQUESTS_PER_SECOND = 8.0
MIN_REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND

# Cache TTLs (seconds)
TICKERS_CACHE_TTL = 1800    # 30 minutes for ticker→CIK mapping
FACTS_CACHE_TTL = 300       # 5 minutes for XBRL company facts
SUBMISSIONS_CACHE_TTL = 120  # 2 minutes for submissions/filings list


# ═══════════════════════════════════════════════════════════════════════════
#  Foreign Private Issuer (FPI) form type mapping
#  FPIs file 20-F (annual) and 6-K (interim) instead of 10-K/10-Q.
#  Examples: ASML, BABA, TSM, SAP, NVO, SHOP, SE, MELI, JD, PDD, SPOT
# ═══════════════════════════════════════════════════════════════════════════

# Map US form types to their FPI equivalents
_ANNUAL_FORMS = ("10-K", "20-F", "10-K/A", "20-F/A")
_QUARTERLY_FORMS = ("10-Q", "6-K", "10-Q/A", "6-K/A")
_ALL_PERIODIC_FORMS = _ANNUAL_FORMS + _QUARTERLY_FORMS


def get_form_alternatives(form_type: str) -> list[str]:
    """Return ordered list of form types to try, including FPI alternatives.

    Given "10-K" → returns ["10-K", "20-F"] (tries US annual, then FPI annual)
    Given "10-Q" → returns ["10-Q", "6-K"] (tries US quarterly, then FPI quarterly)
    Given "20-F" → returns ["20-F", "10-K"] (tries FPI annual, then US annual)
    Given "6-K"  → returns ["6-K", "10-Q"] (tries FPI quarterly, then US quarterly)
    """
    mapping = {
        "10-K": ["10-K", "20-F"],
        "20-F": ["20-F", "10-K"],
        "10-Q": ["10-Q", "6-K"],
        "6-K":  ["6-K", "10-Q"],
    }
    return mapping.get(form_type, [form_type])


def is_annual_form(form_type: str) -> bool:
    """Check if a form type is an annual filing (10-K or 20-F)."""
    return form_type in _ANNUAL_FORMS


def is_quarterly_form(form_type: str) -> bool:
    """Check if a form type is a quarterly/interim filing (10-Q or 6-K)."""
    return form_type in _QUARTERLY_FORMS


# ═══════════════════════════════════════════════════════════════════════════
#  Cache helper
# ═══════════════════════════════════════════════════════════════════════════

class _CacheEntry:
    """Simple timestamped cache entry."""
    __slots__ = ("data", "timestamp")

    def __init__(self, data: Any):
        self.data = data
        self.timestamp = time.time()

    def expired(self, ttl: float) -> bool:
        return (time.time() - self.timestamp) > ttl


# ═══════════════════════════════════════════════════════════════════════════
#  SEC EDGAR Client
# ═══════════════════════════════════════════════════════════════════════════

class SECClient:
    """Direct HTTP client for SEC EDGAR public APIs.

    Replaces edgartools with pure requests-based calls.
    Thread-safe with rate limiting and in-memory caching.
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        # Rate limiter state
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()
        # Caches
        self._tickers_cache: _CacheEntry | None = None
        self._facts_cache: dict[str, _CacheEntry] = {}
        self._submissions_cache: dict[str, _CacheEntry] = {}

    # ── Rate-limited HTTP request ─────────────────────────────────────

    def _request(self, url: str, timeout: int = 30, retries: int = 2) -> requests.Response:
        """Make a GET request with rate limiting and automatic retry.

        Enforces MIN_REQUEST_INTERVAL between calls to stay within
        SEC's 10 req/s limit. Thread-safe via lock.
        Retries on 429 (rate-limit), 500/502/503/504 (server errors),
        and connection errors.
        """
        last_exc: Exception | None = None
        for attempt in range(1 + retries):
            with self._rate_lock:
                elapsed = time.time() - self._last_request_time
                if elapsed < MIN_REQUEST_INTERVAL:
                    time.sleep(MIN_REQUEST_INTERVAL - elapsed)
                self._last_request_time = time.time()

            try:
                resp = requests.get(url, headers=self.headers, timeout=timeout)
                if resp.status_code == 429:
                    wait = min(2 ** attempt, 10)
                    log.warning("SEC rate-limited (429), retrying in %ds…", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code in (500, 502, 503, 504) and attempt < retries:
                    wait = min(2 ** attempt, 8)
                    log.warning("SEC %d error, retrying in %ds…", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if attempt < retries:
                    wait = min(2 ** attempt, 8)
                    log.warning("Connection error, retrying in %ds: %s", wait, exc)
                    time.sleep(wait)
                    continue
                raise
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                if attempt < retries:
                    log.warning("Request timeout, retrying: %s", exc)
                    continue
                raise

        # If we exhausted retries on a non-exception path (e.g. 429 loop)
        if last_exc:
            raise last_exc
        raise requests.exceptions.ConnectionError(f"Failed after {retries + 1} attempts: {url}")

    def _request_json(self, url: str, timeout: int = 30) -> dict:
        """GET request that returns parsed JSON."""
        return self._request(url, timeout=timeout).json()

    # ── Ticker → CIK resolution ──────────────────────────────────────

    def _get_tickers_map(self) -> dict[str, dict]:
        """Load and cache SEC company tickers with exchange data.

        Primary source: company_tickers_exchange.json (has exchange info).
        Fallback: company_tickers.json (basic ticker→CIK mapping).

        Returns a dict keyed by uppercase ticker → {cik_str, ticker, title, exchange}.
        Also includes entries keyed by CIK string for reverse lookup.
        """
        if self._tickers_cache and not self._tickers_cache.expired(TICKERS_CACHE_TTL):
            return self._tickers_cache.data

        by_ticker: dict[str, dict] = {}
        by_cik: dict[str, dict] = {}

        # Try the exchange endpoint first (richer data)
        try:
            log.info("Fetching SEC company_tickers_exchange.json (cached for %ds)", TICKERS_CACHE_TTL)
            raw = self._request_json(TICKERS_EXCHANGE_URL)
            # Format: {"fields":["cik","name","ticker","exchange"],"data":[[cik,name,ticker,exchange],...]}
            fields = raw.get("fields", [])
            data_rows = raw.get("data", [])
            cik_idx = fields.index("cik") if "cik" in fields else 0
            name_idx = fields.index("name") if "name" in fields else 1
            ticker_idx = fields.index("ticker") if "ticker" in fields else 2
            exchange_idx = fields.index("exchange") if "exchange" in fields else 3

            for row in data_rows:
                if len(row) < 3:
                    continue
                ticker = str(row[ticker_idx] if ticker_idx < len(row) else "").upper()
                cik = str(row[cik_idx] if cik_idx < len(row) else "")
                name = str(row[name_idx] if name_idx < len(row) else "")
                exchange = str(row[exchange_idx] if exchange_idx < len(row) and row[exchange_idx] else "")

                record = {
                    "cik_str": cik,
                    "ticker": ticker,
                    "title": name,
                    "exchange": exchange,
                }
                if ticker:
                    by_ticker[ticker] = record
                if cik:
                    by_cik[cik] = record

            log.info(
                "Loaded %d tickers from company_tickers_exchange.json",
                len(by_ticker),
            )
        except Exception as exc:
            log.warning(
                "Failed to fetch company_tickers_exchange.json (%s), "
                "falling back to company_tickers.json",
                exc,
            )
            # Fallback to original endpoint
            try:
                raw = self._request_json(TICKERS_URL)
                for entry in raw.values():
                    ticker = str(entry.get("ticker", "")).upper()
                    cik = str(entry.get("cik_str", ""))
                    record = {
                        "cik_str": cik,
                        "ticker": ticker,
                        "title": entry.get("title", ""),
                        "exchange": "",
                    }
                    if ticker:
                        by_ticker[ticker] = record
                    if cik:
                        by_cik[cik] = record
            except Exception as exc2:
                log.error("Failed to fetch ANY tickers list: %s", exc2)

        merged = {**by_ticker, **{f"CIK:{k}": v for k, v in by_cik.items()}}
        self._tickers_cache = _CacheEntry(merged)
        return merged

    def resolve_cik(self, ticker_or_cik: str) -> str:
        """Resolve a ticker symbol or CIK number to a zero-padded CIK string.

        Accepts: "AAPL", "320193", "0000320193"
        Returns: "0000320193" (10-digit zero-padded)

        Falls back to trying the submissions endpoint directly for CIK-like
        inputs that aren't in the tickers file (some foreign filers, etc.).
        """
        clean = ticker_or_cik.strip().upper()

        # Strip any leading "CIK" prefix
        if clean.startswith("CIK"):
            clean = clean[3:].lstrip("0") or "0"

        # If it's already a numeric CIK
        if clean.isdigit():
            return clean.zfill(10)

        # Look up in tickers map
        try:
            tickers_map = self._get_tickers_map()
        except Exception as exc:
            log.warning("Could not load tickers map: %s", exc)
            tickers_map = {}

        entry = tickers_map.get(clean)
        if entry:
            return str(entry["cik_str"]).zfill(10)

        # Try removing common suffixes and re-checking
        for suffix in ("-A", "-B", ".A", ".B", "/A", "/B"):
            alt = clean.rstrip(suffix)
            if alt != clean:
                entry = tickers_map.get(alt)
                if entry:
                    return str(entry["cik_str"]).zfill(10)

        # Last resort: try partial ticker match (e.g. "BRK" → "BRK-B")
        for key, val in tickers_map.items():
            if key.startswith("CIK:"):
                continue
            if key.startswith(clean) and len(key) - len(clean) <= 2:
                log.info("Fuzzy ticker match: '%s' → '%s'", clean, key)
                return str(val["cik_str"]).zfill(10)

        raise ValueError(
            f"Could not resolve '{ticker_or_cik}' to a CIK number. "
            f"Try using a ticker symbol (e.g., AAPL) or CIK number."
        )

    # ── Company info ──────────────────────────────────────────────────

    def _get_submissions(self, cik: str) -> dict:
        """Fetch and cache the submissions JSON for a company.

        The submissions endpoint returns company metadata + all recent filings.
        This is the richest single endpoint for company data.
        Returns empty dict on any network/HTTP error instead of crashing.
        """
        cik_padded = cik.zfill(10)

        cached = self._submissions_cache.get(cik_padded)
        if cached and not cached.expired(SUBMISSIONS_CACHE_TTL):
            return cached.data

        url = SUBMISSIONS_URL.format(cik=cik_padded)
        try:
            data = self._request_json(url)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            log.warning("Submissions fetch failed for CIK %s (HTTP %d)", cik_padded, status)
            return {}
        except Exception as exc:
            log.warning("Submissions fetch failed for CIK %s: %s", cik_padded, exc)
            return {}

        self._submissions_cache[cik_padded] = _CacheEntry(data)
        return data

    def get_company_info(self, ticker_or_cik: str) -> CompanyInfo:
        """Get company metadata (name, CIK, ticker, SIC, industry).

        Uses the submissions endpoint which has the most complete company data.
        """
        cik = self.resolve_cik(ticker_or_cik)
        data = self._get_submissions(cik)

        # Extract ticker from submissions response
        tickers_list = data.get("tickers") or []
        ticker = tickers_list[0] if tickers_list else None

        # If we can't get ticker from submissions, check our tickers map
        if not ticker:
            tickers_map = self._get_tickers_map()
            cik_entry = tickers_map.get(f"CIK:{str(int(cik))}")
            if cik_entry:
                ticker = cik_entry.get("ticker")

        # Extract company website(s) from submissions
        website = ""
        websites_list = data.get("website") or data.get("websites") or []
        if isinstance(websites_list, str):
            website = websites_list
        elif isinstance(websites_list, list) and websites_list:
            website = websites_list[0]

        return CompanyInfo(
            name=data.get("name", ""),
            cik=int(data.get("cik", cik)),
            ticker=ticker,
            industry=data.get("sicDescription"),
            sic_code=data.get("sic"),
            website=website,
        )

    def search_companies(self, query: str, limit: int = 10) -> list[CompanyInfo]:
        """Search for companies by ticker or name.

        Searches the SEC company_tickers_exchange.json (all public filers).
        Returns exact ticker matches first, then partial name matches.
        Includes exchange info (NYSE, Nasdaq, OTC, etc.).
        """
        tickers_map = self._get_tickers_map()
        query_upper = query.strip().upper()
        results: list[CompanyInfo] = []
        seen_ciks: set[int] = set()

        # Exact ticker match first
        entry = tickers_map.get(query_upper)
        if entry:
            cik_int = int(entry["cik_str"])
            results.append(CompanyInfo(
                name=entry["title"],
                cik=cik_int,
                ticker=entry["ticker"],
                exchange=entry.get("exchange", ""),
            ))
            seen_ciks.add(cik_int)

        # Partial name/ticker search — prefer exact ticker starts, then name matches
        if len(results) < limit:
            starts_with: list[CompanyInfo] = []
            contains: list[CompanyInfo] = []
            for key, entry in tickers_map.items():
                if key.startswith("CIK:"):
                    continue
                cik_int = int(entry["cik_str"])
                if cik_int in seen_ciks:
                    continue
                title = entry.get("title", "").upper()
                ticker = entry.get("ticker", "").upper()
                if ticker.startswith(query_upper):
                    starts_with.append(CompanyInfo(
                        name=entry["title"],
                        cik=cik_int,
                        ticker=entry["ticker"],
                        exchange=entry.get("exchange", ""),
                    ))
                    seen_ciks.add(cik_int)
                elif query_upper in title or query_upper in ticker:
                    contains.append(CompanyInfo(
                        name=entry["title"],
                        cik=cik_int,
                        ticker=entry["ticker"],
                        exchange=entry.get("exchange", ""),
                    ))
                    seen_ciks.add(cik_int)
                if len(starts_with) + len(contains) + len(results) >= limit * 3:
                    break

            # Prioritise: exact match → ticker starts with → name contains
            for r in starts_with:
                if len(results) >= limit:
                    break
                results.append(r)
            for r in contains:
                if len(results) >= limit:
                    break
                results.append(r)

        return results

    # ── Filing list ───────────────────────────────────────────────────

    def get_filings(
        self,
        ticker_or_cik: str,
        form_type: str | None = None,
        limit: int = 40,
    ) -> list[FilingMetadata]:
        """Get recent filings for a company.

        Uses the submissions endpoint which always has the latest filings
        (updated in real-time by SEC). This is the key advantage over
        edgartools which sometimes missed recent filings.

        Returns filings sorted by date descending (most recent first).
        """
        cik = self.resolve_cik(ticker_or_cik)
        data = self._get_submissions(cik)

        # The submissions response has filings in columnar format
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        # Build list of filings from columnar data
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        results: list[FilingMetadata] = []
        for i in range(len(accessions)):
            form = forms[i] if i < len(forms) else ""

            # Filter by form type if specified
            if form_type and form != form_type:
                continue

            desc = descriptions[i] if i < len(descriptions) else None
            if desc and str(desc).lower() in ("nan", "none", ""):
                desc = None

            results.append(FilingMetadata(
                accession_number=accessions[i],
                form_type=form,
                filing_date=dates[i] if i < len(dates) else "",
                description=str(desc) if desc else None,
            ))

            if len(results) >= limit:
                break

        return results

    def get_filings_smart(
        self,
        ticker_or_cik: str,
        form_type: str = "10-K",
        limit: int = 40,
    ) -> list[FilingMetadata]:
        """Get filings with automatic foreign private issuer (FPI) fallback.

        If 10-K returns nothing, tries 20-F. If 10-Q returns nothing, tries 6-K.
        This is the preferred method for all UI/tool code.
        """
        for ft in get_form_alternatives(form_type):
            results = self.get_filings(ticker_or_cik, form_type=ft, limit=limit)
            if results:
                return results
        return []

    # ── XBRL Company Facts (all financial data) ──────────────────────

    def get_company_facts(self, ticker_or_cik: str) -> dict:
        """Fetch ALL XBRL facts for a company.

        This is the core data source for financial extraction. Returns every
        XBRL concept ever reported by the company across all filings.

        The response can be large (10-50MB for big filers), so we cache it.

        Structure: {
            "cik": 320193,
            "entityName": "Apple Inc",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenues",
                        "units": {
                            "USD": [
                                {"start": "2023-10-01", "end": "2024-09-28",
                                 "filed": "2024-11-01", "form": "10-K",
                                 "accn": "0000320193-24-000123", "value": 391035000000, ...},
                                ...
                            ]
                        }
                    },
                    ...
                }
            }
        }
        """
        cik = self.resolve_cik(ticker_or_cik)
        cik_padded = cik.zfill(10)

        # Check cache
        cached = self._facts_cache.get(cik_padded)
        if cached and not cached.expired(FACTS_CACHE_TTL):
            return cached.data

        url = COMPANY_FACTS_URL.format(cik=cik_padded)
        log.info("Fetching XBRL companyfacts for CIK %s", cik_padded)
        try:
            data = self._request_json(url, timeout=60)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 404:
                log.warning(
                    "No XBRL companyfacts for CIK %s (404). "
                    "This company may not file in XBRL format.",
                    cik_padded,
                )
                return {}
            if status in (403, 429, 500, 502, 503, 504):
                log.warning(
                    "SEC returned %d for CIK %s companyfacts — returning empty.",
                    status, cik_padded,
                )
                return {}
            raise
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            log.warning(
                "Network error fetching companyfacts for CIK %s: %s",
                cik_padded, exc,
            )
            return {}
        except Exception as exc:
            log.warning(
                "Unexpected error fetching companyfacts for CIK %s: %s",
                cik_padded, exc,
            )
            return {}

        self._facts_cache[cik_padded] = _CacheEntry(data)
        return data

    def get_facts_dataframe(
        self,
        ticker_or_cik: str,
        form_type: str | None = None,
        accession: str | None = None,
        taxonomy: str = "us-gaap",
    ) -> pd.DataFrame:
        """Convert XBRL company facts into a flat DataFrame.

        This replaces edgartools' `filing.xbrl().facts.to_dataframe()`.

        Each row is one fact (one concept, one period, one value).
        Columns: concept, label, value, start, end, filed, form, accn, fy, units, taxonomy

        Aggressive fallback: if the requested taxonomy (default us-gaap) is empty,
        automatically tries ifrs-full and then ALL available taxonomies. This ensures
        IFRS-reporting companies like BABA, ASML, SAP, etc. always return data.

        Args:
            form_type: Filter to specific form (e.g., "10-K")
            accession: Filter to specific filing accession number
            taxonomy: XBRL taxonomy to try first (default: "us-gaap")
        """
        facts_data = self.get_company_facts(ticker_or_cik)
        all_facts = facts_data.get("facts", {})

        # Build ordered list of taxonomies to try: requested first, then ifrs, then rest
        taxonomies_to_try = [taxonomy]
        if "ifrs-full" not in taxonomies_to_try:
            taxonomies_to_try.append("ifrs-full")
        for tax_key in all_facts:
            if tax_key not in taxonomies_to_try and tax_key != "dei":
                taxonomies_to_try.append(tax_key)
        # dei last (mostly entity info, not financial data)
        if "dei" not in taxonomies_to_try:
            taxonomies_to_try.append("dei")

        rows: list[dict] = []
        used_taxonomy = None

        for tax in taxonomies_to_try:
            taxonomy_data = all_facts.get(tax, {})
            if not taxonomy_data:
                continue

            tax_rows: list[dict] = []
            for concept_name, concept_data in taxonomy_data.items():
                label = concept_data.get("label", concept_name)
                for unit_name, unit_facts in concept_data.get("units", {}).items():
                    for fact in unit_facts:
                        if form_type and fact.get("form") != form_type:
                            continue
                        if accession and fact.get("accn") != accession:
                            continue
                        tax_rows.append({
                            "concept": concept_name,
                            "label": label,
                            "value": fact.get("val") if "val" in fact else fact.get("value"),
                            "start": fact.get("start"),
                            "end": fact.get("end"),
                            "filed": fact.get("filed"),
                            "form": fact.get("form"),
                            "accn": fact.get("accn"),
                            "fy": fact.get("fy"),
                            "fp": fact.get("fp"),
                            "units": unit_name,
                            "taxonomy": tax,
                        })

            if tax_rows:
                if not rows:
                    used_taxonomy = tax
                rows.extend(tax_rows)
                # If the primary taxonomy had data, don't merge others (avoid duplicates)
                if tax == taxonomy and len(tax_rows) > 10:
                    break

        if used_taxonomy and used_taxonomy != taxonomy:
            log.info(
                "Using %s taxonomy for %s (requested %s was empty)",
                used_taxonomy, ticker_or_cik, taxonomy,
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            df = df.sort_values("end", ascending=False)

        return df

    # ── Single concept history ────────────────────────────────────────

    def get_concept_history(
        self,
        ticker_or_cik: str,
        concept: str,
        taxonomy: str = "us-gaap",
    ) -> pd.DataFrame:
        """Get historical values for a single XBRL concept.

        Lighter-weight than get_company_facts when you only need one metric.
        Uses the companyconcept endpoint.
        """
        cik = self.resolve_cik(ticker_or_cik)
        cik_padded = cik.zfill(10)
        url = COMPANY_CONCEPT_URL.format(cik=cik_padded, taxonomy=taxonomy, tag=concept)

        try:
            data = self._request_json(url)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return pd.DataFrame()  # Concept not reported by this company
            raise

        rows: list[dict] = []
        for unit_name, unit_facts in data.get("units", {}).items():
            for fact in unit_facts:
                rows.append({
                    "concept": concept,
                    "value": fact.get("val") if "val" in fact else fact.get("value"),
                    "start": fact.get("start"),
                    "end": fact.get("end"),
                    "filed": fact.get("filed"),
                    "form": fact.get("form"),
                    "accn": fact.get("accn"),
                    "fy": fact.get("fy"),
                    "units": unit_name,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df = df.sort_values("end", ascending=False)
        return df

    # ── Filing document text ──────────────────────────────────────────

    def get_filing_document(
        self,
        ticker_or_cik: str,
        accession: str,
        section: str | None = None,
        max_length: int = 50000,
    ) -> str:
        """Fetch the primary document for a filing (HTML/text).

        Downloads the actual filing document from SEC Archives,
        strips HTML tags, and optionally extracts a specific section.

        Uses the line-level section segmenter (inspired by BERT4ItemSeg)
        for accurate section extraction that avoids TOC/balance-sheet bleed.
        """
        from sec_mcp.section_segmenter import extract_section as seg_extract

        cik = self.resolve_cik(ticker_or_cik)
        cik_raw = str(int(cik))
        cik_padded = cik.zfill(10)

        # Get the primary document filename from submissions
        submissions = self._get_submissions(cik_padded)
        recent = submissions.get("filings", {}).get("recent", {})
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        form_types = recent.get("form", [])

        # Find the primary document for this accession
        primary_doc = None
        filing_form = None
        for i, acc in enumerate(accessions):
            if acc == accession:
                primary_doc = primary_docs[i] if i < len(primary_docs) else None
                filing_form = form_types[i] if i < len(form_types) else None
                break

        if not primary_doc:
            log.warning("Filing %s not found or has no primary document", accession)
            return f"[Filing {accession} not found or has no primary document for this company.]"

        # Build the URL to the actual filing document
        acc_clean = accession.replace("-", "")
        doc_url = f"{SEC_BASE}/Archives/edgar/data/{cik_raw}/{acc_clean}/{primary_doc}"

        log.info("Fetching filing document: %s", doc_url)
        try:
            resp = self._request(doc_url, timeout=60)
            raw_html = resp.text
        except Exception as exc:
            log.warning("Failed to fetch filing document %s: %s", doc_url, exc)
            return f"[Could not fetch filing document: {exc}]"

        # Strip HTML tags to get plain text
        if primary_doc.endswith((".htm", ".html")):
            text = self._strip_html(raw_html)
        else:
            text = raw_html

        # Extract section if requested — use the new line-level segmenter
        if section:
            extracted = seg_extract(
                text, section,
                accession=accession,
                form_type=filing_form or "10-K",
                raw_html=raw_html if primary_doc.endswith((".htm", ".html")) else "",
            )
            if extracted and len(extracted.strip()) >= 200:
                return extracted[:max_length]

            # Fallback: try complete submission text file (often cleaner)
            try:
                txt_url = f"{SEC_BASE}/Archives/edgar/data/{cik_raw}/{acc_clean}/{accession}.txt"
                log.info("Trying complete submission text: %s", txt_url)
                txt_resp = self._request(txt_url, timeout=90)
                txt_content = txt_resp.text
                if txt_content.lstrip().startswith("<"):
                    txt_plain = self._strip_html(txt_content)
                else:
                    txt_plain = txt_content
                extracted2 = seg_extract(
                    txt_plain, section,
                    accession=accession + "_txt",
                    form_type=filing_form or "10-K",
                    raw_html=raw_html if primary_doc.endswith((".htm", ".html")) else "",
                )
                if extracted2 and len(extracted2.strip()) >= 200:
                    return extracted2[:max_length]
            except Exception as e:
                log.warning("Complete submission fallback failed: %s", e)

            log.warning("Section extraction for '%s' returned insufficient content", section)

        # No section requested — return full text (truncated at signatures)
        return self._truncate_at_signatures(text)[:max_length]

    def _truncate_at_signatures(self, text: str) -> str:
        """Truncate text before Power of Attorney / signature block."""
        markers = [
            r"\n\s*Power\s+of\s+Attorney\s*\n",
            r"\n\s*KNOW\s+ALL\s+PERSONS\s+BY\s+THESE\s+PRESENTS",
            r"Pursuant\s+to\s+the\s+requirements\s+of\s+the\s+Securities\s+Exchange\s+Act\s+of\s+1934[\s,]*this\s+report\s+has\s+been\s+signed",
        ]
        for pat in markers:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                return text[: m.start()].strip()
        return text

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and extract readable text using BeautifulSoup.

        Uses block-level element awareness to insert newlines only where
        appropriate, preventing word breaks across inline elements
        (e.g., <span>B</span><span>USINESS</span> → "BUSINESS" not "B\nUSINESS").

        Tables are converted to tab-separated text to preserve structure.
        """
        _BLOCK_TAGS = frozenset([
            "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "ol", "ul", "blockquote", "pre", "hr",
            "section", "article", "header", "footer", "nav",
            "tr", "table", "thead", "tbody", "tfoot",
            "dt", "dd", "dl", "figcaption", "figure",
        ])
        try:
            from bs4 import BeautifulSoup, NavigableString, Tag
            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, and ix:hidden blocks
            for tag in soup(["script", "style"]):
                tag.decompose()
            for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
                tag.decompose()

            # Convert tables to tab-separated text
            for table in soup.find_all("table"):
                rows_text = []
                for tr in table.find_all("tr"):
                    cells = []
                    for td in tr.find_all(["td", "th"]):
                        cell_text = td.get_text(strip=True)
                        if cell_text:
                            cells.append(cell_text)
                    if cells:
                        rows_text.append("\t".join(cells))
                if rows_text:
                    table.replace_with("\n".join(rows_text) + "\n")
                else:
                    table.decompose()

            # Walk the tree: insert "\n" before block elements, " " for inline
            parts: list[str] = []

            def _walk(node: Tag | NavigableString) -> None:
                if isinstance(node, NavigableString):
                    text = str(node)
                    if text.strip():
                        parts.append(text)
                    return
                if not isinstance(node, Tag):
                    return
                tag_name = node.name.lower() if node.name else ""
                is_block = tag_name in _BLOCK_TAGS
                if is_block:
                    parts.append("\n")
                for child in node.children:
                    _walk(child)
                if is_block:
                    parts.append("\n")

            _walk(soup)
            text = "".join(parts)
        except Exception:
            # Fallback to regex stripping if BeautifulSoup fails
            text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<(?:br|p|div|tr|li|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            text = text.replace("&nbsp;", " ").replace("&#160;", " ")
        # Normalize whitespace: collapse spaces within lines, limit blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Utility: build EDGAR URLs for citations ───────────────────────

    def build_filing_urls(
        self,
        cik: int | str,
        accession: str,
        form_type: str = "",
    ) -> dict[str, str]:
        """Build direct EDGAR URLs for a filing (for citation/linking)."""
        cik_raw = str(int(cik)) if str(cik).isdigit() else str(cik)
        cik_padded = cik_raw.zfill(10)
        acc_clean = accession.replace("-", "")

        return {
            "filing_index": (
                f"{SEC_BASE}/Archives/edgar/data/{cik_raw}/{acc_clean}/{accession}-index.htm"
            ),
            "company_page": (
                f"{SEC_BASE}/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik_padded}&type={form_type}"
                f"&dateb=&owner=include&count=40"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level singleton — shared across the app
# ═══════════════════════════════════════════════════════════════════════════

_client: SECClient | None = None


def get_sec_client() -> SECClient:
    """Get or create the shared SECClient singleton.

    Reads EDGAR_IDENTITY from config for the User-Agent header.
    """
    global _client
    if _client is None:
        from sec_mcp.config import get_config
        config = get_config()
        _client = SECClient(user_agent=config.edgar_identity)
    return _client
