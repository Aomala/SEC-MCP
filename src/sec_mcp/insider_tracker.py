"""Insider trading and institutional ownership tracker via SEC EDGAR.

Scrapes Form 4 (insider transactions) and Form 13F (institutional holdings)
data from SEC EDGAR public APIs. No API key required — just User-Agent.

Uses the same rate-limited SECClient singleton from sec_client.py.
In-memory cache with 30-minute TTL to avoid hammering SEC servers.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

import requests

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

SEC_BASE = "https://www.sec.gov"
DATA_BASE = "https://data.sec.gov"
EFTS_BASE = "https://efts.sec.gov/LATEST"

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 1800  # 30 minutes

# Form 4 transaction type codes → human-readable
_TRANSACTION_CODES: dict[str, str] = {
    "P": "Purchase",
    "S": "Sale",
    "A": "Grant/Award",
    "D": "Disposition (non-open-market)",
    "F": "Tax Payment (shares withheld)",
    "M": "Option Exercise",
    "C": "Conversion",
    "E": "Expiration",
    "G": "Gift",
    "H": "Expiration (short position)",
    "I": "Discretionary",
    "J": "Other",
    "K": "Equity Swap",
    "U": "Disposition pursuant to tender offer",
    "V": "Transaction voluntarily reported",
    "W": "Acquisition/disposition by will or laws of descent",
    "X": "Option Exercise",
    "Z": "Deposit/withdrawal from voting trust",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_headers() -> dict[str, str]:
    """Build request headers with SEC-required User-Agent."""
    config = get_config()
    return {
        "User-Agent": config.edgar_identity,
        "Accept-Encoding": "gzip, deflate",
    }


def _cached_get(cache_key: str, url: str, timeout: int = 10, parse_json: bool = True) -> Any:
    """Rate-limited, cached GET request to SEC EDGAR.

    Returns parsed JSON (default) or raw response text.
    Returns None on any error.
    """
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    try:
        # Respect SEC rate limit (~8 req/sec) with a small sleep
        time.sleep(0.125)
        resp = requests.get(url, headers=_get_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json() if parse_json else resp.text
        _CACHE[cache_key] = (time.time(), data)
        return data
    except Exception as exc:
        log.warning("SEC request failed: %s — %s", url, exc)
        return None


def _resolve_cik(ticker: str) -> str | None:
    """Resolve ticker to zero-padded CIK using the shared SECClient."""
    try:
        from sec_mcp.sec_client import get_sec_client
        client = get_sec_client()
        return client.resolve_cik(ticker)
    except Exception as exc:
        log.warning("Failed to resolve CIK for %s: %s", ticker, exc)
        return None


def _get_submissions(cik_padded: str) -> dict | None:
    """Fetch submissions JSON for a CIK (cached via SECClient)."""
    try:
        from sec_mcp.sec_client import get_sec_client
        client = get_sec_client()
        return client._get_submissions(cik_padded)
    except Exception as exc:
        log.warning("Failed to get submissions for CIK %s: %s", cik_padded, exc)
        return None


def _xml_text(element: ET.Element | None, path: str, default: str = "") -> str:
    """Safely extract text from an XML element at a given path."""
    if element is None:
        return default
    # Handle namespace-agnostic search
    node = element.find(path)
    if node is None:
        # Try without namespace
        tag = path.split("}")[-1] if "}" in path else path
        for child in element.iter():
            local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local == tag:
                return (child.text or "").strip()
        return default
    return (node.text or "").strip()


def _find_all_ns(root: ET.Element, local_name: str) -> list[ET.Element]:
    """Find all elements matching a local tag name, ignoring XML namespace."""
    results = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == local_name:
            results.append(elem)
    return results


def _safe_int(val: str | None) -> int | None:
    """Parse string to int, returning None on failure."""
    if not val:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str | None) -> float | None:
    """Parse string to float, returning None on failure."""
    if not val:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Form 4 — Insider Transactions
# ═══════════════════════════════════════════════════════════════════════════

def _parse_form4_xml(xml_text_content: str, accession: str, filing_date: str) -> list[dict]:
    """Parse a Form 4 XML document into transaction dicts."""
    transactions: list[dict] = []
    try:
        root = ET.fromstring(xml_text_content)
    except ET.ParseError as exc:
        log.debug("Failed to parse Form 4 XML for %s: %s", accession, exc)
        return []

    # Extract reporting owner info
    owners = _find_all_ns(root, "reportingOwner")
    owner_name = ""
    owner_title = ""

    if owners:
        owner = owners[0]
        # reportingOwnerId/rptOwnerName
        for name_tag in ("rptOwnerName",):
            name_elem = _find_all_ns(owner, name_tag)
            if name_elem:
                owner_name = (name_elem[0].text or "").strip()
                break
        # reportingOwnerRelationship/officerTitle
        for title_tag in ("officerTitle",):
            title_elems = _find_all_ns(owner, title_tag)
            if title_elems and title_elems[0].text:
                owner_title = title_elems[0].text.strip()
                break
        # If no officer title, check if director
        if not owner_title:
            is_director = _find_all_ns(owner, "isDirector")
            if is_director and (is_director[0].text or "").strip() == "1":
                owner_title = "Director"
            is_officer = _find_all_ns(owner, "isOfficer")
            if is_officer and (is_officer[0].text or "").strip() == "1" and not owner_title:
                owner_title = "Officer"
            is_ten_pct = _find_all_ns(owner, "isTenPercentOwner")
            if is_ten_pct and (is_ten_pct[0].text or "").strip() == "1" and not owner_title:
                owner_title = "10% Owner"

    # Parse non-derivative transactions
    for txn in _find_all_ns(root, "nonDerivativeTransaction"):
        code_elems = _find_all_ns(txn, "transactionCode")
        code = code_elems[0].text.strip() if code_elems and code_elems[0].text else ""
        txn_type = _TRANSACTION_CODES.get(code, code)

        date_elems = _find_all_ns(txn, "transactionDate")
        txn_date = ""
        for d in date_elems:
            val_elems = _find_all_ns(d, "value")
            if val_elems and val_elems[0].text:
                txn_date = val_elems[0].text.strip()
                break

        shares_elems = _find_all_ns(txn, "transactionShares")
        shares_val = None
        for s in shares_elems:
            val_elems = _find_all_ns(s, "value")
            if val_elems and val_elems[0].text:
                shares_val = _safe_int(val_elems[0].text.strip())
                break

        price_elems = _find_all_ns(txn, "transactionPricePerShare")
        price_val = None
        for p in price_elems:
            val_elems = _find_all_ns(p, "value")
            if val_elems and val_elems[0].text:
                price_val = _safe_float(val_elems[0].text.strip())
                break

        # Shares owned after transaction
        post_elems = _find_all_ns(txn, "sharesOwnedFollowingTransaction")
        post_shares = None
        for ps in post_elems:
            val_elems = _find_all_ns(ps, "value")
            if val_elems and val_elems[0].text:
                post_shares = _safe_int(val_elems[0].text.strip())
                break

        total_value = None
        if shares_val is not None and price_val is not None:
            total_value = round(shares_val * price_val, 2)

        transactions.append({
            "insider_name": owner_name,
            "title": owner_title,
            "transaction_date": txn_date or filing_date,
            "transaction_type": txn_type,
            "shares": shares_val or 0,
            "price_per_share": price_val,
            "total_value": total_value,
            "shares_owned_after": post_shares,
            "filing_date": filing_date,
            "accession": accession,
        })

    # Parse derivative transactions (options, etc.)
    for txn in _find_all_ns(root, "derivativeTransaction"):
        code_elems = _find_all_ns(txn, "transactionCode")
        code = code_elems[0].text.strip() if code_elems and code_elems[0].text else ""
        txn_type = _TRANSACTION_CODES.get(code, code)

        date_elems = _find_all_ns(txn, "transactionDate")
        txn_date = ""
        for d in date_elems:
            val_elems = _find_all_ns(d, "value")
            if val_elems and val_elems[0].text:
                txn_date = val_elems[0].text.strip()
                break

        shares_elems = _find_all_ns(txn, "transactionShares")
        shares_val = None
        for s in shares_elems:
            val_elems = _find_all_ns(s, "value")
            if val_elems and val_elems[0].text:
                shares_val = _safe_int(val_elems[0].text.strip())
                break

        price_elems = _find_all_ns(txn, "transactionPricePerShare")
        price_val = None
        for p in price_elems:
            val_elems = _find_all_ns(p, "value")
            if val_elems and val_elems[0].text:
                price_val = _safe_float(val_elems[0].text.strip())
                break

        total_value = None
        if shares_val is not None and price_val is not None:
            total_value = round(shares_val * price_val, 2)

        transactions.append({
            "insider_name": owner_name,
            "title": owner_title,
            "transaction_date": txn_date or filing_date,
            "transaction_type": txn_type,
            "shares": shares_val or 0,
            "price_per_share": price_val,
            "total_value": total_value,
            "shares_owned_after": None,
            "filing_date": filing_date,
            "accession": accession,
        })

    return transactions


def get_insider_transactions(ticker: str, limit: int = 20) -> list[dict]:
    """Get recent insider buys/sells from SEC EDGAR Form 4 filings.

    Uses the SEC EDGAR submissions API to find Form 4 filings for the
    company, then parses each Form 4 XML for transaction details.

    Returns list of:
    {
        "insider_name": str,
        "title": str (CEO, CFO, Director, etc.),
        "transaction_date": str,
        "transaction_type": "Purchase" | "Sale" | "Option Exercise",
        "shares": int,
        "price_per_share": float | None,
        "total_value": float | None,
        "shares_owned_after": int | None,
        "filing_date": str,
        "accession": str,
    }
    """
    cache_key = f"insider_txns|{ticker.upper()}|{limit}"
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    cik = _resolve_cik(ticker)
    if not cik:
        return []

    submissions = _get_submissions(cik)
    if not submissions:
        return []

    # Extract Form 4 filings from submissions
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    # Collect Form 4 filing metadata
    form4_filings: list[dict] = []
    for i in range(len(accessions)):
        form = forms[i] if i < len(forms) else ""
        if form not in ("4", "4/A"):
            continue
        form4_filings.append({
            "accession": accessions[i],
            "filing_date": dates[i] if i < len(dates) else "",
            "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
        })
        # Fetch more filings than limit since some may fail to parse
        if len(form4_filings) >= limit * 2:
            break

    all_transactions: list[dict] = []
    cik_raw = str(int(cik))

    for filing in form4_filings:
        if len(all_transactions) >= limit:
            break

        acc = filing["accession"]
        acc_clean = acc.replace("-", "")
        primary_doc = filing["primary_doc"]

        if not primary_doc:
            continue

        # Fetch the Form 4 XML document
        doc_url = f"{SEC_BASE}/Archives/edgar/data/{cik_raw}/{acc_clean}/{primary_doc}"
        xml_content = _cached_get(
            f"form4_xml|{acc}",
            doc_url,
            timeout=10,
            parse_json=False,
        )
        if not xml_content:
            continue

        txns = _parse_form4_xml(xml_content, acc, filing["filing_date"])
        all_transactions.extend(txns)

    result = all_transactions[:limit]
    _CACHE[cache_key] = (time.time(), result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Form 13F — Institutional Holdings
# ═══════════════════════════════════════════════════════════════════════════

def get_institutional_holders(ticker: str, limit: int = 20) -> list[dict]:
    """Get top institutional holders from SEC EDGAR 13F filings.

    Uses the EDGAR full-text search API (EFTS) to find 13F-HR filings
    that mention this company's ticker, then parses the filing index
    for holder and position information.

    NOTE: 13F data is inherently complex — each 13F lists ALL holdings
    for an institution, not per-company. This function searches for
    recent 13F filings mentioning the ticker and extracts what's available.

    Returns list of:
    {
        "institution": str,
        "shares": int,
        "value": float,
        "change_shares": int | None,
        "change_pct": float | None,
        "filing_date": str,
    }
    """
    cache_key = f"inst_holders|{ticker.upper()}|{limit}"
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    # Search EFTS for 13F-HR filings mentioning this ticker
    search_url = (
        f"{EFTS_BASE}/search-index?"
        f"q=%22{ticker.upper()}%22&forms=13-F&dateRange=custom"
        f"&startdt={(datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')}"
        f"&enddt={datetime.now().strftime('%Y-%m-%d')}"
    )
    search_data = _cached_get(f"13f_search|{ticker.upper()}", search_url)

    holders: list[dict] = []

    if not search_data:
        # Fallback: use EFTS full-text search endpoint
        fallback_url = (
            f"{EFTS_BASE}/search-index?"
            f"q=%22{ticker.upper()}%22&forms=13-F-HR"
            f"&dateRange=custom"
            f"&startdt={(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')}"
            f"&enddt={datetime.now().strftime('%Y-%m-%d')}"
        )
        search_data = _cached_get(f"13f_search_fb|{ticker.upper()}", fallback_url)

    if not search_data or not isinstance(search_data, dict):
        _CACHE[cache_key] = (time.time(), [])
        return []

    hits = search_data.get("hits", {}).get("hits", [])
    if not hits:
        _CACHE[cache_key] = (time.time(), [])
        return []

    seen_filers: set[str] = set()

    for hit in hits[:limit * 2]:
        if len(holders) >= limit:
            break

        source = hit.get("_source", {})
        filer_name = source.get("display_names", [""])[0] if source.get("display_names") else ""
        if not filer_name:
            filer_name = source.get("entity_name", "Unknown")

        # Deduplicate by filer
        filer_key = filer_name.upper().strip()
        if filer_key in seen_filers:
            continue
        seen_filers.add(filer_key)

        filing_date = source.get("file_date", "")
        period = source.get("period_of_report", "")

        # Try to get the 13F information table for detailed holdings
        accession = source.get("accession_no", "")
        cik = source.get("entity_id", "")

        shares = 0
        value = 0.0

        if accession and cik:
            # Attempt to parse the 13F XML info table
            acc_clean = accession.replace("-", "")
            # The info table is usually named infotable.xml or primary_doc.xml
            info_url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{acc_clean}/"
            index_data = _cached_get(
                f"13f_idx|{accession}",
                f"{DATA_BASE}/submissions/CIK{str(cik).zfill(10)}.json",
            )
            # Extract shares from info table if we can find it
            if index_data and isinstance(index_data, dict):
                # Search through the recent filings for the matching accession
                recent = index_data.get("filings", {}).get("recent", {})
                acc_list = recent.get("accessionNumber", [])
                doc_list = recent.get("primaryDocument", [])
                for idx, a in enumerate(acc_list):
                    if a == accession and idx < len(doc_list):
                        doc_name = doc_list[idx]
                        xml_url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{acc_clean}/{doc_name}"
                        xml_content = _cached_get(
                            f"13f_xml|{accession}",
                            xml_url,
                            timeout=10,
                            parse_json=False,
                        )
                        if xml_content:
                            shares, value = _parse_13f_for_ticker(
                                xml_content, ticker.upper()
                            )
                        break

        holders.append({
            "institution": filer_name,
            "shares": shares,
            "value": value,
            "change_shares": None,
            "change_pct": None,
            "filing_date": filing_date or period,
        })

    # Sort by value descending, push zero-value entries to the end
    holders.sort(key=lambda h: h["value"], reverse=True)
    result = holders[:limit]
    _CACHE[cache_key] = (time.time(), result)
    return result


def _parse_13f_for_ticker(xml_content: str, ticker: str) -> tuple[int, float]:
    """Parse a 13F info table XML to find shares/value for a specific ticker.

    Returns (shares, value_usd). Returns (0, 0.0) if not found or parse fails.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        # 13F primary docs are often HTML, not XML — try text search fallback
        return _parse_13f_text_fallback(xml_content, ticker)

    total_shares = 0
    total_value = 0.0

    # Look for infoTable entries (namespace varies)
    for entry in _find_all_ns(root, "infoTable"):
        # Check if this entry is for our ticker
        name_elems = _find_all_ns(entry, "nameOfIssuer")
        cusip_elems = _find_all_ns(entry, "cusip")
        title_elems = _find_all_ns(entry, "titleOfClass")

        issuer_name = name_elems[0].text.strip().upper() if name_elems and name_elems[0].text else ""

        # Match by ticker appearing in issuer name (heuristic)
        if ticker not in issuer_name:
            continue

        shares_elems = _find_all_ns(entry, "sshPrnamt")
        value_elems = _find_all_ns(entry, "value")

        if shares_elems and shares_elems[0].text:
            s = _safe_int(shares_elems[0].text.strip())
            if s:
                total_shares += s
        if value_elems and value_elems[0].text:
            v = _safe_float(value_elems[0].text.strip())
            if v:
                # 13F values are in thousands of USD
                total_value += v * 1000

    return total_shares, total_value


def _parse_13f_text_fallback(content: str, ticker: str) -> tuple[int, float]:
    """Fallback text-based parsing for 13F documents that aren't clean XML."""
    # 13F primary docs are often HTML — just return zeros for now
    # Full HTML table parsing would add significant complexity
    return 0, 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Insider Summary — Dashboard-ready aggregation
# ═══════════════════════════════════════════════════════════════════════════

def get_insider_summary(ticker: str) -> dict:
    """Summary of insider activity for dashboard display.

    Aggregates the last 90 days of insider transactions into a
    concise overview with net sentiment signal.

    Returns:
    {
        "net_insider_sentiment": "Buying" | "Selling" | "Neutral",
        "buys_90d": int,
        "sells_90d": int,
        "net_shares_90d": int,
        "notable_transactions": list[dict] (top 5 by value),
    }
    """
    cache_key = f"insider_summary|{ticker.upper()}"
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    # Fetch a larger batch to ensure we get 90 days of data
    transactions = get_insider_transactions(ticker, limit=50)

    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    buys = 0
    sells = 0
    net_shares = 0
    valued_txns: list[dict] = []

    for txn in transactions:
        txn_date = txn.get("transaction_date", "")
        if txn_date < cutoff:
            continue

        txn_type = txn.get("transaction_type", "")
        shares = txn.get("shares", 0)

        if txn_type == "Purchase":
            buys += 1
            net_shares += shares
        elif txn_type == "Sale":
            sells += 1
            net_shares -= shares
        elif txn_type in ("Option Exercise",):
            # Option exercises are neutral — they don't indicate sentiment
            pass

        if txn.get("total_value") is not None:
            valued_txns.append(txn)

    # Determine sentiment
    if buys > sells and net_shares > 0:
        sentiment = "Buying"
    elif sells > buys and net_shares < 0:
        sentiment = "Selling"
    else:
        sentiment = "Neutral"

    # Top 5 by absolute value
    valued_txns.sort(key=lambda t: abs(t.get("total_value", 0) or 0), reverse=True)
    notable = valued_txns[:5]

    result = {
        "net_insider_sentiment": sentiment,
        "buys_90d": buys,
        "sells_90d": sells,
        "net_shares_90d": net_shares,
        "notable_transactions": notable,
    }
    _CACHE[cache_key] = (time.time(), result)
    return result
