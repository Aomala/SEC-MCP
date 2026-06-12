"""get_filings + get_filing_section — filing discovery and clean section text.

get_filings: submissions index (fast path) or EDGAR full-text search (EFTS)
when full_text_query is set. Every row carries accession number, acceptance
timestamp (when known), and a direct EDGAR URL.

get_filing_section: resolves an accession to its document, extracts a named
section (risk_factors, mdna, business, financial_statements, item_X for
8-Ks) as clean text with no HTML artifacts.

Caching: the per-company filings index honors the EDGAR-business-hours TTL
(60s live / 600s off-hours) so new filings surface within a minute while
off-hours polling stays cheap.
"""

from __future__ import annotations

# stdlib
import logging
import re
import threading
import time

import requests

from sec_mcp.config import get_config

# rate-limited EDGAR client
from sec_mcp.sec_client import get_sec_client

# response contract
from sec_mcp.surface.meta import (
    INVALID_INPUT,
    NOT_FOUND,
    UNKNOWN_TICKER,
    UPSTREAM_ERROR,
    ToolError,
    build_meta,
    parse_iso_date,
    require_pos_int,
    require_ticker,
)

# dynamic TTL for the filings index
from sec_mcp.surface.session import filings_index_ttl

log = logging.getLogger(__name__)

# EDGAR full-text search backend (the API behind efts.sec.gov/LATEST)
_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# Caller-friendly form aliases → official EDGAR form names
_FORM_ALIASES: dict[str, list[str]] = {
    "10-K": ["10-K", "10-K/A", "20-F", "20-F/A"],          # FPI annuals included
    "10-Q": ["10-Q", "10-Q/A", "6-K"],                     # FPI interim included
    "8-K": ["8-K", "8-K/A"],
    "S-1": ["S-1", "S-1/A", "F-1", "F-1/A"],               # FPI IPO form included
    "13F": ["13F-HR", "13F-HR/A"],
    "13D": ["SC 13D", "SC 13D/A"],
    "13G": ["SC 13G", "SC 13G/A"],
    "13D/G": ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"],
    "DEF 14A": ["DEF 14A", "DEFA14A"],
    "4": ["4", "4/A"],
}

# Filings-index cache: {cik: (fetched_unix, rows)} — TTL evaluated at read
# time via filings_index_ttl() so business-hours freshness is automatic.
_index_cache: dict[str, tuple[float, list[dict]]] = {}
_index_lock = threading.Lock()

# Section aliases this tool accepts (superset of the segmenter's, including
# the spec spelling "mdna")
_SECTION_NORMALIZE = {
    "risk_factors": "risk_factors",
    "mdna": "mda", "mda": "mda", "md&a": "mda",
    "business": "business",
    "financial_statements": "financial_statements",
}


def _normalize_accession(acc: str) -> str:
    """Accept '0000320193-24-000123' or bare digits; return dashed form."""
    digits = re.sub(r"[^0-9]", "", str(acc or ""))
    # A valid accession is exactly 18 digits: 10 (filer) + 2 (year) + 6 (seq)
    if len(digits) != 18:
        raise ToolError(INVALID_INPUT, f"Invalid accession number {acc!r}.",
                        "Accession format: 0000320193-24-000123 (18 digits).")
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def _expand_forms(form_type: str | None) -> list[str] | None:
    """Map a caller form alias to the official EDGAR form name list."""
    if form_type is None:
        return None
    ft = str(form_type).strip().upper()
    if ft in _FORM_ALIASES:
        return _FORM_ALIASES[ft]
    # Allow any literal EDGAR form name as an escape hatch (e.g. '25-NSE')
    if re.fullmatch(r"[A-Z0-9/\- ]{1,12}", ft):
        return [ft]
    raise ToolError(INVALID_INPUT, f"Unrecognized form_type {form_type!r}.",
                    f"Known aliases: {sorted(_FORM_ALIASES)} — or pass an exact EDGAR form name.")


def _filing_url(cik: str, accession: str, primary_doc: str | None) -> str:
    """Direct EDGAR URL: the primary document if known, else the filing index."""
    cik_raw = str(int(cik))                                # strip zero padding
    acc_nodash = accession.replace("-", "")
    if primary_doc:
        return f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_nodash}/{primary_doc}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_nodash}/"


def _load_company_index(cik: str) -> list[dict]:
    """All filings for a CIK from the submissions API, with acceptance times.

    Cached with the EDGAR-business-hours TTL: 60s while filings can arrive,
    600s while the index is frozen overnight/weekends.
    """
    ttl = filings_index_ttl()                              # dynamic, time-of-day aware
    with _index_lock:
        hit = _index_cache.get(cik)
        if hit and (time.time() - hit[0]) < ttl:
            return hit[1]
    client = get_sec_client()
    subs = client._get_submissions(cik)                    # rate-limited, retried
    recent = (subs.get("filings") or {}).get("recent") or {}
    rows: list[dict] = []
    # Columnar → row-wise, keeping the fields the tool contract requires
    accs = recent.get("accessionNumber") or []
    for i in range(len(accs)):
        rows.append({
            "accession": accs[i],
            "form": (recent.get("form") or [""] * len(accs))[i],
            "filingDate": (recent.get("filingDate") or [""] * len(accs))[i],
            # acceptanceDateTime is EDGAR's authoritative receipt timestamp
            "acceptedAt": (recent.get("acceptanceDateTime") or [None] * len(accs))[i],
            "reportDate": (recent.get("reportDate") or [None] * len(accs))[i],
            "primaryDoc": (recent.get("primaryDocument") or [None] * len(accs))[i],
            "items": (recent.get("items") or [None] * len(accs))[i],  # 8-K item list
        })
    with _index_lock:
        _index_cache[cik] = (time.time(), rows)
    return rows


def _efts_search(query: str, forms: list[str] | None, date_from, date_to,
                 cik: str | None, limit: int) -> list[dict]:
    """EDGAR full-text search (EFTS) — phrase search across filing contents."""
    params: dict[str, str] = {"q": f'"{query}"'}           # quoted = phrase search
    if forms:
        params["forms"] = ",".join(forms)                  # comma-separated form filter
    if date_from or date_to:
        params["dateRange"] = "custom"
        if date_from:
            params["startdt"] = date_from.isoformat()
        if date_to:
            params["enddt"] = date_to.isoformat()
    if cik:
        params["ciks"] = cik.zfill(10)                     # entity filter wants 10 digits
    try:
        resp = requests.get(_EFTS_URL, params=params, timeout=15,
                            headers={"User-Agent": get_config().edgar_identity})
        resp.raise_for_status()
        hits = ((resp.json() or {}).get("hits") or {}).get("hits") or []
    except Exception as exc:
        raise ToolError(UPSTREAM_ERROR, f"EDGAR full-text search failed: {exc}",
                        "EFTS only covers filings since 2001; retry once — "
                        "if it persists, drop full_text_query and filter by form/date.") from None
    rows: list[dict] = []
    for h in hits[:limit]:
        src = h.get("_source") or {}
        acc = src.get("adsh") or ""                        # accession in EFTS speak
        ciks = src.get("ciks") or []
        row_cik = str(ciks[0]) if ciks else ""
        # _id is "accession:document" — recover the matched document name
        doc = (h.get("_id") or "").split(":", 1)[1] if ":" in (h.get("_id") or "") else None
        rows.append({
            "accession": acc,
            "form": src.get("file_type") or src.get("form"),
            "filingDate": src.get("file_date"),
            "acceptedAt": None,                            # EFTS doesn't expose acceptance time
            "reportDate": src.get("period_of_report"),
            "entityNames": src.get("display_names") or [],
            "cik": row_cik.zfill(10) if row_cik else None,
            "url": _filing_url(row_cik, acc, doc) if (row_cik and acc) else None,
        })
    return rows


def get_filings_impl(ticker_or_cik=None, form_type=None, date_from=None,
                     date_to=None, full_text_query=None, limit=None) -> dict:
    """Core implementation for the get_filings tool."""
    t0 = time.time()                                       # latency clock
    limit = require_pos_int(limit, "limit", default=20, hi=100)
    d_from = parse_iso_date(date_from, "date_from")        # validated optional dates
    d_to = parse_iso_date(date_to, "date_to")
    if d_from and d_to and d_from > d_to:
        raise ToolError(INVALID_INPUT, "date_from is after date_to.",
                        "Swap the dates: date_from must be the earlier one.")
    forms = _expand_forms(form_type)                       # alias → EDGAR names

    # Resolve the company (optional for pure full-text searches)
    cik = None
    if ticker_or_cik:
        tk = require_ticker(ticker_or_cik, "ticker_or_cik")
        try:
            cik = get_sec_client().resolve_cik(tk)         # raises on unknown
        except Exception:
            raise ToolError(UNKNOWN_TICKER, f"Could not resolve {tk!r} to a CIK.",
                            "Use search_companies to find the right ticker or pass the CIK directly.") from None

    # ── full-text path (EFTS) ────────────────────────────────────────────
    if full_text_query:
        if not str(full_text_query).strip():
            raise ToolError(INVALID_INPUT, "full_text_query must be a non-empty string.",
                            "Example: full_text_query='supply chain disruption'.")
        rows = _efts_search(str(full_text_query).strip(), forms, d_from, d_to, cik, limit)
        return {
            "mode": "full_text",                           # tells the caller which path ran
            "query": full_text_query,
            "count": len(rows),
            "filings": rows,
            "meta": build_meta("edgar:efts_full_text_search", t0, cache_hit=False),
        }

    # ── index path (submissions) ─────────────────────────────────────────
    if not cik:
        raise ToolError(INVALID_INPUT, "Provide ticker_or_cik (or a full_text_query).",
                        "Example: get_filings('AAPL', form_type='10-K').")
    ttl_hit = False
    with _index_lock:                                      # detect cache-hit for meta
        hit = _index_cache.get(cik)
        ttl_hit = bool(hit and (time.time() - hit[0]) < filings_index_ttl())
    all_rows = _load_company_index(cik)
    out: list[dict] = []
    for r in all_rows:
        # Form filter (after alias expansion, exact match against EDGAR names)
        if forms and r["form"] not in forms:
            continue
        # Date-range filter on the filing date
        fd = parse_iso_date(r["filingDate"], "filingDate") if r["filingDate"] else None
        if d_from and (not fd or fd < d_from):
            continue
        if d_to and (not fd or fd > d_to):
            continue
        out.append({
            "accession": r["accession"],
            "form": r["form"],
            "filingDate": r["filingDate"],
            "acceptedAt": r["acceptedAt"],                 # EDGAR acceptance timestamp
            "reportDate": r["reportDate"],
            "items": r["items"] or None,                   # 8-K item codes when present
            "url": _filing_url(cik, r["accession"], r["primaryDoc"]),
        })
        if len(out) >= limit:
            break
    return {
        "mode": "index",
        "cik": cik.zfill(10),
        "count": len(out),
        "filings": out,
        "meta": build_meta("edgar:submissions", t0, cache_hit=ttl_hit),
    }


# ── get_filing_section ───────────────────────────────────────────────────────

def _resolve_accession_cik(accession: str, ticker_or_cik: str | None) -> str:
    """Find the CIK whose archives hold this accession.

    Order: explicit ticker/CIK → accession's leading 10 digits (the filer) →
    EFTS lookup by accession id.
    """
    client = get_sec_client()
    # 1. Caller told us the company — trust it
    if ticker_or_cik:
        return client.resolve_cik(require_ticker(ticker_or_cik, "ticker_or_cik"))
    # 2. The first 10 digits of an accession are the filing entity's CIK —
    #    correct whenever the company filed under its own ID
    lead = accession[:10]
    try:
        subs = client._get_submissions(lead.zfill(10))
        accs = ((subs.get("filings") or {}).get("recent") or {}).get("accessionNumber") or []
        if accession in accs:
            return lead.zfill(10)
    except Exception:
        pass                                               # fall through to EFTS
    # 3. EFTS indexes every filing by accession id
    try:
        resp = requests.get(_EFTS_URL, params={"q": f'"{accession}"'}, timeout=15,
                            headers={"User-Agent": get_config().edgar_identity})
        resp.raise_for_status()
        hits = ((resp.json() or {}).get("hits") or {}).get("hits") or []
        for h in hits:
            src = h.get("_source") or {}
            if src.get("adsh") == accession and src.get("ciks"):
                return str(src["ciks"][0]).zfill(10)
    except Exception:
        pass
    raise ToolError(NOT_FOUND, f"Could not locate filer for accession {accession}.",
                    "Pass ticker_or_cik alongside the accession to disambiguate.")


def _extract_8k_item(text: str, item: str) -> str | None:
    """Pull one item (e.g. 'Item 2.02') out of cleaned 8-K text."""
    # Normalize "item_2.02" / "2.02" / "item 2.02" → the numeric id
    m = re.search(r"(\d+\.\d+)", item)
    if not m:
        return None
    num = re.escape(m.group(1))
    # Match from "Item 2.02" up to the next "Item d.dd" or the signature block
    pattern = re.compile(
        rf"item\s*{num}[.:\s]*(.*?)(?=item\s*\d+\.\d+|signature)", re.I | re.S)
    found = pattern.search(text)
    if not found:
        return None
    body = found.group(1).strip()
    return body or None


def _clean_text(text: str) -> str:
    """Final whitespace normalization — get_filing_document already strips HTML."""
    # Collapse 3+ newlines to 2, strip trailing spaces per line
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove stray HTML entities + non-breaking spaces that survive tag stripping
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#160;", " ")
    text = text.replace("\xa0", " ")                       # NBSP → regular space
    text = re.sub(r" {3,}", "  ", text)                    # collapse space runs
    return text.strip()


def get_filing_section_impl(accession, section, ticker_or_cik=None,
                            max_length=None) -> dict:
    """Core implementation for the get_filing_section tool."""
    t0 = time.time()                                       # latency clock
    acc = _normalize_accession(accession)                  # validates format
    max_length = require_pos_int(max_length, "max_length", default=80000,
                                 lo=1000, hi=400000)
    sec_raw = str(section or "").strip().lower()
    if not sec_raw:
        raise ToolError(INVALID_INPUT, "'section' is required.",
                        "Use one of: risk_factors, mdna, business, "
                        "financial_statements, or item_X for 8-Ks (e.g. item_2.02).")
    is_8k_item = sec_raw.startswith("item")                # item_2.02-style request
    if not is_8k_item and sec_raw not in _SECTION_NORMALIZE:
        raise ToolError(INVALID_INPUT, f"Unknown section {section!r}.",
                        "Use: risk_factors, mdna, business, financial_statements, "
                        "or item_X (8-K), e.g. item_8.01.")

    cik = _resolve_accession_cik(acc, ticker_or_cik)       # raises structured errors
    client = get_sec_client()

    if is_8k_item:
        # 8-Ks are short — fetch full cleaned text, slice the requested item
        full = client.get_filing_document(cik, acc, section=None, max_length=max_length)
        if not full or not full.strip():
            raise ToolError(NOT_FOUND, f"No document text for accession {acc}.",
                            "Verify the accession with get_filings first.")
        body = _extract_8k_item(full, sec_raw)
        if not body:
            raise ToolError(NOT_FOUND, f"Item {sec_raw!r} not present in filing {acc}.",
                            "Check the filing's `items` list from get_filings — "
                            "8-Ks only contain the items they disclose.")
        return {
            "accession": acc,
            "cik": cik.zfill(10),
            "section": sec_raw,
            "text": _clean_text(body)[:max_length],
            "meta": build_meta("edgar:archives+8k_item_parser", t0, cache_hit=False),
        }

    # 10-K/10-Q style sections go through the line-level segmenter
    canonical = _SECTION_NORMALIZE[sec_raw]                # mdna → mda etc.
    text = client.get_filing_document(cik, acc, section=canonical, max_length=max_length)
    # The segmenter falls back to full text with a bracket note — treat a
    # too-short or missing extraction as NOT_FOUND rather than returning junk
    if not text or len(text.strip()) < 200 or text.startswith("[Could not"):
        raise ToolError(NOT_FOUND,
                        f"Section {section!r} could not be isolated in filing {acc}.",
                        "Older or non-standard filings may not segment cleanly; "
                        "fetch the full document via the filing URL from get_filings.")
    return {
        "accession": acc,
        "cik": cik.zfill(10),
        "section": sec_raw,
        "text": _clean_text(text)[:max_length],
        "meta": build_meta("edgar:archives+section_segmenter", t0, cache_hit=False),
    }
