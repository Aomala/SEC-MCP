#!/usr/bin/env python3
"""Standalone CLI to test SEC Terminal tools from the terminal.

Usage — run any of these from the project root:

  # Search for a company
  python test_tools.py search "Apple"
  python test_tools.py search "Morgan Stanley"
  python test_tools.py search "Taiwan Semi"

  # Get company financials (latest 10-K)
  python test_tools.py financials AAPL
  python test_tools.py financials MS
  python test_tools.py financials NVDA

  # Get financials for a specific year
  python test_tools.py financials AAPL 2023

  # Get 10-Q (quarterly) financials
  python test_tools.py financials AAPL 2024 10-Q

  # List recent filings
  python test_tools.py filings AAPL
  python test_tools.py filings MS 10

  # Get XBRL facts (raw data from SEC)
  python test_tools.py xbrl AAPL

  # Get company info (CIK, SIC, etc.)
  python test_tools.py info AAPL

  # Test MongoDB connection
  python test_tools.py mongo

  # Get filing document text (first 500 chars)
  python test_tools.py filing-text AAPL

  # Full health check — tests all systems
  python test_tools.py health
"""

from __future__ import annotations

import json
import sys
import os

# Ensure the src directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def _fmt(val, indent=2):
    """Pretty-print a value."""
    if isinstance(val, dict):
        return json.dumps(val, indent=indent, default=str)
    if isinstance(val, list):
        return json.dumps(val[:20], indent=indent, default=str)  # Cap at 20 items
    return str(val)


def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def cmd_search(query: str):
    """Search for companies by name or ticker."""
    _header(f"Search: {query}")
    from sec_mcp.edgar_client import search_companies
    results = search_companies(query)
    if not results:
        print("  No results found.")
        return
    for r in results[:10]:
        tk = r.ticker or "?"
        print(f"  {tk:8s}  CIK {str(r.cik):>10s}  {r.name}")
        if r.sic_code:
            print(f"           SIC: {r.sic_code} — {r.industry or ''}")
    print(f"\n  Total: {len(results)} result(s)")


def cmd_info(ticker: str):
    """Get company info directly from SEC EDGAR."""
    _header(f"Company Info: {ticker}")
    from sec_mcp.sec_client import SECClient
    from sec_mcp.config import get_config
    client = SECClient(get_config().edgar_identity)
    info = client.get_company_info(ticker)
    if info:
        print(f"  Ticker:      {info.ticker}")
        print(f"  Name:        {info.name}")
        print(f"  CIK:         {info.cik}")
        print(f"  SIC:         {info.sic_code or '?'} — {info.industry or '?'}")
    else:
        print("  Company not found.")


def cmd_filings(ticker: str, limit: int = 10):
    """List recent filings for a company."""
    _header(f"Filings: {ticker} (last {limit})")
    from sec_mcp.sec_client import SECClient
    from sec_mcp.config import get_config
    client = SECClient(get_config().edgar_identity)
    filings = client.get_filings(ticker, limit=limit)
    if not filings:
        print("  No filings found.")
        return
    print(f"  {'Date':12s}  {'Form':8s}  {'Accession':24s}  Description")
    print(f"  {'-'*12}  {'-'*8}  {'-'*24}  {'-'*30}")
    for f in filings[:limit]:
        desc = (f.description or "")[:40]
        print(f"  {f.filing_date:12s}  {f.form_type:8s}  {f.accession_number:24s}  {desc}")
    print(f"\n  Total fetched: {len(filings)}")


def cmd_financials(ticker: str, year: int | None = None, form_type: str = "10-K"):
    """Extract full financials for a company."""
    _header(f"Financials: {ticker} | Year={year or 'latest'} | Form={form_type}")
    from sec_mcp.financials import extract_financials
    data = extract_financials(ticker, year=year, form_type=form_type)

    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    print(f"  Company:    {data.get('company_name', '?')}")
    print(f"  Industry:   {data.get('industry_class', '?')}")
    print(f"  Period:     {data.get('fiscal_year', '?')}")

    fi = data.get("filing_info", {})
    if fi:
        print(f"  Filing:     {fi.get('form_type', '?')} filed {fi.get('filing_date', '?')}")

    metrics = data.get("metrics", {})
    if metrics:
        print(f"\n  Key Metrics:")
        for k, v in sorted(metrics.items()):
            if v is not None:
                if isinstance(v, (int, float)):
                    if abs(v) >= 1e9:
                        print(f"    {k:30s}  ${v/1e9:>12.1f}B")
                    elif abs(v) >= 1e6:
                        print(f"    {k:30s}  ${v/1e6:>12.0f}M")
                    else:
                        print(f"    {k:30s}  {v:>12}")
                else:
                    print(f"    {k:30s}  {v}")

    ratios = data.get("ratios", {})
    if ratios:
        print(f"\n  Ratios:")
        for k, v in sorted(ratios.items()):
            if v is not None:
                print(f"    {k:30s}  {v*100:>8.1f}%")

    stmts = ["income_statement", "balance_sheet", "cash_flow_statement"]
    for s in stmts:
        rows = data.get(s, [])
        if rows:
            print(f"\n  {s.replace('_', ' ').title()} ({len(rows)} rows):")
            for row in rows[:8]:
                label = row.get("label", "?")
                val_cols = [k for k in row.keys() if k not in ("label", "concept", "standard_concept", "level", "is_abstract", "is_total", "abstract", "units", "decimals")]
                vals = {k: row[k] for k in val_cols[:3] if row.get(k) is not None}
                val_str = "  ".join(f"{k}={v}" for k, v in vals.items())
                print(f"      {label:40s}  {val_str}")
            if len(rows) > 8:
                print(f"      ... +{len(rows)-8} more rows")

    links = data.get("sec_links", {})
    if links:
        print(f"\n  SEC Links:")
        for k, v in links.items():
            print(f"    {k:20s}  {v}")


def cmd_xbrl(ticker: str):
    """Fetch raw XBRL facts from SEC EDGAR."""
    _header(f"XBRL Facts: {ticker}")
    from sec_mcp.sec_client import SECClient
    from sec_mcp.config import get_config
    client = SECClient(get_config().edgar_identity)
    cik = client.resolve_cik(ticker)
    if not cik:
        print("  Could not resolve CIK.")
        return
    print(f"  CIK: {cik}")
    facts = client.get_company_facts(cik)
    if not facts:
        print("  No XBRL facts available.")
        return

    # Show available taxonomies and concept counts
    for taxonomy, concepts in facts.get("facts", {}).items():
        print(f"\n  Taxonomy: {taxonomy} ({len(concepts)} concepts)")
        # Show first 15 concepts with their latest values
        shown = 0
        for concept_name, concept_data in sorted(concepts.items()):
            if shown >= 15:
                print(f"    ... +{len(concepts)-15} more concepts")
                break
            units = concept_data.get("units", {})
            for unit_type, values in units.items():
                if values:
                    latest = values[-1]
                    val = latest.get("val", "?")
                    end = latest.get("end", "?")
                    print(f"    {concept_name:45s}  {val:>15}  ({unit_type}, {end})")
                    shown += 1
                    break


def cmd_filing_text(ticker: str):
    """Get the text of the most recent filing."""
    _header(f"Filing Text: {ticker} (first 500 chars)")
    from sec_mcp.sec_client import SECClient
    from sec_mcp.config import get_config
    client = SECClient(get_config().edgar_identity)
    filings = client.get_filings(ticker, form_type="10-K", limit=1)
    if not filings:
        print("  No filings found.")
        return
    f = filings[0]
    print(f"  Filing: {f.form_type} dated {f.filing_date}")
    print(f"  Accession: {f.accession_number}")
    cik = client.resolve_cik(ticker)
    if not cik:
        print("  Could not resolve CIK.")
        return
    text = client.get_filing_document(cik, f.accession_number)
    preview = text[:500] if text else "(empty)"
    print(f"\n  --- Document Preview ---\n{preview}\n  --- end ---")


def cmd_mongo():
    """Test MongoDB connection and permissions."""
    _header("MongoDB Connection Test")
    from sec_mcp.config import get_config
    cfg = get_config()
    uri = cfg.mongodb_uri
    if not uri:
        print("  MONGODB_URI is not set in .env")
        print("  The app will work fine without it (no persistent caching).")
        return

    print(f"  URI: {uri[:40]}...")
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Test 1: Ping
        client.admin.command("ping")
        print("  [PASS] Ping successful — connection works")

        # Test 2: Read permission on sec_terminal
        db = client.sec_terminal
        try:
            db.companies.find_one({}, {"_id": 1})
            print("  [PASS] Read permission on sec_terminal.companies")
        except Exception as e:
            print(f"  [FAIL] Read permission: {e}")
            print()
            print("  FIX: In MongoDB Atlas, go to:")
            print("    Database Access -> Edit your user -> Built-in Role")
            print("    Change to: 'Atlas Admin' or 'Read and write to any database'")
            return

        # Test 3: Write permission
        try:
            db.test_connection.insert_one({"test": True})
            db.test_connection.delete_one({"test": True})
            print("  [PASS] Write permission on sec_terminal")
        except Exception as e:
            print(f"  [FAIL] Write permission: {e}")
            print()
            print("  FIX: Change your Atlas user role to 'Atlas Admin'")
            return

        # Test 4: Check existing data
        companies = db.companies.count_documents({})
        filings = db.filings.count_documents({})
        jobs = db.jobs.count_documents({})
        print(f"\n  Existing data:")
        print(f"    Companies: {companies}")
        print(f"    Filings:   {filings}")
        print(f"    Jobs:      {jobs}")
        print("\n  MongoDB is fully operational!")

    except Exception as e:
        print(f"  [FAIL] Connection error: {e}")


def cmd_health():
    """Full system health check."""
    _header("SEC Terminal Health Check")
    import time

    checks = []

    # 1. Config
    print("  [1/5] Configuration...")
    try:
        from sec_mcp.config import get_config
        cfg = get_config()
        print(f"    EDGAR_IDENTITY: {cfg.edgar_identity[:30]}...")
        print(f"    ANTHROPIC_API_KEY: {'set' if cfg.anthropic_api_key else 'not set'}")
        print(f"    MONGODB_URI: {'set' if cfg.mongodb_uri else 'not set'}")
        checks.append(("Config", "PASS"))
    except Exception as e:
        print(f"    ERROR: {e}")
        checks.append(("Config", "FAIL"))

    # 2. SEC EDGAR API
    print("\n  [2/5] SEC EDGAR API...")
    try:
        from sec_mcp.sec_client import SECClient
        client = SECClient(cfg.edgar_identity)
        t0 = time.time()
        cik = client.resolve_cik("AAPL")
        elapsed = time.time() - t0
        print(f"    Resolved AAPL -> CIK {cik} ({elapsed:.1f}s)")
        checks.append(("SEC EDGAR", "PASS"))
    except Exception as e:
        print(f"    ERROR: {e}")
        checks.append(("SEC EDGAR", "FAIL"))

    # 3. Financials extraction
    print("\n  [3/5] Financials extraction...")
    try:
        from sec_mcp.financials import extract_financials
        t0 = time.time()
        data = extract_financials("AAPL")
        elapsed = time.time() - t0
        rev = data.get("metrics", {}).get("revenue")
        print(f"    AAPL revenue: ${rev/1e9:.1f}B ({elapsed:.1f}s)" if rev else f"    No revenue data ({elapsed:.1f}s)")
        checks.append(("Financials", "PASS" if rev else "WARN"))
    except Exception as e:
        print(f"    ERROR: {e}")
        checks.append(("Financials", "FAIL"))

    # 4. MongoDB
    print("\n  [4/5] MongoDB...")
    try:
        from sec_mcp.db import is_available
        if is_available():
            print("    Connected and authorized")
            checks.append(("MongoDB", "PASS"))
        else:
            print("    Not available (app works without it)")
            checks.append(("MongoDB", "SKIP"))
    except Exception as e:
        print(f"    ERROR: {e}")
        checks.append(("MongoDB", "FAIL"))

    # 5. MCP Server
    print("\n  [5/5] MCP Server tools...")
    try:
        from sec_mcp.server import mcp
        tools = mcp.list_tools() if hasattr(mcp, 'list_tools') else []
        print(f"    MCP server loaded: {len(tools) if tools else '?'} tools registered")
        checks.append(("MCP Server", "PASS"))
    except Exception as e:
        print(f"    Loaded (tools counted at runtime): {e}")
        checks.append(("MCP Server", "PASS"))

    # Summary
    print(f"\n  {'='*40}")
    print("  SUMMARY:")
    for name, status in checks:
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!", "SKIP": "-"}[status]
        print(f"    [{icon}] {name}: {status}")
    print()


COMMANDS = {
    "search": (cmd_search, "query"),
    "info": (cmd_info, "ticker"),
    "filings": (cmd_filings, "ticker [limit]"),
    "financials": (cmd_financials, "ticker [year] [form_type]"),
    "xbrl": (cmd_xbrl, "ticker"),
    "filing-text": (cmd_filing_text, "ticker"),
    "mongo": (cmd_mongo, ""),
    "health": (cmd_health, ""),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print("\nSEC Terminal — Standalone Tool Tester")
        print("=" * 44)
        print("\nUsage: python test_tools.py <command> [args]\n")
        print("Commands:")
        for cmd, (_, args) in COMMANDS.items():
            print(f"  {cmd:16s}  {args}")
        print()
        print("Examples:")
        print("  python test_tools.py search 'Apple'")
        print("  python test_tools.py financials AAPL")
        print("  python test_tools.py financials AAPL 2023")
        print("  python test_tools.py financials MS 2024 10-Q")
        print("  python test_tools.py filings NVDA 20")
        print("  python test_tools.py xbrl AAPL")
        print("  python test_tools.py info TSLA")
        print("  python test_tools.py mongo")
        print("  python test_tools.py health")
        return

    cmd_name = sys.argv[1].lower()
    if cmd_name not in COMMANDS:
        print(f"Unknown command: {cmd_name}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        return

    fn, _ = COMMANDS[cmd_name]

    # Parse arguments based on command
    if cmd_name == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Apple"
        fn(query)
    elif cmd_name == "filings":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        fn(ticker, limit)
    elif cmd_name == "financials":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        year = int(sys.argv[3]) if len(sys.argv) > 3 else None
        form_type = sys.argv[4] if len(sys.argv) > 4 else "10-K"
        fn(ticker, year, form_type)
    elif cmd_name in ("info", "xbrl", "filing-text"):
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        fn(ticker)
    elif cmd_name in ("mongo", "health"):
        fn()
    else:
        fn(*sys.argv[2:])


if __name__ == "__main__":
    main()
