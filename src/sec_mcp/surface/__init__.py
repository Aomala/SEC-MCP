"""sec_mcp.surface — the v2 MCP tool surface (9 tools, 24/7 reliable).

Every module here follows three hard rules:
  1. Every tool response carries a `meta` block: {source, asOf, cacheHit, latencyMs}.
  2. Every error is structured: {error, code, hint} — never a raw stack trace.
  3. Off-hours is never an error — quotes label `session: "closed"`, filings
     and fundamentals answer normally at any hour.
"""
