"""get_etf_profile — fund detection + profile (expense ratio, AUM, class).

Answers "is this an ETF, and what is it?" for tickers where XBRL
fundamentals don't apply (SPY, QQQ, VOO…). Seed registry merged with
Polygon reference details when the key is present.
"""

from __future__ import annotations

# stdlib
import time

# seed registry + Polygon enrichment
from sec_mcp.core.etf import get_etf_profile as _core_profile

# response contract helpers
from sec_mcp.surface.meta import build_meta, require_ticker


def get_etf_profile_impl(ticker) -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")

    result = _core_profile(tk)                              # never raises; isEtf=False for equities
    out = {
        "ticker": tk,
        "isEtf": bool(result.get("isEtf")),
        "profile": result.get("profile"),                   # null for ordinary equities
    }
    src = (result.get("meta") or {}).get("source", "seed")
    out["meta"] = build_meta(src, t0, cache_hit=False)
    if not out["isEtf"]:
        # not an error — the caller just learned it should use get_fundamentals
        out["hint"] = "Not a fund — use get_fundamentals/get_quote for this ticker."
    return out
