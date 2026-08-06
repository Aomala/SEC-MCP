"""find_peers — comparable-company discovery with relevance scoring.

Curated PEER_MAP first (industry consensus groups), SIC-code lookup as the
fallback so mid-caps outside the map (CAVA and friends) still get peers.
Feed the result into compare() for the side-by-side.
"""

from __future__ import annotations

# stdlib
import time

# three-strategy peer discovery (PEER_MAP → SIC → custom)
from sec_mcp.core.peer_engine import PeerEngine

# response contract helpers
from sec_mcp.surface.meta import (
    NOT_FOUND,
    ToolError,
    build_meta,
    require_ticker,
)

# lazy singleton — engine is stateless but imports financials at construction
_engine: PeerEngine | None = None


def _get_engine() -> PeerEngine:
    global _engine
    if _engine is None:
        _engine = PeerEngine()
    return _engine


def find_peers_impl(ticker, max_peers: int = 5,
                    custom_peers: list[str] | None = None) -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")
    n = max(1, min(int(max_peers or 5), 10))

    criteria = {"custom_peers": custom_peers} if custom_peers else None
    peers = _get_engine().find_peers(tk, max_peers=n, criteria=criteria)
    if not peers:
        raise ToolError(
            NOT_FOUND, f"No peers found for {tk}.",
            "Pass custom_peers=[...] to compare against a hand-picked list, "
            "or use screen() with a sector filter to build one.",
        )
    return {
        "ticker": tk,
        "peers": [{
            "ticker": p.get("ticker"),
            "name": p.get("name"),
            "sic": p.get("sic"),
            "relevanceScore": p.get("relevance_score"),
            "reason": p.get("reason"),                      # which strategy matched
        } for p in peers],
        "count": len(peers),
        "meta": build_meta("peer_map+sic", t0, cache_hit=False),
    }
