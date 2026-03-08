"""SEC-MCP V2 Core Modules — High-level analysis and enrichment engines.

Public interfaces:
  - MarketDataProvider: Real-time price, valuation, 52-week metrics
  - FilingDiffEngine: Compare financials and narrative sections between periods
  - PeerEngine: Find comparable companies by industry, SIC, or custom lists
  - Screener: Filter companies by financial metrics (multithreaded, cached)
  - CacheLayer: Unified L1/L2/L3 multi-tier cache with TTL and promotion
"""

# Import public classes for convenient access from upstream code.
# This allows: from sec_mcp.core import MarketDataProvider, FilingDiffEngine, etc.

from sec_mcp.core.cache import CacheLayer, get_cache  # noqa: F401
from sec_mcp.core.filing_diff import FilingDiffEngine  # noqa: F401
from sec_mcp.core.market_data import MarketDataProvider  # noqa: F401
from sec_mcp.core.peer_engine import PeerEngine  # noqa: F401
from sec_mcp.core.screener import Screener  # noqa: F401

# Expose public API version
__version__ = "2.0.0"
__all__ = [
    "MarketDataProvider",
    "FilingDiffEngine",
    "PeerEngine",
    "Screener",
    "CacheLayer",
    "get_cache",
]
