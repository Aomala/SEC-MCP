"""Market data provider using yfinance with graceful fallback.

Provides real-time stock prices, valuations, and 52-week metrics via yfinance.
Falls back gracefully if yfinance is not installed. Uses lazy singleton pattern
with 5-minute TTL for prices and 15-minute for valuations. Thread-safe caching.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

# Attempt to import yfinance; set flag to False if unavailable
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Module logger for all warning/error messages
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Lazy Singleton Pattern (follows sec_client.py convention)
# ═══════════════════════════════════════════════════════════════════════════

# Global instance holder — initially None, populated by get_market_data_provider()
_provider: MarketDataProvider | None = None

# Lock to ensure thread-safe singleton initialization
_provider_lock = threading.Lock()


def get_market_data_provider() -> MarketDataProvider:
    """Return or create the global MarketDataProvider singleton instance.
    
    Thread-safe lazy initialization using double-checked locking pattern.
    First call creates and caches the instance; subsequent calls return cached instance.
    """
    # Check global _provider without lock first (fast path)
    global _provider
    if _provider is not None:
        return _provider
    
    # Acquire lock and check again (standard double-checked locking pattern)
    with _provider_lock:
        if _provider is None:
            # Create new instance and store in global _provider
            _provider = MarketDataProvider()
        return _provider


# ═══════════════════════════════════════════════════════════════════════════
#  Cache Entry Data Structure
# ═══════════════════════════════════════════════════════════════════════════

class _CacheEntry:
    """Simple cache entry with timestamp and TTL tracking.
    
    Attributes:
        value: The cached data (dict or None)
        timestamp: When this entry was created (datetime with UTC timezone)
        ttl_seconds: Time to live in seconds before expiration
    """
    
    def __init__(self, value: Any, ttl_seconds: int):
        """Initialize cache entry with value and TTL.
        
        Args:
            value: The data to cache (typically dict)
            ttl_seconds: Seconds until this entry expires
        """
        # Store the actual cached value
        self.value = value
        # Capture current UTC time for TTL calculation
        self.timestamp = datetime.now(timezone.utc)
        # Store TTL duration in seconds
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if this cache entry has exceeded its TTL.
        
        Returns:
            True if current time >= timestamp + ttl_seconds, False otherwise
        """
        # Calculate when this entry expires (timestamp + TTL duration)
        expiration_time = self.timestamp + timedelta(seconds=self.ttl_seconds)
        # Return True if current UTC time is past expiration
        return datetime.now(timezone.utc) >= expiration_time


# ═══════════════════════════════════════════════════════════════════════════
#  Market Data Provider Class
# ═══════════════════════════════════════════════════════════════════════════

class MarketDataProvider:
    """Provides real-time market data via yfinance with graceful fallback.
    
    Features:
      - get_price(ticker) → dict with price, change, volume, market_cap, 52-week ranges, P/E
      - get_valuation(ticker, xbrl_metrics) → computed P/S, P/B, EV/EBITDA, EV/Revenue
      - 5-min TTL for prices, 15-min for valuations (in-memory dict cache)
      - Thread-safe with threading.Lock
      - Graceful degradation: returns None with warning if yfinance unavailable
    
    Follows sec_client.py patterns:
      - Lazy singleton via get_market_data_provider()
      - Rate limiting built-in (yfinance handles it)
      - Logging for debugging/monitoring
    """
    
    def __init__(self):
        """Initialize MarketDataProvider with empty caches.
        
        Sets up:
          - _price_cache: dict mapping ticker → _CacheEntry (5-min TTL)
          - _valuation_cache: dict mapping ticker → _CacheEntry (15-min TTL)
          - _cache_lock: threading.Lock for thread-safe cache access
        """
        # Cache for price data (ticker → _CacheEntry); 5 minute TTL
        self._price_cache: dict[str, _CacheEntry] = {}
        
        # Cache for valuation ratios (ticker → _CacheEntry); 15 minute TTL
        self._valuation_cache: dict[str, _CacheEntry] = {}
        
        # Lock protecting both caches during read/write
        self._cache_lock = threading.Lock()
        
        # Log initialization status based on yfinance availability
        if YFINANCE_AVAILABLE:
            log.info("MarketDataProvider initialized with yfinance available")
        else:
            log.warning("yfinance not installed — MarketDataProvider will return None")
    
    def get_price(self, ticker: str) -> dict[str, Any] | None:
        """Fetch current stock price and metrics for a ticker.
        
        Uses cached data if available and not expired (5-min TTL).
        Falls back to None if yfinance unavailable or network error.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "JPM")
        
        Returns:
            Dict with keys:
              - price: Current stock price (float)
              - change: Dollar change from previous close (float)
              - change_pct: Percent change from previous close (float, e.g., 2.5 for +2.5%)
              - volume: Current trading volume (int)
              - market_cap: Market capitalization in USD (int), or None if unavailable
              - high_52w: 52-week high price (float), or None if unavailable
              - low_52w: 52-week low price (float), or None if unavailable
              - pe_ratio: Trailing P/E ratio (float), or None if unavailable
              - forward_pe: Forward P/E ratio (float), or None if unavailable
            Returns None if yfinance unavailable or fetch fails.
        
        Raises:
            No exceptions — logs warnings and returns None on error.
        """
        # Short-circuit if yfinance is not available
        if not YFINANCE_AVAILABLE:
            log.warning("get_price(%s): yfinance not installed, returning None", ticker)
            return None
        
        # Normalize ticker to uppercase for consistency
        ticker_upper = ticker.upper()
        
        # Acquire lock to safely check cache
        with self._cache_lock:
            # Check if we have this ticker cached and it's not expired
            if ticker_upper in self._price_cache:
                entry = self._price_cache[ticker_upper]
                # If cache entry is still valid (not expired), return cached value
                if not entry.is_expired():
                    log.debug("get_price(%s): returning cached data", ticker_upper)
                    return entry.value
                # If expired, remove from cache (it will be fetched fresh below)
                else:
                    del self._price_cache[ticker_upper]
        
        # Cache miss or expired — fetch fresh data from yfinance
        try:
            # Fetch ticker data from yfinance (network call)
            # progress=False suppresses yfinance progress output
            ticker_obj = yf.Ticker(ticker_upper)
            
            # Fetch historical data to get previous close and ranges
            # period="1y" gets 1 year of daily data for 52-week highs/lows
            hist = ticker_obj.history(period="1y")
            
            # Get current info snapshot with latest price and metrics
            # info dict contains most current market data
            info = ticker_obj.info or {}
            
            # Get latest price (bid-ask if available, else previous close)
            # Check multiple keys for compatibility with different info payloads
            current_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or (hist["Close"].iloc[-1] if not hist.empty else None)
            )
            
            # If we couldn't determine current price, log and return None
            if current_price is None:
                log.warning("get_price(%s): could not determine current price", ticker_upper)
                return None
            
            # Get previous close (for calculating change); default to current price if unavailable
            previous_close = (
                info.get("regularMarketPreviousClose")
                or (hist["Close"].iloc[-2] if len(hist) >= 2 else current_price)
            )
            
            # Calculate dollar change from previous close
            change_dollars = current_price - previous_close
            
            # Calculate percent change, avoiding division by zero
            if previous_close > 0:
                change_percent = (change_dollars / previous_close) * 100
            else:
                change_percent = 0.0
            
            # Get trading volume (default to 0 if unavailable)
            volume = info.get("regularMarketVolume") or info.get("volume") or 0
            
            # Get market cap (in USD); may be None if not available
            market_cap = info.get("marketCap")
            
            # Calculate 52-week high and low from historical data
            # high_52w: maximum Close price in the year of data
            high_52w = hist["Close"].max() if not hist.empty else None
            # low_52w: minimum Close price in the year of data
            low_52w = hist["Close"].min() if not hist.empty else None
            
            # Get trailing P/E ratio (earnings/price); None if unavailable
            pe_ratio = info.get("trailingPE")
            
            # Get forward P/E ratio (estimated next earnings/price); None if unavailable
            forward_pe = info.get("forwardPE")
            
            # Build result dict with all available metrics
            result = {
                "price": float(current_price),
                "change": float(change_dollars),
                "change_pct": float(change_percent),
                "volume": int(volume),
                "market_cap": int(market_cap) if market_cap else None,
                "high_52w": float(high_52w) if high_52w is not None else None,
                "low_52w": float(low_52w) if low_52w is not None else None,
                "pe_ratio": float(pe_ratio) if pe_ratio else None,
                "forward_pe": float(forward_pe) if forward_pe else None,
            }
            
            # Store result in cache with 5-minute TTL
            with self._cache_lock:
                self._price_cache[ticker_upper] = _CacheEntry(result, ttl_seconds=300)
            
            # Log successful fetch
            log.debug("get_price(%s): fetched and cached (price=%.2f)", ticker_upper, current_price)
            
            # Return the result dict to caller
            return result
        
        except Exception as e:
            # Catch all exceptions (network, missing ticker, etc.) and log
            log.warning("get_price(%s) failed: %s", ticker_upper, e)
            # Return None to indicate failure (graceful degradation)
            return None
    
    def get_valuation(
        self,
        ticker: str,
        xbrl_metrics: dict[str, float | int],
    ) -> dict[str, Any] | None:
        """Compute valuation ratios by combining market data with XBRL fundamentals.
        
        Uses cached result if available and not expired (15-min TTL).
        Falls back to None if yfinance unavailable or required metrics missing.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "JPM")
            xbrl_metrics: Dict of XBRL-extracted fundamentals, expected to contain:
              - revenue (or NetRevenues)
              - net_income (or NetIncome)
              - total_assets (or Assets)
              - total_equity (or Equity, StockholdersEquity)
              - ebitda (or computed from: Operating Income + Depreciation + Amortization)
        
        Returns:
            Dict with valuation ratios:
              - pe_ratio: Market Cap / Net Income (float), or None if unavailable
              - ps_ratio: Market Cap / Revenue (float), or None if unavailable
              - pb_ratio: Market Cap / Total Equity (float), or None if unavailable
              - ev_ebitda: Enterprise Value / EBITDA (float), or None if unavailable
              - ev_revenue: Enterprise Value / Revenue (float), or None if unavailable
              - market_cap: Market capitalization in USD (int), or None if unavailable
              - enterprise_value: Market Cap + Total Debt (int), or None if unavailable
            Returns None if yfinance unavailable, price data unavailable, or metrics incomplete.
        
        Raises:
            No exceptions — logs warnings and returns None on error.
        """
        # Short-circuit if yfinance not available
        if not YFINANCE_AVAILABLE:
            log.warning("get_valuation(%s): yfinance not installed, returning None", ticker)
            return None
        
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Acquire lock and check valuation cache
        with self._cache_lock:
            # Check if cached valuation exists and hasn't expired
            if ticker_upper in self._valuation_cache:
                entry = self._valuation_cache[ticker_upper]
                # If still valid (not expired), return immediately
                if not entry.is_expired():
                    log.debug("get_valuation(%s): returning cached data", ticker_upper)
                    return entry.value
                # If expired, remove from cache
                else:
                    del self._valuation_cache[ticker_upper]
        
        try:
            # Fetch current price data (uses get_price which handles caching)
            price_data = self.get_price(ticker_upper)
            
            # If price data unavailable, return None (can't compute valuations without price)
            if price_data is None:
                log.warning("get_valuation(%s): price data unavailable", ticker_upper)
                return None
            
            # Extract market cap from price data; if None, we can't compute valuations
            market_cap = price_data.get("market_cap")
            if market_cap is None:
                log.warning("get_valuation(%s): market_cap not available from price data", ticker_upper)
                return None
            
            # Extract XBRL metrics, providing fallback keys for flexibility
            # (e.g., code might pass either "revenue" or "NetRevenues")
            revenue = (
                xbrl_metrics.get("revenue")
                or xbrl_metrics.get("NetRevenues")
                or xbrl_metrics.get("Revenues")
                or None
            )
            
            # Extract net income, trying multiple key names for flexibility
            net_income = (
                xbrl_metrics.get("net_income")
                or xbrl_metrics.get("NetIncome")
                or xbrl_metrics.get("NetIncomeAvailableToCommonShareholders")
                or None
            )
            
            # Extract total assets, trying multiple key names
            total_assets = (
                xbrl_metrics.get("total_assets")
                or xbrl_metrics.get("Assets")
                or None
            )
            
            # Extract total equity (shareholders' equity), trying multiple key names
            total_equity = (
                xbrl_metrics.get("total_equity")
                or xbrl_metrics.get("StockholdersEquity")
                or xbrl_metrics.get("Equity")
                or None
            )
            
            # Extract EBITDA or try to compute from components
            ebitda = (
                xbrl_metrics.get("ebitda")
                or xbrl_metrics.get("EBITDA")
                or xbrl_metrics.get("OperatingIncome")
                or None
            )
            
            # Extract total debt for enterprise value calculation
            total_debt = (
                xbrl_metrics.get("total_debt")
                or xbrl_metrics.get("Debt")
                or xbrl_metrics.get("LongTermDebt")
                or 0
            )
            
            # Initialize result dict with None defaults (will be filled if data available)
            result = {
                "pe_ratio": None,
                "ps_ratio": None,
                "pb_ratio": None,
                "ev_ebitda": None,
                "ev_revenue": None,
                "market_cap": market_cap,
                "enterprise_value": None,
            }
            
            # Compute P/E ratio if we have net income (and it's positive to avoid div by zero)
            if net_income is not None and net_income > 0:
                # P/E = Market Cap / Net Income
                result["pe_ratio"] = float(market_cap) / float(net_income)
            
            # Compute P/S ratio if we have revenue (and it's positive)
            if revenue is not None and revenue > 0:
                # P/S = Market Cap / Revenue
                result["ps_ratio"] = float(market_cap) / float(revenue)
            
            # Compute P/B ratio if we have total equity (and it's positive)
            if total_equity is not None and total_equity > 0:
                # P/B = Market Cap / Total Equity (Book Value)
                result["pb_ratio"] = float(market_cap) / float(total_equity)
            
            # Compute enterprise value and EV ratios
            # EV = Market Cap + Total Debt - Cash (simplified: assume we have debt, no cash adjustment)
            enterprise_value = market_cap + total_debt
            result["enterprise_value"] = enterprise_value
            
            # Compute EV/EBITDA if we have EBITDA (and it's positive)
            if ebitda is not None and ebitda > 0:
                # EV/EBITDA = Enterprise Value / EBITDA
                result["ev_ebitda"] = float(enterprise_value) / float(ebitda)
            
            # Compute EV/Revenue if we have revenue (and it's positive)
            if revenue is not None and revenue > 0:
                # EV/Revenue = Enterprise Value / Revenue
                result["ev_revenue"] = float(enterprise_value) / float(revenue)
            
            # Store result in cache with 15-minute TTL
            with self._cache_lock:
                self._valuation_cache[ticker_upper] = _CacheEntry(result, ttl_seconds=900)
            
            # Log successful computation
            log.debug("get_valuation(%s): computed and cached", ticker_upper)
            
            # Return the result dict to caller
            return result
        
        except Exception as e:
            # Catch all exceptions (division by zero, type errors, etc.) and log
            log.warning("get_valuation(%s) failed: %s", ticker_upper, e)
            # Return None to indicate failure (graceful degradation)
            return None
    
    def invalidate_cache(self, ticker: str | None = None) -> None:
        """Invalidate price and valuation cache entries.
        
        Args:
            ticker: Specific ticker to invalidate, or None to clear all caches.
                   If provided, removes ticker from both price and valuation caches.
                   If None, clears both caches entirely (useful for testing/resets).
        """
        # Acquire lock for thread-safe cache clearing
        with self._cache_lock:
            # If specific ticker provided, remove only that ticker
            if ticker is not None:
                ticker_upper = ticker.upper()
                # Remove from price cache if present
                if ticker_upper in self._price_cache:
                    del self._price_cache[ticker_upper]
                # Remove from valuation cache if present
                if ticker_upper in self._valuation_cache:
                    del self._valuation_cache[ticker_upper]
                log.debug("Invalidated cache for %s", ticker_upper)
            # If ticker is None, clear both caches entirely
            else:
                self._price_cache.clear()
                self._valuation_cache.clear()
                log.debug("Cleared all market data caches")
