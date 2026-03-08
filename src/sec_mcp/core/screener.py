"""Company screening/filtering engine — filters companies by financial metrics.

Supports filters like revenue, profitability ratios, asset ratios, market cap.
Fetches universe from SEC tickers list and extracts fundamentals via ThreadPoolExecutor.
Caches results by filter hash (30-min TTL).
Limits concurrent network requests (max 100 candidates, 10 workers).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Any

# Import existing financials extraction for batch processing
from sec_mcp.financials import extract_financials

# Import SEC client for ticker universe
from sec_mcp.sec_client import get_sec_client

# Module logger for warnings and errors
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Screener Cache Entry (simple TTL tracking)
# ═══════════════════════════════════════════════════════════════════════════

class _ScreenerCacheEntry:
    """Cache entry with timestamp and TTL for screener results.
    
    Attributes:
        value: The screener result (list of matching companies)
        timestamp: When this entry was created (datetime UTC)
        ttl_seconds: Time to live before expiration
    """
    
    def __init__(self, value: Any, ttl_seconds: int):
        """Initialize cache entry.
        
        Args:
            value: The screener results to cache
            ttl_seconds: Seconds until expiration
        """
        # Store the cached screener results
        self.value = value
        # Capture creation time in UTC
        self.timestamp = datetime.now(timezone.utc)
        # Store TTL duration
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL.
        
        Returns:
            True if expired, False if still valid
        """
        # Calculate expiration time (creation + TTL)
        expiration = self.timestamp + timedelta(seconds=self.ttl_seconds)
        # Return True if current time >= expiration time
        return datetime.now(timezone.utc) >= expiration


# ═══════════════════════════════════════════════════════════════════════════
#  Screener Class
# ═══════════════════════════════════════════════════════════════════════════

class Screener:
    """Screens companies by financial metric filters with parallel extraction.
    
    Features:
      - screen(filters, limit) → companies matching all filter conditions
        Filters: {metric: "revenue", operator: ">", value: 1000000000}
        Operators: >, <, >=, <=, ==, between
        Metrics: revenue, net_income, gross_margin, operating_margin, etc.
      - Parallel XBRL extraction via ThreadPoolExecutor (max 10 workers)
      - Candidate limiting (max 100 checked by default, configurable)
      - Result caching by filter hash (30-min TTL)
      - Thread-safe cache access
    
    IMPORTANT: This is expensive (many network calls). Use max_candidates to limit scope.
    """
    
    # Supported metric names that can be used in filters
    # Maps friendly names to XBRL metric keys in extract_financials result
    METRIC_ALIASES: dict[str, str] = {
        "revenue": "revenue",
        "net_income": "net_income",
        "gross_margin": "gross_margin",
        "operating_margin": "operating_margin",
        "net_margin": "net_margin",
        "roa": "roa",  # Return on Assets
        "roe": "roe",  # Return on Equity
        "total_assets": "total_assets",
        "market_cap": "market_cap",
    }
    
    def __init__(self):
        """Initialize Screener with empty result cache.
        
        Sets up:
          - _result_cache: dict mapping filter hash → _ScreenerCacheEntry
          - _cache_lock: threading.Lock for thread-safe access
        """
        # Cache for screener results (filter_hash → _ScreenerCacheEntry)
        # 30-minute TTL to allow periodic re-screening
        self._result_cache: dict[str, _ScreenerCacheEntry] = {}
        
        # Lock protecting cache during concurrent access
        self._cache_lock = threading.Lock()
        
        # Log initialization
        log.info("Screener initialized (cache TTL: 30 min)")
    
    def screen(
        self,
        filters: list[dict[str, Any]],
        limit: int = 50,
        max_candidates: int = 100,
        year: int | None = None,
        form_type: str = "10-K",
    ) -> list[dict[str, Any]]:
        """Screen companies by financial metric filters.
        
        Process:
          1. Check result cache (by filter hash); return if valid
          2. Get ticker universe from SEC (company_tickers.json)
          3. Extract financials for up to max_candidates in parallel (ThreadPoolExecutor)
          4. Apply all filter conditions to each company
          5. Sort results by first filter metric
          6. Cache and return up to limit results
        
        Args:
            filters: List of filter dicts. Each contains:
              - "metric": supported metric name (see METRIC_ALIASES)
              - "operator": one of ">", "<", ">=", "<=", "==", "between"
              - "value": numeric value to compare against
              - For "between": also include "value_max" key
              Example:
                [
                  {"metric": "revenue", "operator": ">=", "value": 1000000000},
                  {"metric": "net_margin", "operator": ">", "value": 0.10},
                  {"metric": "roe", "operator": "between", "value": 0.15, "value_max": 0.25},
                ]
            limit: Maximum number of results to return (default: 50)
            max_candidates: Maximum companies to check for matches (default: 100)
              Higher values = more comprehensive but slower and more network calls
              Lower values = faster but may miss matches
            year: Fiscal year to screen (None = most recent filing)
            form_type: Filing type ("10-K" annual, "10-Q" quarterly, default: "10-K")
        
        Returns:
            List of matching company dicts, each containing:
            {
              "ticker": "AAPL",
              "company_name": "Apple Inc.",
              "metrics": {  # All metrics extracted for this company
                "revenue": 383285000000,
                "net_income": 93736000000,
                "gross_margin": 0.456,
                ...
              },
              "ratios": {  # Computed ratios
                "roa": 0.227,
                "roe": 0.732,
                ...
              },
              "matched": True,  # Flag indicating this passed all filters
            }
            
            Result is sorted by first filter metric (ascending if ">" operator,
            descending if "<" operator, etc.) and limited to limit parameter.
            Returns [] if no matches found.
        
        Raises:
            ValueError: If filter metric not supported, operator invalid, or required
                       keys missing. Message indicates which filter caused the error.
            No network exceptions are caught — they propagate to caller.
        """
        # ─────────────────────────────────────────────────────────────────
        # Validate filters before proceeding
        # ─────────────────────────────────────────────────────────────────
        
        # Check that filters list is not empty
        if not filters:
            log.warning("screen: empty filters list")
            return []
        
        # Validate each filter dict
        for i, f in enumerate(filters):
            # Check required keys are present
            if "metric" not in f:
                raise ValueError(f"Filter {i}: missing 'metric' key")
            if "operator" not in f:
                raise ValueError(f"Filter {i}: missing 'operator' key")
            if "value" not in f:
                raise ValueError(f"Filter {i}: missing 'value' key")
            
            # Check metric is recognized
            metric_name = f.get("metric")
            if metric_name not in self.METRIC_ALIASES:
                raise ValueError(f"Filter {i}: unknown metric '{metric_name}'")
            
            # Check operator is valid
            operator = f.get("operator")
            if operator not in [">", "<", ">=", "<=", "==", "between"]:
                raise ValueError(f"Filter {i}: invalid operator '{operator}'")
            
            # Check "between" operator has both bounds
            if operator == "between":
                if "value_max" not in f:
                    raise ValueError(f"Filter {i}: 'between' requires 'value_max' key")
        
        # ─────────────────────────────────────────────────────────────────
        # Check result cache before doing expensive extraction
        # ─────────────────────────────────────────────────────────────────
        
        # Create a hash of the filter list to use as cache key
        # This allows identical screening requests to reuse cached results
        filter_json = str(sorted([(f["metric"], f["operator"], f.get("value")) for f in filters]))
        filter_hash = hashlib.md5(filter_json.encode()).hexdigest()
        
        # Acquire lock and check cache
        with self._cache_lock:
            # Check if we have a cached result for these filters
            if filter_hash in self._result_cache:
                entry = self._result_cache[filter_hash]
                # If cache entry still valid (not expired), return it
                if not entry.is_expired():
                    log.info("screen: cache hit for filters (returning %d results)",
                             len(entry.value))
                    # Return cached results, limited to the requested limit
                    return entry.value[:limit]
                # If expired, remove from cache
                else:
                    del self._result_cache[filter_hash]
        
        # ─────────────────────────────────────────────────────────────────
        # Cache miss — fetch ticker universe and extract financials
        # ─────────────────────────────────────────────────────────────────
        
        try:
            # Get SEC client singleton
            client = get_sec_client()
            
            # Fetch all company tickers from SEC (company_tickers.json)
            # This returns a list of CompanyInfo objects with ticker, cik, name
            all_tickers = client.get_all_tickers()
        except Exception as e:
            # Failed to fetch ticker universe (network error, etc.)
            log.error("screen: failed to fetch ticker universe: %s", e)
            return []
        
        # Limit to max_candidates to avoid excessive network calls
        # Take first N tickers (can be randomized if needed)
        candidates = all_tickers[:max_candidates]
        
        log.info("screen: screening %d candidates against %d filters",
                 len(candidates), len(filters))
        
        # ─────────────────────────────────────────────────────────────────
        # Extract financials in parallel using ThreadPoolExecutor
        # ─────────────────────────────────────────────────────────────────
        
        # List to collect results
        results = []
        
        # Parallel extraction: use ThreadPoolExecutor with max 10 workers
        # Each worker calls extract_financials(ticker) in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit extraction jobs for all candidates
            futures = {}
            for ticker_info in candidates:
                # Get ticker symbol
                ticker = ticker_info.ticker
                
                # Submit extraction job; map future → ticker for tracking
                future = executor.submit(
                    extract_financials,
                    ticker,
                    year=year,
                    form_type=form_type,
                )
                futures[future] = ticker
            
            # Collect results as they complete (not necessarily in order)
            for future in as_completed(futures):
                ticker = futures[future]
                
                try:
                    # Get extraction result for this ticker
                    data = future.result()
                    
                    # Skip if extraction failed or returned None
                    if data is None or data.get("error"):
                        continue
                    
                    # Extract metrics and build result dict
                    metrics = data.get("metrics", {})
                    
                    # Check if this company passes all filters
                    passes_all = True
                    for f in filters:
                        # Get the metric value for this filter
                        metric_key = self.METRIC_ALIASES[f["metric"]]
                        metric_value = metrics.get(metric_key)
                        
                        # Skip if metric data not available
                        if metric_value is None:
                            passes_all = False
                            break
                        
                        # Apply filter condition
                        operator = f["operator"]
                        filter_value = f["value"]
                        
                        if operator == ">":
                            condition = metric_value > filter_value
                        elif operator == "<":
                            condition = metric_value < filter_value
                        elif operator == ">=":
                            condition = metric_value >= filter_value
                        elif operator == "<=":
                            condition = metric_value <= filter_value
                        elif operator == "==":
                            condition = abs(metric_value - filter_value) < 0.001  # Float tolerance
                        elif operator == "between":
                            condition = filter_value <= metric_value <= f["value_max"]
                        else:
                            condition = False  # Should not reach here (validated above)
                        
                        # If any filter fails, company doesn't match
                        if not condition:
                            passes_all = False
                            break
                    
                    # If company passed all filters, add to results
                    if passes_all:
                        result_dict = {
                            "ticker": ticker,
                            "company_name": data.get("company_name", ""),
                            "metrics": metrics,
                            "ratios": data.get("ratios", {}),
                            "matched": True,
                        }
                        results.append(result_dict)
                        
                except Exception as e:
                    # Individual extraction failed; log and continue
                    log.warning("screen: extraction failed for %s: %s", ticker, e)
        
        # ─────────────────────────────────────────────────────────────────
        # Sort results by first filter metric
        # ─────────────────────────────────────────────────────────────────
        
        # Get the primary sorting metric (first filter)
        first_filter = filters[0]
        sort_metric = self.METRIC_ALIASES[first_filter["metric"]]
        sort_operator = first_filter["operator"]
        
        # Determine sort order: descending if ">", ascending if "<", etc.
        # This makes results most relevant to the filter naturally ordered
        reverse_sort = sort_operator in [">", ">="]  # Higher values first
        
        # Sort results by the metric
        results.sort(
            key=lambda x: x["metrics"].get(sort_metric, 0),
            reverse=reverse_sort,
        )
        
        # ─────────────────────────────────────────────────────────────────
        # Cache and return results
        # ─────────────────────────────────────────────────────────────────
        
        # Limit to requested number of results
        final_results = results[:limit]
        
        # Cache the results (30-minute TTL)
        with self._cache_lock:
            self._result_cache[filter_hash] = _ScreenerCacheEntry(
                final_results, ttl_seconds=1800
            )
        
        # Log results and return
        log.info("screen: found %d matches (cached %d, returning %d)",
                 len(results), len(results), len(final_results))
        return final_results
