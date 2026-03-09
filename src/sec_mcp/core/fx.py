"""Foreign exchange rate lookups for currency conversion.

Uses FMP API (free tier supports forex) to convert foreign-currency
XBRL financials to USD. Falls back gracefully if FMP key is missing
or API is unreachable.

Cache: in-memory with 1-hour TTL (FX rates don't change that fast for
annual/quarterly financial data).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

# In-memory cache: {pair: (rate, timestamp)}
_fx_cache: dict[str, tuple[float, float]] = {}
_FX_TTL = 3600  # 1 hour


def get_fx_rate(from_currency: str, to_currency: str = "USD") -> float | None:
    """Get exchange rate from `from_currency` to `to_currency`.

    Returns the multiplier: 1 unit of from_currency = rate * to_currency.
    Returns None if unavailable. Returns 1.0 if same currency.
    """
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    if from_currency == to_currency:
        return 1.0

    pair = f"{from_currency}{to_currency}"

    # Check cache
    if pair in _fx_cache:
        rate, ts = _fx_cache[pair]
        if time.time() - ts < _FX_TTL:
            return rate

    # Try FMP API
    rate = _fetch_fmp(from_currency, to_currency)
    if rate is not None:
        _fx_cache[pair] = (rate, time.time())
        return rate

    # Fallback: try inverse
    inverse_pair = f"{to_currency}{from_currency}"
    inv_rate = _fetch_fmp(to_currency, from_currency)
    if inv_rate is not None and inv_rate > 0:
        rate = 1.0 / inv_rate
        _fx_cache[pair] = (rate, time.time())
        return rate

    log.warning("Could not fetch FX rate for %s/%s", from_currency, to_currency)
    return None


def _fetch_fmp(from_ccy: str, to_ccy: str) -> float | None:
    """Fetch rate from Financial Modeling Prep API."""
    try:
        from sec_mcp.config import get_config
        api_key = get_config().fmp_api_key
        if not api_key:
            return _fetch_fallback(from_ccy, to_ccy)

        url = f"https://financialmodelingprep.com/api/v3/quote/{from_ccy}{to_ccy}=X"
        resp = requests.get(url, params={"apikey": api_key}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            price = data[0].get("price")
            if price and price > 0:
                log.info("FX rate %s→%s = %.6f (FMP)", from_ccy, to_ccy, price)
                return float(price)
    except Exception as exc:
        log.debug("FMP FX lookup failed: %s", exc)

    return _fetch_fallback(from_ccy, to_ccy)


def _fetch_fallback(from_ccy: str, to_ccy: str) -> float | None:
    """Fallback: use exchangerate.host (no API key needed)."""
    try:
        url = f"https://open.er-api.com/v6/latest/{from_ccy}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            rates = data.get("rates", {})
            rate = rates.get(to_ccy)
            if rate and rate > 0:
                log.info("FX rate %s→%s = %.6f (er-api fallback)", from_ccy, to_ccy, rate)
                return float(rate)
    except Exception as exc:
        log.debug("Fallback FX lookup failed: %s", exc)
    return None


def detect_currency(facts_df: Any) -> str:
    """Detect the primary reporting currency from an XBRL facts DataFrame.

    Looks at the 'units' column and finds the most common ISO currency code.
    Returns 'USD' if undetermined.
    """
    if facts_df is None or facts_df.empty:
        return "USD"

    if "units" not in facts_df.columns:
        return "USD"

    # Known ISO currency codes that appear in XBRL units
    iso_currencies = {
        "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD", "KRW",
        "HKD", "SGD", "TWD", "INR", "BRL", "MXN", "ZAR", "SEK", "NOK",
        "DKK", "ILS", "NZD", "THB", "PHP", "MYR", "IDR", "RUB", "TRY",
        "PLN", "CZK", "HUF", "CLP", "COP", "PEN", "ARS", "SAR", "AED",
        "QAR", "KWD", "BHD", "OMR", "EGP", "NGN", "KES",
        # IFRS variants sometimes use iso4217 prefix
    }

    units = facts_df["units"].dropna().astype(str)
    # Filter to only currency units (not "shares", "pure", etc.)
    currency_units = units[units.isin(iso_currencies)]

    if currency_units.empty:
        # Try stripping iso4217: prefix
        cleaned = units.str.replace("iso4217:", "", regex=False)
        currency_units = cleaned[cleaned.isin(iso_currencies)]

    if currency_units.empty:
        return "USD"

    most_common = currency_units.value_counts().index[0]
    return most_common


def convert_metrics_to_usd(
    metrics: dict[str, float | None],
    reporting_currency: str,
) -> tuple[dict[str, float | None], float | None]:
    """Convert all non-None numeric metrics from reporting_currency to USD.

    Returns (converted_metrics, fx_rate_used).
    If conversion fails or currency is already USD, returns original metrics unchanged.

    Skips ratio-like metrics (margins, per-share, shares) that shouldn't be converted.
    """
    if reporting_currency.upper() == "USD":
        return metrics, 1.0

    rate = get_fx_rate(reporting_currency, "USD")
    if rate is None:
        log.warning("No FX rate for %s→USD, returning unconverted", reporting_currency)
        return metrics, None

    # Metrics that should NOT be currency-converted
    skip_keys = {
        "eps_basic", "eps_diluted", "shares_outstanding", "book_value_per_share",
        "reit_occupancy", "insurance_combined_ratio", "bank_efficiency",
        "bank_net_interest_margin", "fintech_take_rate",
        "gross_margin", "operating_margin", "net_margin",
        "return_on_assets", "return_on_equity", "current_ratio",
        "debt_to_equity", "debt_to_assets", "ebitda_margin",
        "fcf_margin", "ocf_to_net_income",
        "retail_same_store", "retail_store_count",
    }

    converted = {}
    for key, val in metrics.items():
        if val is None or key in skip_keys:
            converted[key] = val
        else:
            converted[key] = val * rate

    # EPS should be converted (it's per-share price, not a ratio)
    for eps_key in ("eps_basic", "eps_diluted", "book_value_per_share"):
        if metrics.get(eps_key) is not None:
            converted[eps_key] = metrics[eps_key] * rate

    log.info("Converted %s→USD at rate %.6f (%d metrics)", reporting_currency, rate, len(converted))
    return converted, rate
