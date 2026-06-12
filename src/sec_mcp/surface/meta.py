"""Response envelope for the v2 tool surface.

Two invariants enforced here for every tool:
  - success payloads always carry meta = {source, asOf, cacheHit, latencyMs}
  - failures are always {error, code, hint} (+ meta) — never a raw traceback
"""

from __future__ import annotations

# stdlib only — this module must import fast and never fail
import functools
import logging
import time
from datetime import date, datetime, timezone

log = logging.getLogger(__name__)

# ── Stable machine-readable error codes ─────────────────────────────────────
# Clients branch on `code`; `hint` tells the caller how to fix the call.
INVALID_INPUT = "INVALID_INPUT"          # malformed/unknown parameter value
UNKNOWN_TICKER = "UNKNOWN_TICKER"        # ticker/CIK not resolvable on EDGAR
NOT_FOUND = "NOT_FOUND"                  # entity exists but no matching data
UPSTREAM_ERROR = "UPSTREAM_ERROR"        # SEC/provider failed after retries
UNAVAILABLE = "UNAVAILABLE"              # feature needs a key/dataset we lack
INTERNAL = "INTERNAL"                    # our bug — caught, never re-raised


class ToolError(Exception):
    """Raise inside a tool to emit a structured {error, code, hint} response."""

    def __init__(self, code: str, message: str, hint: str = ""):
        # Human-readable message becomes the `error` field
        super().__init__(message)
        self.code = code          # machine-readable enum above
        self.message = message    # what went wrong
        self.hint = hint          # how the caller can fix or work around it


def utcnow_iso() -> str:
    """Current UTC time, second precision, ISO-8601 — used for asOf stamps."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_meta(source: str, t0: float, cache_hit: bool, as_of: str | None = None) -> dict:
    """The mandatory meta block: where the data came from, how fresh, how fast."""
    return {
        "source": source,                                  # provider chain that answered
        "asOf": as_of or utcnow_iso(),                     # data timestamp (UTC ISO)
        "cacheHit": bool(cache_hit),                       # served from cache?
        "latencyMs": int((time.time() - t0) * 1000),       # wall time for this call
    }


def error_payload(code: str, message: str, hint: str, source: str, t0: float) -> dict:
    """Uniform failure shape — `error` first so it's unmissable in logs."""
    return {
        "error": message,                                  # human-readable description
        "code": code,                                      # stable enum for branching
        "hint": hint,                                      # actionable fix suggestion
        "meta": build_meta(source, t0, cache_hit=False),   # even errors carry meta
    }


def tool_guard(source: str):
    """Decorator wrapping every MCP tool with the response contract.

    - ToolError → structured error with the raiser's code/hint
    - any other Exception → INTERNAL with the exception type as a hint,
      logged server-side with full traceback, NEVER sent to the client raw
    - success dicts get meta injected if the tool didn't set one already
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()                               # latency clock starts
            try:
                result = fn(*args, **kwargs)               # run the actual tool
                # Tools usually return dicts; inject meta if absent
                if isinstance(result, dict) and "meta" not in result:
                    result["meta"] = build_meta(source, t0, cache_hit=False)
                return result
            except ToolError as exc:
                # Expected, structured failure — pass the tool's own context
                return error_payload(exc.code, exc.message, exc.hint, source, t0)
            except Exception as exc:                       # noqa: BLE001 — boundary
                # Unexpected bug: full trace to server log, sanitized to client
                log.exception("tool %s crashed", fn.__name__)
                return error_payload(
                    INTERNAL,
                    f"Internal error in {fn.__name__}: {type(exc).__name__}",
                    "This is a server-side bug — retry once; if it persists, "
                    "report the tool name and arguments.",
                    source, t0,
                )
        return wrapper
    return decorate


# ── Validation helpers (raise ToolError so failures stay structured) ────────

def require_ticker(value: str | None, param: str = "ticker") -> str:
    """Validate and normalize a ticker/CIK string (uppercased, stripped)."""
    # Reject missing / non-string / empty values up front
    if not value or not isinstance(value, str) or not value.strip():
        raise ToolError(INVALID_INPUT, f"'{param}' is required and must be a non-empty string.",
                        f"Pass a ticker like 'AAPL' or a CIK like '0000320193' as {param}.")
    cleaned = value.strip().upper()
    # Tickers/CIKs are short — anything longer is almost certainly a mistake
    if len(cleaned) > 12:
        raise ToolError(INVALID_INPUT, f"'{param}' value {cleaned[:20]!r} is too long to be a ticker or CIK.",
                        "Use the exchange ticker symbol (e.g. 'BRK.B') or the numeric SEC CIK.")
    return cleaned


def parse_iso_date(value, param: str):
    """Parse 'YYYY-MM-DD' → datetime.date, or None when value is None."""
    # Optional parameter — None passes through untouched
    if value is None:
        return None
    # Accept date objects directly (already valid)
    if isinstance(value, date):
        return value
    try:
        # Strict ISO format keeps the API predictable
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        raise ToolError(INVALID_INPUT, f"'{param}' must be an ISO date (YYYY-MM-DD), got {value!r}.",
                        f"Example: {param}='2025-01-31'.") from None


def require_choice(value, param: str, choices: tuple[str, ...], default: str | None = None) -> str:
    """Validate an enum-ish string parameter against allowed choices."""
    # Apply default when omitted
    if value is None and default is not None:
        return default
    # Normalize for case-insensitive matching
    v = str(value).strip().lower()
    if v not in choices:
        raise ToolError(INVALID_INPUT, f"'{param}' must be one of {list(choices)}, got {value!r}.",
                        f"Example: {param}='{choices[0]}'.")
    return v


def require_pos_int(value, param: str, default: int, lo: int = 1, hi: int = 500) -> int:
    """Validate an integer parameter within [lo, hi]; coerce numeric strings."""
    # Apply default when omitted
    if value is None:
        return default
    try:
        n = int(value)                                     # coerce "10" → 10
    except (TypeError, ValueError):
        raise ToolError(INVALID_INPUT, f"'{param}' must be an integer, got {value!r}.",
                        f"Example: {param}={default}.") from None
    # Clamp errors instead of silently truncating — caller should know
    if not (lo <= n <= hi):
        raise ToolError(INVALID_INPUT, f"'{param}' must be between {lo} and {hi}, got {n}.",
                        f"Example: {param}={default}.")
    return n


def require_number(value, param: str):
    """Validate an optional numeric filter value (int or float)."""
    # Optional — None passes through
    if value is None:
        return None
    # bool is an int subclass — explicitly reject (True is not a threshold)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            return float(value)                            # coerce "25.5" → 25.5
        except (TypeError, ValueError):
            raise ToolError(INVALID_INPUT, f"'{param}' must be a number, got {value!r}.",
                            f"Example: {param}=25.0.") from None
    return float(value)
