"""Plan catalogue — the price ladder, as DATA.

Pricing is the thing most likely to change and the thing you least want to
redeploy for, so plans live here as a plain dict and can be overridden at
runtime with the API_PLANS env var (a JSON object, deep-merged over these
defaults). Changing a price or a quota is a config edit, not a code change.

Quota model, per key:
  daily_calls  — hard ceiling per UTC day; over it → 429 quota_exceeded
  rpm          — burst ceiling per rolling minute → 429 rate_limited
  scopes       — which endpoint families the key may call
  bulk         — may hit /v1/bulk/* (the whole-universe dumps)
  history_years— how far back statement history may go; None = unlimited

`scopes` is checked against the SCOPES a route declares, so adding a paid-only
endpoint is one decorator argument rather than a branch inside the handler.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy

log = logging.getLogger(__name__)

# Every scope the API knows about. A plan carrying "*" gets all of them.
ALL_SCOPES = (
    "search",        # company/ticker lookup
    "fundamentals",  # metrics, ratios, statements
    "filings",       # filing lists, sections, full text
    "market",        # quotes, price history, news, corporate actions
    "ownership",     # insider (Form 4), 13F/13D-G
    "segments",      # product/service + geographic revenue
    "screen",        # screener, peers, comps
    "bulk",          # whole-universe dumps
)

DEFAULT_PLANS: dict[str, dict] = {
    # Anonymous / no key. Deliberately usable so the docs page is live, but
    # small enough that anyone building something real must get a key.
    "anon": {
        "label": "Anonymous",
        "price_usd_month": 0,
        "daily_calls": 50,
        "rpm": 10,
        "scopes": ["search", "fundamentals", "market"],
        "bulk": False,
        "history_years": 3,
    },
    # Free, keyed. The signup incentive: 10x the anonymous budget.
    "free": {
        "label": "Free",
        "price_usd_month": 0,
        "daily_calls": 500,
        "rpm": 30,
        "scopes": ["search", "fundamentals", "market", "filings", "segments"],
        "bulk": False,
        "history_years": 5,
    },
    # The self-serve tier — an indie dev, a quant, an AI agent builder.
    "starter": {
        "label": "Starter",
        "price_usd_month": 49,
        "daily_calls": 10_000,
        "rpm": 120,
        "scopes": ["*"],
        "bulk": False,
        "history_years": 10,
    },
    # The one that pays the bills: bulk access is the differentiator.
    "pro": {
        "label": "Pro",
        "price_usd_month": 199,
        "daily_calls": 100_000,
        "rpm": 600,
        "scopes": ["*"],
        "bulk": True,
        "history_years": None,
    },
    # Negotiated. Quotas here are the default before a contract overrides them.
    "business": {
        "label": "Business",
        "price_usd_month": 999,
        "daily_calls": 1_000_000,
        "rpm": 3_000,
        "scopes": ["*"],
        "bulk": True,
        "history_years": None,
    },
    # Internal: Fineas's own edge functions and the MCP server. Never sold.
    "internal": {
        "label": "Internal",
        "price_usd_month": 0,
        "daily_calls": 10_000_000,
        "rpm": 100_000,
        "scopes": ["*"],
        "bulk": True,
        "history_years": None,
    },
}

_plans: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _plans
    if _plans is not None:
        return _plans
    plans = deepcopy(DEFAULT_PLANS)
    raw = os.environ.get("API_PLANS", "").strip()
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                for name, cfg in override.items():
                    if isinstance(cfg, dict):
                        plans.setdefault(name, {}).update(cfg)
                log.info("Plan overrides applied from API_PLANS: %s",
                         ", ".join(sorted(override)))
        except (ValueError, TypeError) as exc:
            # A malformed override must not take the API down — log and ship
            # the defaults, which are always a valid ladder.
            log.warning("API_PLANS is not valid JSON, using defaults: %s", exc)
    _plans = plans
    return _plans


def get_plan(name: str | None) -> dict:
    """Plan config by name; unknown names fall back to `free`, never to error."""
    plans = _load()
    return plans.get((name or "").lower()) or plans["free"]


def all_plans() -> dict[str, dict]:
    return _load()


def public_catalogue() -> list[dict]:
    """The ladder as the pricing page / docs should show it (sellable tiers only)."""
    plans = _load()
    return [
        {
            "plan": name,
            "label": cfg.get("label", name.title()),
            "priceUsdMonth": cfg.get("price_usd_month", 0),
            "dailyCalls": cfg.get("daily_calls"),
            "requestsPerMinute": cfg.get("rpm"),
            "bulkAccess": bool(cfg.get("bulk")),
            "historyYears": cfg.get("history_years"),
            "scopes": (list(ALL_SCOPES) if "*" in cfg.get("scopes", [])
                       else cfg.get("scopes", [])),
        }
        for name, cfg in plans.items()
        if name not in ("anon", "internal")
    ]


def plan_allows(plan: dict, scope: str) -> bool:
    scopes = plan.get("scopes") or []
    return "*" in scopes or scope in scopes


def reset_cache() -> None:
    """Test hook — drop the memoized catalogue so env overrides re-read."""
    global _plans
    _plans = None
