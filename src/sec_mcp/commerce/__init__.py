"""Commercial layer for the public API: keys, plans, quota, metering, billing.

Nothing in here touches financial data. It answers three questions and nothing
else: who is calling, what is their plan allowed to do, and how much have they
used. The data endpoints stay unaware of billing.

    plans.py   — the price ladder as config (overridable via API_PLANS)
    keys.py    — issuance + hashed storage + cached lookup
    usage.py   — in-memory counters, batched flush to Postgres, quota checks
    auth.py    — the FastAPI dependency that wires the three together
    billing.py — Stripe checkout + webhook → plan changes
"""
