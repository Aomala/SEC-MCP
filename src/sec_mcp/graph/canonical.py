"""Canonical concept layer: metric key → ordered XBRL concept seeds.

The seeds come from the hand-curated lists in xbrl_mappings.py (they ARE the
canonical dictionary — source='manual'); the graph resolver uses them to
recognise which node in a filing's calculation tree expresses each canonical
metric. Structural tree position arbitrates when several seeds appear.

# Decision: reuse xbrl_mappings lists instead of vendoring edgartools'
# gaap_mappings.json for now — the per-filing tree already provides the
# cross-company arbitration that the learned mappings approximate. The
# learned file can be merged into concept_mappings later without touching
# this API.
"""

from __future__ import annotations

from sec_mcp.xbrl_mappings import (
    CONCEPT_MAP,
    IndustryClass,
    get_revenue_concepts,
)

# canonical key → statement the concept lives on
STATEMENT_OF: dict[str, str] = {
    "revenue": "income", "cost_of_revenue": "income", "gross_profit": "income",
    "operating_income": "income", "operating_expenses": "income",
    "sga_expense": "income", "rd_expense": "income", "interest_expense": "income",
    "income_tax_expense": "income", "income_before_tax": "income",
    "net_income": "income", "eps_basic": "income", "eps_diluted": "income",
    "depreciation_amortization": "income",
    "total_assets": "balance", "total_liabilities": "balance",
    "stockholders_equity": "balance", "current_assets": "balance",
    "current_liabilities": "balance", "cash_and_equivalents": "balance",
    "long_term_debt": "balance", "short_term_debt": "balance",
    "inventory": "balance", "accounts_receivable": "balance",
    "accounts_payable": "balance", "goodwill": "balance",
    "retained_earnings": "balance", "shares_outstanding": "balance",
    "operating_cash_flow": "cashflow", "investing_cash_flow": "cashflow",
    "financing_cash_flow": "cashflow", "capital_expenditures": "cashflow",
    "dividends_paid": "cashflow", "shares_repurchased": "cashflow",
    "share_based_compensation": "cashflow",
}


def seed_concepts(canonical_key: str, industry: IndustryClass) -> list[str]:
    """Ordered XBRL tag names that can express this canonical concept."""
    if canonical_key == "revenue":
        entries = get_revenue_concepts(industry)
    else:
        entries = CONCEPT_MAP.get(canonical_key, [])
    return [e.xbrl_concept for e in entries]


def statement_of(canonical_key: str) -> str:
    return STATEMENT_OF.get(canonical_key, "income")
