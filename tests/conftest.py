"""Shared fixtures for SEC-MCP tests."""

import pytest


@pytest.fixture
def sample_financial_metrics():
    """Sample financial metrics dict matching canonical metric keys from xbrl_mappings."""
    return {
        "revenue": 394_328_000_000,
        "net_income": 99_803_000_000,
        "total_assets": 352_583_000_000,
        "total_liabilities": 287_912_000_000,
        "stockholders_equity": 64_671_000_000,
        "operating_income": 119_437_000_000,
        "operating_cash_flow": 110_543_000_000,
        "eps_diluted": 6.13,
        "shares_outstanding": 15_943_425_000,
        "cost_of_revenue": 223_546_000_000,
        "gross_profit": 170_782_000_000,
        "cash_and_equivalents": 29_965_000_000,
        "long_term_debt": 98_959_000_000,
    }


@pytest.fixture
def sample_xbrl_facts_df():
    """Sample XBRL facts as a list of dicts (can be passed to pd.DataFrame)."""
    import pandas as pd

    rows = [
        {
            "taxonomy": "us-gaap",
            "tag": "Revenues",
            "value": 394_328_000_000,
            "unit": "USD",
            "period_end": "2023-09-30",
            "form": "10-K",
            "filed": "2023-11-03",
        },
        {
            "taxonomy": "us-gaap",
            "tag": "NetIncomeLoss",
            "value": 99_803_000_000,
            "unit": "USD",
            "period_end": "2023-09-30",
            "form": "10-K",
            "filed": "2023-11-03",
        },
        {
            "taxonomy": "us-gaap",
            "tag": "Assets",
            "value": 352_583_000_000,
            "unit": "USD",
            "period_end": "2023-09-30",
            "form": "10-K",
            "filed": "2023-11-03",
        },
        {
            "taxonomy": "us-gaap",
            "tag": "EarningsPerShareDiluted",
            "value": 6.13,
            "unit": "USD/shares",
            "period_end": "2023-09-30",
            "form": "10-K",
            "filed": "2023-11-03",
        },
    ]
    return pd.DataFrame(rows)
