"""Tests for EDGAR client."""

import pytest

from sec_mcp.edgar_client import (
    search_companies,
    get_company,
    list_filings,
)


@pytest.mark.integration
def test_search_company_by_ticker():
    results = search_companies("AAPL")
    assert len(results) >= 1
    assert results[0].cik == 320193


@pytest.mark.integration
def test_get_company():
    company = get_company("MSFT")
    assert company.cik == 789019
    assert "Microsoft" in company.name


@pytest.mark.integration
def test_list_10k_filings():
    filings = list_filings("AAPL", form_type="10-K", limit=3)
    assert len(filings) <= 3
    assert all(f.form_type == "10-K" for f in filings)
