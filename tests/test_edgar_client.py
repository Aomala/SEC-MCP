"""Tests for EDGAR client."""

import pytest

from sec_mcp.edgar_client import (
    search_companies,
    get_company,
    list_filings,
    _resolve_section,
)


def test_resolve_section_alias():
    assert _resolve_section("risk factors") == "Item 1A"
    assert _resolve_section("mda") == "Item 7"
    assert _resolve_section("MD&A") == "Item 7"
    assert _resolve_section("business") == "Item 1"


def test_resolve_section_passthrough():
    assert _resolve_section("Item 7") == "Item 7"
    assert _resolve_section("Item 1.01") == "Item 1.01"


def test_resolve_section_none():
    assert _resolve_section(None) is None


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
