"""Integration tests for MCP server tools."""

import pytest


@pytest.mark.integration
def test_search_company_tool():
    from sec_mcp.server import search_company
    results = search_company("AAPL")
    assert len(results) >= 1
    assert results[0]["cik"] == 320193


@pytest.mark.integration
def test_get_filing_list_tool():
    from sec_mcp.server import get_filing_list
    results = get_filing_list("AAPL", form_type="10-K", limit=2)
    assert len(results) <= 2
    assert all(r["form_type"] == "10-K" for r in results)


@pytest.mark.integration
@pytest.mark.slow
def test_analyze_sentiment_direct_text():
    from sec_mcp.server import analyze_sentiment
    result = analyze_sentiment(text="Revenue grew strongly with expanding margins and record profits.")
    assert result["overall_label"] in ("positive", "negative", "neutral")
    assert result["num_chunks"] >= 1
