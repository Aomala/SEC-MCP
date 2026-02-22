"""Tests for sec_mcp.intent_parser — parse_intent function."""

import pytest

from sec_mcp.intent_parser import parse_intent


class TestBasicFinancialQuery:
    def test_aapl_financials_tool(self):
        result = parse_intent("AAPL financials")
        assert result["tool"] == "financials"

    def test_aapl_financials_tickers(self):
        result = parse_intent("AAPL financials")
        assert "AAPL" in result["tickers"]

    def test_single_ticker_extracted(self):
        result = parse_intent("MSFT financials")
        assert "MSFT" in result["tickers"]


class TestComparison:
    def test_compare_tool(self):
        result = parse_intent("Compare AAPL vs MSFT")
        assert result["tool"] == "compare"

    def test_compare_tickers_both_present(self):
        result = parse_intent("Compare AAPL vs MSFT")
        assert "AAPL" in result["tickers"]
        assert "MSFT" in result["tickers"]

    def test_compare_versus_keyword(self):
        result = parse_intent("GOOG versus AMZN")
        assert result["tool"] == "compare"
        assert "GOOG" in result["tickers"]
        assert "AMZN" in result["tickers"]


class TestYearExtraction:
    def test_year_2023(self):
        result = parse_intent("AAPL 2023")
        assert result["year"] == 2023

    def test_year_2020(self):
        result = parse_intent("MSFT 2020 financials")
        assert result["year"] == 2020

    def test_no_year(self):
        result = parse_intent("AAPL financials")
        assert result["year"] is None


class TestFormType:
    def test_quarterly_gives_10q(self):
        result = parse_intent("AAPL quarterly")
        assert result["form_type"] == "10-Q"

    def test_default_is_10k(self):
        result = parse_intent("AAPL financials")
        assert result["form_type"] == "10-K"

    def test_10q_explicit(self):
        result = parse_intent("AAPL 10-Q")
        assert result["form_type"] == "10-Q"

    def test_quarter_reference(self):
        result = parse_intent("AAPL Q2 2023")
        assert result["form_type"] == "10-Q"


class TestSectionDetection:
    def test_risk_factors_section(self):
        result = parse_intent("AAPL risk factors")
        assert result["tool"] == "filing_text"
        assert result["section"] == "risk_factors"

    def test_mda_section(self):
        result = parse_intent("AAPL md&a")
        assert result["tool"] == "filing_text"
        assert result["section"] == "mda"

    def test_no_section_by_default(self):
        result = parse_intent("AAPL financials")
        assert result["section"] is None


class TestEntityQuery:
    def test_who_is_ceo(self):
        result = parse_intent("who is CEO of AAPL")
        assert result["tool"] == "entity"

    def test_company_info(self):
        result = parse_intent("company info TSLA")
        assert result["tool"] == "entity"

    def test_headquarters(self):
        result = parse_intent("where is MSFT headquartered")
        assert result["tool"] == "entity"


class TestQAQuery:
    def test_why_question(self):
        result = parse_intent("why did revenue decline")
        assert result["tool"] == "qa"

    def test_what_caused(self):
        result = parse_intent("what caused the margin drop")
        assert result["tool"] == "qa"

    def test_insight_query(self):
        result = parse_intent("give me insight on AAPL cash flow")
        assert result["tool"] == "qa"


class TestStopWords:
    def test_stop_words_not_tickers(self):
        result = parse_intent("show me the data")
        assert result["tickers"] == []

    def test_common_words_filtered(self):
        result = parse_intent("get the latest info")
        assert result["tickers"] == []

    def test_please_show_details(self):
        result = parse_intent("please show full details")
        assert result["tickers"] == []


class TestReturnStructure:
    def test_all_keys_present(self):
        result = parse_intent("AAPL financials")
        expected_keys = {"tool", "tickers", "year", "section", "form_type", "raw", "reasoning"}
        assert set(result.keys()) == expected_keys

    def test_raw_preserves_input(self):
        msg = "AAPL financials 2023"
        result = parse_intent(msg)
        assert result["raw"] == msg

    def test_reasoning_is_list(self):
        result = parse_intent("AAPL financials")
        assert isinstance(result["reasoning"], list)
        assert len(result["reasoning"]) > 0
