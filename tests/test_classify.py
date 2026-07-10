"""Tests for sec_mcp.classify — GICS sector/industry classification (offline)."""

from sec_mcp.classify import (
    INDUSTRY_SECTOR,
    SECTOR_ETF,
    SECTORS,
    classify,
    sector_etf,
    sector_for,
)


class TestTaxonomy:
    def test_eleven_gics_sectors(self):
        assert len(SECTORS) == 11
        # Names match the SPDR/UI tiles
        for s in ("Technology", "Consumer Discretionary", "Financials",
                  "Communication Services", "Real Estate", "Health Care",
                  "Industrials", "Utilities", "Consumer Staples", "Materials", "Energy"):
            assert s in SECTORS

    def test_every_sector_has_an_etf(self):
        for s in SECTORS:
            assert SECTOR_ETF[s].startswith("XL")

    def test_every_industry_maps_to_a_real_sector(self):
        for industry, sector in INDUSTRY_SECTOR.items():
            assert sector in SECTORS, f"{industry} → unknown sector {sector}"

    def test_sector_etf_helper(self):
        assert sector_etf("Technology") == "XLK"
        assert sector_etf("Nonexistent") is None


class TestTickerOverride:
    # The GICS↔SIC mismatches that pure SIC gets wrong — must be handled by override
    def test_gics_sic_mismatches(self):
        assert classify(ticker="AMZN").sector == "Consumer Discretionary"
        assert classify(ticker="GOOGL").sector == "Communication Services"
        assert classify(ticker="META").sector == "Communication Services"
        assert classify(ticker="NFLX").sector == "Communication Services"
        assert classify(ticker="TSLA").sector == "Consumer Discretionary"
        assert classify(ticker="WMT").sector == "Consumer Staples"
        assert classify(ticker="COST").sector == "Consumer Staples"

    def test_override_beats_sic(self):
        # Even with a conflicting SIC, the ticker override wins
        assert classify(sic_code=3711, ticker="AAPL").sector == "Technology"

    def test_large_cap_sectors(self):
        expect = {
            "AAPL": "Technology", "NVDA": "Technology", "JPM": "Financials",
            "BRK-B": "Financials", "UNH": "Health Care", "XOM": "Energy",
            "NEE": "Utilities", "PLD": "Real Estate", "LIN": "Materials",
            "BA": "Industrials", "T": "Communication Services", "PG": "Consumer Staples",
        }
        for tk, sec in expect.items():
            assert classify(ticker=tk).sector == sec, f"{tk} misclassified"

    def test_brk_dot_and_dash_normalize(self):
        assert classify(ticker="BRK.B").sector == "Financials"
        assert classify(ticker="brk-b").sector == "Financials"


class TestSicBase:
    def test_sic_boundaries(self):
        cases = {
            2834: "Health Care", 3674: "Technology", 7372: "Technology",
            4813: "Communication Services", 4911: "Utilities", 1311: "Energy",
            6022: "Financials", 6311: "Financials", 6798: "Real Estate",
            2080: "Consumer Staples", 5411: "Consumer Staples", 5912: "Consumer Staples",
            3711: "Consumer Discretionary", 5812: "Consumer Discretionary",
            2851: "Materials", 1040: "Materials", 3721: "Industrials",
        }
        for sic, sector in cases.items():
            assert classify(sic_code=sic).sector == sector, f"SIC {sic}"

    def test_staples_vs_discretionary_split(self):
        # The split the old 10-bucket map couldn't do
        assert classify(sic_code=2011).sector == "Consumer Staples"       # meat packing
        assert classify(sic_code=5651).sector == "Consumer Discretionary"  # clothing

    def test_restaurants_industry(self):
        assert classify(sic_code=5812).industry == "Restaurants"


class TestFallback:
    def test_never_returns_none(self):
        c = classify(sic_code=None, ticker=None)
        assert c.sector == "Other" and c.industry == "Unknown"
        c2 = classify(sic_code=9999, ticker="ZZZZ")   # unmapped SIC, unknown ticker
        assert c2.sector == "Other"

    def test_source_labels(self):
        assert classify(ticker="AAPL").source == "ticker"
        assert classify(sic_code=2834).source == "sic"
        assert classify().source == "fallback"

    def test_sector_for_helper(self):
        assert sector_for(ticker="NVDA") == "Technology"
