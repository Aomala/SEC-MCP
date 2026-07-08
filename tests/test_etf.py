"""Tests for the ETF profile layer (core/etf.py) — pure, no network."""

from sec_mcp.core.etf import ETF_SEED, default_peers, merge_etf_profile


class TestSeedIntegrity:
    def test_all_seeds_complete(self):
        for tk, seed in ETF_SEED.items():
            assert seed["name"], tk
            assert seed["issuer"], tk
            assert isinstance(seed["expense_ratio"], float), tk
            assert 0 < seed["expense_ratio"] < 1.5, f"{tk} expense ratio looks wrong (percent units expected)"
            assert seed["asset_class"] in {"Equity", "Fixed Income", "Commodity", "Real Estate"}, tk
            assert seed["inception_date"], tk
            assert seed["holdings_count"] and seed["holdings_count"] >= 1, tk
            assert seed["peers"], tk

    def test_seed_peers_do_not_include_self(self):
        for tk, seed in ETF_SEED.items():
            assert tk not in seed["peers"], tk

    def test_spy_facts(self):
        spy = ETF_SEED["SPY"]
        assert spy["issuer"] == "State Street Global Advisors"
        assert spy["expense_ratio"] == 0.0945
        assert spy["asset_class"] == "Equity"


class TestMergeEtfProfile:
    def test_seed_only_no_polygon(self):
        p = merge_etf_profile("SPY", None)
        assert p is not None
        assert p["name"] == "SPDR S&P 500 ETF Trust"
        assert p["issuer"] == "State Street Global Advisors"
        assert p["expenseRatio"] == 0.0945
        assert p["aum"] is None  # no Polygon, no price → no fake AUM
        assert p["peers"] == ["VOO", "IVV", "VTI", "QQQ", "DIA"]
        assert p["profileSource"] == "seed"

    def test_polygon_market_cap_becomes_aum(self):
        details = {"type": "ETF", "name": "SPDR S&P 500 ETF Trust", "market_cap": 610e9}
        p = merge_etf_profile("SPY", details)
        assert p["aum"] == 610e9
        assert p["aumSource"] == "polygon_market_cap"
        assert p["profileSource"] == "seed+polygon"

    def test_shares_times_price_fallback(self):
        details = {"type": "ETF", "name": "Some ETF", "weighted_shares_outstanding": 1e9}
        p = merge_etf_profile("SPY", details, price=500.0)
        assert p["aum"] == 500e9
        assert p["aumSource"] == "shares_x_price"

    def test_non_seeded_fund_type_detected(self):
        details = {"type": "ETV", "name": "Random Commodity Trust", "market_cap": 3e9,
                   "list_date": "2015-02-03"}
        p = merge_etf_profile("XYZQ", details)
        assert p is not None
        assert p["name"] == "Random Commodity Trust"
        assert p["expenseRatio"] is None  # not seeded — no invented numbers
        assert p["inceptionDate"] == "2015-02-03"
        assert len(p["peers"]) > 0

    def test_stock_returns_none(self):
        details = {"type": "CS", "name": "Apple Inc.", "market_cap": 3e12}
        assert merge_etf_profile("AAPL", details) is None

    def test_unknown_ticker_no_details_returns_none(self):
        assert merge_etf_profile("ZZZZ", None) is None

    def test_no_nan_leaks(self):
        details = {"type": "ETF", "market_cap": float("nan")}
        p = merge_etf_profile("SPY", details)
        assert p["aum"] is None


class TestDefaultPeers:
    def test_excludes_self(self):
        assert "VOO" not in default_peers("Equity", "VOO")

    def test_fixed_income_pool(self):
        peers = default_peers("Fixed Income", "ZROZ")
        assert "BND" in peers and "AGG" in peers

    def test_unknown_class_defaults_to_equity(self):
        assert default_peers(None, "XXXX") == ["VOO", "IVV", "VTI", "QQQ", "DIA"]
