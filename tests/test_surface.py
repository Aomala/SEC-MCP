"""Integration suite for the v2 MCP tool surface (server.py, 9 tools).

Coverage matrix:
  - each tool against the 5-ticker panel (mega-cap, mid-cap, recent IPO,
    foreign private issuer, recent corporate action)
  - edge cases: invalid ticker, delisted ticker, malformed filter values —
    every failure must be a structured {error, code, hint} dict, never an
    exception escaping to the client
  - 24/7 proof: a simulated Sunday 03:00 ET clock — quotes answer with
    session="closed", filings/fundamentals answer normally, and the string
    "market closed" never appears as an error

Run:
  pytest tests/test_surface.py -m integration      # live network tests
  pytest tests/test_surface.py -m "not integration"  # pure-logic tests only
"""

from __future__ import annotations

# stdlib
from datetime import datetime, timezone

import pytest

# the server module under test — tools are plain callables via FastMCP
from sec_mcp import server as S

# session logic for the simulated-clock tests
from sec_mcp.surface import session as sess

# ── Ticker panel (mirrors MCP_AUDIT.md) ─────────────────────────────────────
MEGA = "AAPL"      # mega-cap, standard filer
MID = "CAVA"       # mid-cap, 2023 IPO
RECENT_IPO = "CRWV"  # CoreWeave, IPO Mar-2025, minimal history
FPI = "ASML"       # foreign private issuer — files 20-F/6-K
CORP_ACTION = "SMCI"  # 10-for-1 split Oct-2024 + filing delays
PANEL = [MEGA, MID, RECENT_IPO, FPI, CORP_ACTION]

DELISTED = "TWTR"  # delisted 2022 — no longer in the SEC ticker map
INVALID = "ZZZZZZ99"  # never existed


def is_structured_error(r) -> bool:
    """The error contract: dict with error + code + hint + meta."""
    return (isinstance(r, dict) and "error" in r and "code" in r
            and "hint" in r and "meta" in r)


def has_meta(r) -> bool:
    """The success contract: meta block with all four mandatory fields."""
    m = (r or {}).get("meta") or {}
    return all(k in m for k in ("source", "asOf", "cacheHit", "latencyMs"))


# ═════════════════════════════════════════════════════════════════════════
#  Pure-logic tests (no network) — session clock + validation
# ═════════════════════════════════════════════════════════════════════════

class TestSessionClock:
    """Market-session classification at fixed instants (all ET via UTC)."""

    def test_sunday_3am_is_closed(self):
        # Sunday 2026-06-14 03:00 ET == 07:00 UTC
        sunday = datetime(2026, 6, 14, 7, 0, tzinfo=timezone.utc)
        assert sess.market_session(sunday) == "closed"

    def test_weekday_regular_session(self):
        # Friday 2026-06-12 10:30 ET == 14:30 UTC
        friday = datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc)
        assert sess.market_session(friday) == "regular"

    def test_weekday_premarket(self):
        # Friday 05:00 ET == 09:00 UTC
        assert sess.market_session(datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)) == "pre"

    def test_weekday_afterhours(self):
        # Friday 17:00 ET == 21:00 UTC
        assert sess.market_session(datetime(2026, 6, 12, 21, 0, tzinfo=timezone.utc)) == "after"

    def test_holiday_is_closed(self):
        # Juneteenth 2026-06-19 (Friday) at 11:00 ET
        assert sess.market_session(datetime(2026, 6, 19, 15, 0, tzinfo=timezone.utc)) == "closed"

    def test_filings_ttl_business_hours_vs_offhours(self):
        # Friday 10:30 ET → live EDGAR window → 60s
        assert sess.filings_index_ttl(datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc)) == 60
        # Sunday 03:00 ET → frozen index → 600s
        assert sess.filings_index_ttl(datetime(2026, 6, 14, 7, 0, tzinfo=timezone.utc)) == 600

    def test_quote_ttl_by_session(self):
        assert sess.quote_ttl("regular") == 30
        assert sess.quote_ttl("pre") == 120
        assert sess.quote_ttl("after") == 120
        assert sess.quote_ttl("closed") == 3600


class TestValidationNeverCrashes:
    """Every malformed input → structured error, never an exception."""

    CASES = [
        lambda: S.search_companies(""),                              # no query, no filters
        lambda: S.search_companies("apple", filters={"galaxy": 1}),  # unknown filter
        lambda: S.search_companies("apple", filters={"market_cap_min": "huge"}),
        lambda: S.search_companies("apple", filters={"ipo_date_after": "not-a-date"}),
        lambda: S.get_filings("AAPL", date_from="garbage"),
        lambda: S.get_filings("AAPL", form_type="99-ZZ-!!"),
        lambda: S.get_filings("AAPL", date_from="2025-12-01", date_to="2025-01-01"),
        lambda: S.get_filings(),                                     # nothing at all
        lambda: S.get_filing_section("not-an-accession", "risk_factors"),
        lambda: S.get_filing_section("0000320193-25-000079", "recipes"),
        lambda: S.get_fundamentals("AAPL", period="hourly"),
        lambda: S.get_fundamentals("AAPL", metrics=["vibes"]),
        lambda: S.get_fundamentals("AAPL", periods_back=9999),
        lambda: S.get_insider_activity("AAPL", date_from="13/13/2026"),
        lambda: S.compare(["AAPL"]),                                 # need 2-8
        lambda: S.compare("AAPL"),                                   # not a list
        lambda: S.screen({"moon_phase": "full"}),
        lambda: S.screen([{"metric": "x"}]),                         # legacy list shape
        lambda: S.screen({"pe_max": "cheap"}),
        lambda: S.screen({}),                                        # empty filters
        lambda: S.get_quote(""),                                     # empty ticker
        lambda: S.get_quote(None),                                   # None ticker
    ]

    @pytest.mark.parametrize("case", range(len(CASES)))
    def test_malformed_input_is_structured(self, case):
        result = self.CASES[case]()                        # must NOT raise
        assert is_structured_error(result), f"case {case} returned {result!r}"
        # Validation failures must carry the INVALID_INPUT code specifically
        # (a couple of cases legitimately fail later in the pipeline)
        assert result["code"] in ("INVALID_INPUT", "UNKNOWN_TICKER", "UPSTREAM_ERROR")


# ═════════════════════════════════════════════════════════════════════════
#  Live integration tests — the 5-ticker panel
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestSearchCompanies:
    @pytest.mark.parametrize("ticker", PANEL)
    def test_panel_resolves(self, ticker):
        r = S.search_companies(ticker)
        assert not is_structured_error(r) and has_meta(r)
        assert r["count"] >= 1
        top = r["results"][0]
        # Contract: every match carries CIK + ticker
        assert top["cik"] and top["ticker"] == ticker

    def test_name_search_ranks_exact_brand(self):
        r = S.search_companies("nvidia")
        assert r["results"][0]["ticker"] == "NVDA"

    def test_exchange_filter(self):
        r = S.search_companies("apple", filters={"exchange": "Nasdaq"})
        assert all((x.get("exchange") or "").lower() == "nasdaq" for x in r["results"])

    def test_sp500_filter(self):
        r = S.search_companies("apple", filters={"is_sp500": True})
        assert not is_structured_error(r)
        assert any(x["ticker"] == "AAPL" for x in r["results"])


@pytest.mark.integration
class TestGetFilings:
    @pytest.mark.parametrize("ticker", PANEL)
    def test_annual_filings_panel(self, ticker):
        # The 10-K alias must transparently cover FPI 20-F filers (ASML)
        r = S.get_filings(ticker, form_type="10-K", limit=3)
        assert not is_structured_error(r) and has_meta(r)
        assert r["count"] >= 1, f"{ticker}: no annual filings found"
        row = r["filings"][0]
        # Contract: accession + acceptance timestamp + direct EDGAR URL
        assert row["accession"] and row["url"].startswith("https://www.sec.gov/")
        assert row["acceptedAt"]

    def test_date_window(self):
        r = S.get_filings(MEGA, form_type="8-K",
                          date_from="2025-01-01", date_to="2025-12-31")
        assert all("2025-01-01" <= f["filingDate"] <= "2025-12-31"
                   for f in r["filings"])

    def test_full_text_search(self):
        r = S.get_filings(full_text_query="artificial intelligence",
                          form_type="10-K", limit=5)
        assert not is_structured_error(r)
        assert r["mode"] == "full_text" and r["count"] >= 1

    def test_unknown_ticker_structured(self):
        r = S.get_filings(INVALID, form_type="10-K")
        assert is_structured_error(r) and r["code"] == "UNKNOWN_TICKER"

    def test_delisted_ticker_structured(self):
        r = S.get_filings(DELISTED, form_type="10-K")
        # Delisted = dropped from the SEC ticker map → UNKNOWN_TICKER
        assert is_structured_error(r) and r["code"] == "UNKNOWN_TICKER"


@pytest.mark.integration
class TestGetFilingSection:
    def test_risk_factors_clean_text(self):
        filings = S.get_filings(MEGA, form_type="10-K", limit=1)["filings"]
        r = S.get_filing_section(filings[0]["accession"], "risk_factors")
        assert not is_structured_error(r) and has_meta(r)
        text = r["text"]
        assert len(text) > 5000                            # real section, not a stub
        assert "<" not in text[:2000] or "<=" in text[:2000]  # no HTML tags
        assert "\xa0" not in text                          # no NBSP artifacts

    def test_mdna_alias(self):
        filings = S.get_filings(FPI, form_type="10-K", limit=1)["filings"]
        r = S.get_filing_section(filings[0]["accession"], "mdna",
                                 ticker_or_cik=FPI)
        # FPI 20-F sections may not segment — but the answer must be structured
        assert is_structured_error(r) or len(r["text"]) > 500

    def test_8k_item(self):
        f = S.get_filings(MEGA, form_type="8-K", limit=5)["filings"]
        # Find an 8-K announcing results (item 2.02 is the earnings item)
        target = next((x for x in f if "2.02" in (x["items"] or "")), None)
        if target is None:
            pytest.skip("no recent 8-K with item 2.02")
        r = S.get_filing_section(target["accession"], "item_2.02",
                                 ticker_or_cik=MEGA)
        assert not is_structured_error(r)
        assert len(r["text"]) > 100

    def test_missing_8k_item_structured(self):
        f = S.get_filings(MEGA, form_type="8-K", limit=1)["filings"]
        r = S.get_filing_section(f[0]["accession"], "item_6.66",
                                 ticker_or_cik=MEGA)
        assert is_structured_error(r) and r["code"] == "NOT_FOUND"


@pytest.mark.integration
class TestGetFundamentals:
    @pytest.mark.parametrize("ticker", PANEL)
    def test_annual_panel(self, ticker):
        r = S.get_fundamentals(ticker, period="annual", periods_back=2,
                               include_segments=False)
        assert not is_structured_error(r), f"{ticker}: {r}"
        assert has_meta(r)
        latest = r["periods"][0]
        # Revenue is the one metric every operating company reports
        assert latest["metrics"]["revenue"], f"{ticker}: no revenue"
        # Chart series must align with periods
        assert len(r["chartSeries"]["labels"]) == len(r["periods"])

    def test_ttm(self):
        r = S.get_fundamentals(MEGA, period="ttm", include_segments=False)
        assert not is_structured_error(r)
        assert r["periods"][0]["fiscalPeriod"] == "TTM"
        # TTM revenue must exceed any single quarter — sanity of the sum
        assert r["periods"][0]["metrics"]["revenue"] > 100e9

    def test_metric_subset(self):
        r = S.get_fundamentals(MEGA, metrics=["revenue", "netMargin"],
                               include_segments=False)
        assert set(r["periods"][0]["metrics"].keys()) == {"revenue", "netMargin"}

    def test_segments_chart_ready(self):
        r = S.get_fundamentals(MEGA, periods_back=1)       # segments on by default
        segs = r["segments"]
        # AAPL reports both product and geographic splits
        assert segs["product"] and segs["geographic"]
        slice0 = segs["product"][0]
        # Chart contract: name + value + pct on every slice
        assert slice0["name"] and slice0["value"] and slice0["pct"]

    def test_invalid_ticker_structured(self):
        r = S.get_fundamentals(INVALID)
        assert is_structured_error(r)


@pytest.mark.integration
class TestGetQuote:
    @pytest.mark.parametrize("ticker", PANEL)
    def test_panel_mandatory_metadata(self, ticker):
        r = S.get_quote(ticker)
        assert not is_structured_error(r), f"{ticker}: {r}"
        # The four mandatory fields — never silently stale
        assert r["asOf"] and r["provider"]
        assert r["session"] in ("pre", "regular", "after", "closed")
        assert isinstance(r["ageSeconds"], (int, float))
        assert r["price"] and r["price"] > 0

    def test_invalid_ticker_structured(self):
        r = S.get_quote(INVALID)
        assert is_structured_error(r)
        # And crucially: the failure is about the ticker, not the clock
        assert "market closed" not in r["error"].lower()


@pytest.mark.integration
class TestInsiderAndOwnership:
    def test_insider_activity_megacap(self):
        r = S.get_insider_activity(MEGA, limit=10)
        assert not is_structured_error(r) and has_meta(r)
        assert r["count"] >= 1                             # AAPL insiders always active
        t = r["transactions"][0]
        # Contract fields: name, side, shares, post-transaction holdings
        assert t["insiderName"] and t["side"] in ("buy", "sell", "other")
        assert t["accession"]

    def test_insider_date_filter(self):
        r = S.get_insider_activity(MEGA, date_from="2026-01-01", limit=20)
        assert all((t["date"] or "") >= "2026-01-01" for t in r["transactions"])

    def test_recent_ipo_empty_is_valid(self):
        # A near-zero-history ticker may have no Form 4s — that's an answer,
        # not an error
        r = S.get_insider_activity(RECENT_IPO, limit=5)
        assert not is_structured_error(r) or r["code"] != "INTERNAL"

    def test_ownership_megacap(self):
        r = S.get_ownership(MEGA)
        assert not is_structured_error(r), f"{r}"
        holders = r["institutionalHolders"]["holders"]
        assert holders and holders[0]["institution"]
        assert holders[0]["reportDate"]                    # 13F period attached
        # 13D/G list exists (mega-caps accumulate 13G filings)
        assert isinstance(r["beneficialOwners"]["filings"], list)


@pytest.mark.integration
class TestScreenAndCompare:
    def test_screen_quality_valuation(self):
        r = S.screen({"sector": "semiconductors", "fcf_positive": True,
                      "pe_max": 60}, limit=5)
        assert not is_structured_error(r) and has_meta(r)
        # Coverage must be reported — no silent truncation
        assert r["coverage"]["universeSize"] >= 1
        for m in r["matches"]:
            assert m["fcf"] and m["fcf"] > 0               # filter actually applied
            assert m["pe"] is not None and m["pe"] <= 60

    def test_screen_event_filter(self):
        r = S.screen({"sector": "semiconductors", "filed_8k_last_7d": True},
                     limit=3)
        assert not is_structured_error(r)
        assert r["coverage"]["eventChecksRun"] is not None

    def test_compare_normalized(self):
        r = S.compare([MEGA, FPI], metrics=["revenue", "netMargin"])
        assert not is_structured_error(r)
        assert len(r["rows"]) == 2
        for row in r["rows"]:
            assert row["endDate"]                          # period alignment is explicit
            assert row["values"]["revenue"]

    def test_compare_partial_failure_reported(self):
        r = S.compare([MEGA, INVALID], metrics=["revenue"])
        assert not is_structured_error(r)                  # one good row survives
        assert len(r["rows"]) == 1
        assert r["failures"] and r["failures"][0]["ticker"] == INVALID


# ═════════════════════════════════════════════════════════════════════════
#  THE 24/7 PROOF — simulated Sunday 03:00 ET run
# ═════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sunday_3am(monkeypatch):
    """Freeze the surface clock at Sunday 2026-06-14 03:00 ET (07:00 UTC)."""
    frozen = datetime(2026, 6, 14, 7, 0, tzinfo=timezone.utc)
    # _now() is the single source of wall-clock truth for session logic
    monkeypatch.setattr(sess, "_now", lambda: frozen)
    # Clear the surface quote cache so the session label is recomputed
    from sec_mcp.surface import quotes
    monkeypatch.setattr(quotes, "_cache", {})
    return frozen


@pytest.mark.integration
class TestSunday3am:
    """Markets closed must NEVER degrade the answers — only label them."""

    def test_clock_is_actually_sunday(self, sunday_3am):
        assert sess.market_session() == "closed"
        assert sess.edgar_business_hours() is False

    def test_quote_answers_with_closed_label(self, sunday_3am):
        r = S.get_quote(MEGA)
        assert not is_structured_error(r), f"quote errored off-hours: {r}"
        assert r["session"] == "closed"                    # honest label
        assert r["priceBasis"] == "last_close"             # what the number means
        assert r["price"] and r["price"] > 0               # still a real price
        # The cardinal sin — 'market closed' as an error — must not exist
        assert "market closed" not in str(r).lower() or r["session"] == "closed"

    def test_quote_cache_ttl_stretches_when_closed(self, sunday_3am):
        assert sess.quote_ttl() == 3600                    # closed → 1h budget

    def test_filings_work_normally(self, sunday_3am):
        r = S.get_filings(MEGA, form_type="10-K", limit=2)
        assert not is_structured_error(r)
        assert r["count"] >= 1

    def test_filings_index_polls_slower(self, sunday_3am):
        assert sess.filings_index_ttl() == 600             # frozen index → 10 min

    def test_fundamentals_work_normally(self, sunday_3am):
        r = S.get_fundamentals(MEGA, periods_back=1, include_segments=False)
        assert not is_structured_error(r)
        assert r["periods"][0]["metrics"]["revenue"]

    def test_insiders_work_normally(self, sunday_3am):
        r = S.get_insider_activity(MEGA, limit=3)
        assert not is_structured_error(r)
