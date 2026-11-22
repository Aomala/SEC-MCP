from datetime import date
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from sec_mcp import financials
from sec_mcp.facts import interim
from sec_mcp.ingest_monthly import ingest_month, month_filings
from sec_mcp.models import CompanyInfo, FilingMetadata
from sec_mcp.sec_client import SECClient, fiscal_year_end_from_submissions
from sec_mcp.surface.filings import _expand_forms


# Small representations of the issuer's release formats. Values and dates:
# TSM accession 0001046179-26-000451, BABA 0001104659-26-099220,
# ASML 0001628280-26-048235. Guidance and comparative columns are distractors.
def table(rows):
    return (
        "<table>"
        + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
        + "</table>"
    )


TSM = (
    "consolidated revenue of NT$1,270.38 billion, net income of NT$706.56 billion "
    "for the second quarter ended June 30, 2026. Unit: NT$ million, except for EPS"
    + table(
        [
            ["", "2Q26 Amount", "2Q25 Amount", "%"],
            ["Net sales", "1,270,381", "933,792", "36.0"],
            ["Net income", "706,562", "398,273", "77.4"],
        ]
    )
)
BABA = (
    "In the quarter ended June 30, 2026: • Revenue was RMB268,953 million (US$39,639 million). "
    "Non-GAAP net income was RMB20,715 million. Net income was RMB10,444 million."
)
ASML = "ASML reports €9.3 billion total net sales and €2.9 billion net income in Q2 2026. " + table(
    [
        ["Figures in millions of euros", "Q1 2026", "Q2 2026"],
        ["Total net sales", "8,767", "9,326"],
        ["Net income", "2,757", "2,918"],
    ]
)


@pytest.mark.parametrize(
    "ticker,html,rd,fye,expected",
    [
        (
            "TSM",
            TSM,
            "2026-07-16",
            "1231",
            ("2026-06-30", (2026, "Q2"), "TWD", 1270381e6, 706562e6),
        ),
        (
            "BABA",
            BABA,
            "2026-08-20",
            "0331",
            ("2026-06-30", (2027, "Q1"), "CNY", 268953e6, 10444e6),
        ),
        ("ASML", ASML, "2026-06-28", "1231", ("2026-06-28", (2026, "Q2"), "EUR", 9326e6, 2918e6)),
    ],
)
def test_reported_exhibit_quarter(ticker, html, rd, fye, expected):
    end, fiscal, currency, metrics = interim.parse_release(ticker, html, rd, fye)
    assert (
        end.isoformat(),
        fiscal,
        currency,
        metrics["revenue"],
        metrics["net_income"],
    ) == expected


def test_release_rejects_wrong_period_and_guidance():
    assert interim.parse_release("ASML", ASML, "2026-07-15", "1231") is None
    assert (
        interim.parse_release(
            "TSM", TSM.replace("2Q26 Amount", "3Q26 Guidance"), "2026-07-16", "1231"
        )
        is None
    )
    assert (
        interim.parse_release(
            "BABA", BABA.replace("Revenue was", "Projected revenue was"), None, "0331"
        )
        is None
    )


def test_sec_exhibit_search_includes_initial_reports(monkeypatch):
    def search(url, timeout):
        filters = parse_qs(urlsplit(url).query)
        assert filters["forms"] == ["6-K"]
        assert filters["ciks"] == ["0001046179"]
        return {"hits": {"hits": [{"_id": "result"}]}}

    monkeypatch.setattr(interim, "get_sec_client", lambda: SimpleNamespace(_request_json=search))
    assert interim._release_hits.__wrapped__("1046179", "per ADR unit", date(2026, 11, 22)) == [
        {"_id": "result"}
    ]


def test_untagged_history_uses_actual_exhibit(monkeypatch):
    cik, accession = "1046179", "0001046179-26-000451"
    hit = {
        "_id": accession + ":release.htm",
        "_source": {
            "form": "6-K",
            "file_type": "EX-99.1",
            "ciks": [cik.zfill(10)],
            "period_ending": "2026-06-30",
            "file_date": "2026-07-16",
        },
    }
    client = SimpleNamespace(
        get_periodic_filings_smart=lambda *a, **kw: [],
        _get_submissions=lambda _: {"fiscalYearEnd": "1231"},
    )
    monkeypatch.setattr(financials, "get_sec_client", lambda: client)
    monkeypatch.setattr(interim, "get_sec_client", lambda: client)
    monkeypatch.setattr(interim, "_release_hits", lambda *args: [hit, hit])

    def parse_filed_release(tk, url, rd, fye):
        assert rd == "2026-06-30"
        return interim.parse_release(tk, TSM, rd, fye)

    monkeypatch.setattr(interim, "_read_release", parse_filed_release)
    monkeypatch.setattr("sec_mcp.core.fx.convert_metrics_to_usd", lambda m, c: (m, None))
    history = financials.get_fmp_shaped_history("TSM", period="quarter")
    (inc,) = history["income"]
    assert (inc["period"], inc["date"], inc["reportedCurrency"], inc["revenue"]) == (
        "Q2",
        "2026-06-30",
        "TWD",
        1270381e6,
    )
    assert inc["_meta"]["accession"] == accession
    assert inc["_meta"]["sources"]["revenue"].endswith("/release.htm")
    assert history["cashflow"][0]["operatingCashFlow"] is None


@pytest.mark.parametrize("form", ["6-K", "6-K/A"])
def test_event_or_half_year_never_borrows_an_annual(monkeypatch, form):
    filing = FilingMetadata(
        accession_number="event", filing_date="2026-07-16", report_date="2026-06-30", form_type=form
    )
    company = CompanyInfo(cik=123, name="Test", ticker="TEST", fiscal_year_end="1231")
    client = SimpleNamespace(get_company_info=lambda _: company)
    monkeypatch.setattr(financials, "get_sec_client", lambda: client)
    monkeypatch.setattr(financials.disk_cache, "get", lambda *args: None)

    def facts(ticker, *, accession, form_type):
        assert accession == "event" and form_type == form
        return pd.DataFrame(
            [
                {
                    "concept": "Revenue",
                    "start": "2026-01-01",
                    "end": pd.Timestamp("2026-06-30"),
                    "value": 35,
                }
            ]
        )

    monkeypatch.setattr(financials, "_get_facts_for_filing", facts)
    result = financials.extract_financials("TEST", filing=filing)
    assert result["metrics"] == {} and result["error"]
    assert result["filing_info"]["accession_number"] == "event"


def test_tagged_ifrs_quarter_with_annual_focus(monkeypatch):
    filing = FilingMetadata(
        accession_number="interim",
        filing_date="2026-07-16",
        report_date="2026-07-16",
        form_type="6-K/A",
    )
    company = CompanyInfo(cik=123, name="Test", ticker="TEST", fiscal_year_end="0331")
    base = {
        "concept": "Revenue",
        "end": pd.Timestamp("2026-06-30"),
        "form": "6-K/A",
        "units": "USD",
        "fy": 2026,
        "fp": "FY",
        "taxonomy": "ifrs-full",
        "accn": "interim",
    }
    frame = pd.DataFrame(
        [
            {**base, "start": start, "value": n}
            for start, n in [("2026-04-01", 25), ("2026-01-01", 55)]
        ]
    )
    monkeypatch.setattr(
        financials, "get_sec_client", lambda: SimpleNamespace(get_company_info=lambda _: company)
    )
    monkeypatch.setattr(financials, "_get_facts_for_filing", lambda *a, **kw: frame)
    monkeypatch.setattr(financials.disk_cache, "get", lambda *args: None)
    monkeypatch.setattr(financials.disk_cache, "put", lambda *args: None)
    monkeypatch.setattr("sec_mcp.facts.ingest.record_observations", lambda _: None)
    out = financials.extract_financials("TEST", filing=filing)
    assert out["metrics"]["revenue"] == 25
    assert out["fiscal_year"] == 2027 and out["quarter_label"].startswith("Q1 ")
    assert not out["is_ytd"] and out["filing_info"]["report_date"] == "2026-06-30"


def test_discovery_keeps_amendment_and_ignores_event(monkeypatch):
    client = SECClient()
    end = pd.Timestamp("2026-06-30")
    base = {
        "form": "6-K",
        "accn": "first",
        "taxonomy": "ifrs-full",
        "end": end,
        "start": "2026-04-01",
        "filed": end + pd.Timedelta(days=16),
    }
    frame = pd.DataFrame(
        [
            base,
            {**base, "accn": "amend", "form": "6-K/A", "filed": end + pd.Timedelta(days=22)},
            {**base, "accn": "event", "start": None, "taxonomy": "dei"},
        ]
    )
    monkeypatch.setattr(client, "get_facts_dataframe", lambda *a, **kw: frame)
    filings = client.get_periodic_filings("TEST", form_type="6-K")
    assert [(f.form_type, f.accession_number) for f in filings] == [
        ("6-K/A", "amend"),
        ("6-K", "first"),
    ]
    assert "6-K/A" in _expand_forms("6-K") and "6-K/A" in _expand_forms("10-Q")


def test_month_index_includes_untagged_forms_and_amendments():
    lines = []
    for i, (form, day) in enumerate(
        [
            ("6-K", "2026-10-01"),
            ("6-K/A", "2026-10-31"),
            ("10-Q", "2026-10-15"),
            ("4", "2026-10-15"),
            ("6-K", "2026-11-01"),
        ]
    ):
        lines.append(f"123|Example|{form}|{day}|edgar/data/123/0000000123-26-00000{i}.txt")
    result = month_filings("\n".join(lines + [lines[0]]), "2026-10")
    assert [r["form"] for r in result] == ["6-K", "6-K/A", "10-Q"]
    assert result[1]["is_amendment"]
    with pytest.raises(ValueError):
        month_filings("temporary service error", "2026-10")
    with pytest.raises(ValueError):
        ingest_month("2026-11", date(2026, 11, 21))


def test_missing_fiscal_year_end_uses_latest_annual_report_date():
    submissions = {
        "fiscalYearEnd": None,
        "filings": {
            "recent": {
                "form": ["6-K", "20-F", "20-F", "20-F"],
                "reportDate": ["2026-08-20", "2025-03-31", "2026-03-31", ""],
                "filingDate": ["2026-08-20", "2025-06-26", "2026-05-20", ""],
            }
        },
    }
    year_end = fiscal_year_end_from_submissions(submissions)
    assert year_end == "0331"
    assert interim.parse_release("BABA", BABA, "2026-08-20", year_end)[1] == (2027, "Q1")
    assert fiscal_year_end_from_submissions({"fiscalYearEnd": "0930"}) == "0930"
    assert fiscal_year_end_from_submissions({}) is None
