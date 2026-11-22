"""Identify standalone fiscal quarters in foreign issuers' SEC disclosures."""

import calendar
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from sec_mcp.sec_client import SEC_BASE, fiscal_year_end_from_submissions, get_sec_client

# Exhibit 99.1 release formats validated against the issuers' quarterly disclosures.
# For other issuers the regular accession-scoped XBRL resolver supplies financials.
_RELEASES = {
    "TSM": ("1046179", "per ADR unit"),
    "BABA": ("1577552", "business highlights"),
    "ASML": ("937966", "total net sales and"),
}
_DATE = r"[A-Za-z]+\s+\d{1,2},\s+20\d{2}"


def fiscal_quarter(end: date, year_end: str):
    """Match a period end to the issuer's fiscal calendar, allowing a 52/53-week year."""
    if not re.fullmatch(r"\d{4}", year_end or ""):
        return None
    month, day = int(year_end[:2]), int(year_end[2:])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    for year in (end.year - 1, end.year, end.year + 1):
        for quarter in range(1, 5):
            ordinal = year * 12 + month - 1 - 3 * (4 - quarter)
            y, m = divmod(ordinal, 12)
            last = calendar.monthrange(y, m + 1)[1]
            # Month-end filers use the end of each quarter's month.
            d = last if day >= 28 else min(day, last)
            if abs((end - date(y, m + 1, d)).days) <= 7:
                return year, f"Q{quarter}"
    return None


def _number(value):
    return float(value.replace(",", ""))


def _date(text):
    return datetime.strptime(" ".join(text.split()), "%B %d, %Y").date()


def _table_metrics(soup, column, header_matches, labels):
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            cells = [
                " ".join(cell.stripped_strings).strip()
                for cell in tr.find_all(["td", "th"], recursive=False)
            ]
            rows.append([text for text in cells if text])
        if not any(header_matches(row) for row in rows):
            continue
        values = {}
        for row in rows:
            if len(row) <= column:
                continue
            key = labels.get(row[0].casefold())
            if key and re.fullmatch(r"[\d,]+(?:\.\d+)?", row[column]):
                values[key] = _number(row[column]) * 1e6
        if "revenue" in values:
            return values
    return {}


def parse_release(ticker, html, report_date, year_end):
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.get_text(" ", strip=True).split())
    if ticker == "TSM":
        match = re.search(
            r"consolidated revenue of NT\$[\d,.]+ billion, net income of NT\$[\d,.]+ billion.{0,150}?quarter ended ("
            + _DATE
            + ")",
            text,
            re.I,
        )
        if not match:
            return None
        end = _date(match[1])
        fiscal = fiscal_quarter(end, year_end)
        if not fiscal:
            return None
        year, quarter = fiscal
        if not re.search(r"(?:unit\s*:\s*)?NT\$\s+million,?\s+except for EPS", text, re.I):
            return None
        header = f"{quarter[1]}q{year % 100:02} amount"
        metrics = _table_metrics(
            soup,
            1,
            lambda row: bool(row) and row[0].casefold().startswith(header),
            {
                "net sales": "revenue",
                "gross profit": "gross_profit",
                "income from operations": "operating_income",
                "net income": "net_income",
            },
        )
        currency = "TWD"
    elif ticker == "ASML":
        m = re.search(
            r"ASML reports €\s*[\d,.]+ billion total net sales and €\s*[\d,.]+ billion net income in (Q[1-4]) (20\d\d)",
            text,
            re.I,
        )
        if not m or not report_date or not re.search("figures in millions of euros", text, re.I):
            return None
        end = date.fromisoformat(report_date)
        fiscal = fiscal_quarter(end, year_end)
        if fiscal != (int(m[2]), m[1].upper()):
            return None
        # Each selected release must put the actual quarter in the second data column.
        labels = {
            "total net sales": "revenue",
            "net income": "net_income",
            "gross profit": "gross_profit",
        }
        metrics = _table_metrics(
            soup,
            2,
            lambda cells: (
                len(cells) >= 3
                and "millions of euros" in cells[0].lower()
                and cells[2].upper() == f"{m[1].upper()} {m[2]}"
            ),
            labels,
        )
        currency = "EUR"
    elif ticker == "BABA":
        m = re.search(r"in the quarter ended (" + _DATE + r")\s*:", text, re.I)
        if not m:
            return None
        end = _date(m[1])
        fiscal = fiscal_quarter(end, year_end)
        # The first consolidated highlights are scoped explicitly to this quarter.
        highlights = text[m.end() : m.end() + 3600]
        highlights = re.split(
            r"(?:in|for) (?:the )?(?:six|nine|twelve) months|fiscal year ended",
            highlights,
            maxsplit=1,
            flags=re.I,
        )[0]
        metrics = {}
        for label, key in (
            ("Revenue", "revenue"),
            ("Net income", "net_income"),
            ("Income from operations", "operating_income"),
        ):
            match = re.search(
                r"(?:^|[.·•]\s*)\s*" + label + r" was RMB\s*([\d,.]+) million", highlights, re.I
            )
            if match:
                metrics[key] = _number(match[1]) * 1e6
        currency = "CNY"
    else:
        return None
    if fiscal and "revenue" in metrics:
        return end, fiscal, currency, metrics
    return None


@lru_cache(maxsize=120)
def _read_release(ticker, url, report_date, year_end):
    response = get_sec_client()._request(url, timeout=12)
    return parse_release(
        ticker, response.content.decode("utf-8", errors="replace"), report_date, year_end
    )


@lru_cache(maxsize=12)
def _release_hits(cik, phrase, today):
    params = {
        "q": f'"{phrase}"',
        "ciks": cik.zfill(10),
        # EFTS combines a root form and an explicit amendment with AND. The root already includes amendments.
        "forms": "6-K",
        "dateRange": "custom",
        "startdt": (today - timedelta(days=1200)).isoformat(),
        "enddt": today.isoformat(),
        "size": "100",
    }
    data = get_sec_client()._request_json(
        "https://efts.sec.gov/LATEST/search-index?" + urlencode(params), timeout=12
    )
    return data["hits"]["hits"]


def has_release_format(ticker):
    return ticker in _RELEASES


def release_quarters(ticker, limit, covered):
    if ticker not in _RELEASES:
        return []
    from sec_mcp.core.fx import convert_metrics_to_usd

    cik, phrase = _RELEASES[ticker]
    year_end = fiscal_year_end_from_submissions(get_sec_client()._get_submissions(cik))
    hits = _release_hits(cik, phrase, date.today())
    results = []
    for hit in sorted(hits, key=lambda h: h["_source"].get("file_date", ""), reverse=True):
        src = hit["_source"]
        accession, _, doc = hit.get("_id", "").partition(":")
        if (
            src.get("form") not in {"6-K", "6-K/A"}
            or src.get("file_type") != "EX-99.1"
            or cik.zfill(10) not in [str(c).zfill(10) for c in src.get("ciks", [])]
            or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.html?", doc, re.I)
        ):
            continue
        url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}"
        try:
            parsed = _read_release(ticker, url, src.get("period_ending"), year_end)
        except (ValueError, OSError):
            continue
        if not parsed:
            continue
        end, (year, quarter), currency, metrics = parsed
        filed = src["file_date"]
        if not 0 <= (date.fromisoformat(filed) - end).days <= 180:
            continue
        if (year, quarter) in covered:
            continue
        converted, rate = convert_metrics_to_usd(metrics, currency)
        results.append(
            {
                "ticker_or_cik": ticker,
                "cik": int(cik),
                "fiscal_year": year,
                "period_type": "quarterly",
                "quarter_label": quarter,
                "is_ytd": False,
                "quality": "standalone",
                "reporting_currency": currency,
                "fx_rate": rate,
                "filing_info": {
                    "accession_number": accession,
                    "form_type": src["form"],
                    "report_date": end.isoformat(),
                    "filing_date": filed,
                },
                "metrics": converted,
                "metrics_sourced": dict.fromkeys(metrics, url),
            }
        )
        covered.add((year, quarter))
        if len(results) >= limit:
            break
    return results
