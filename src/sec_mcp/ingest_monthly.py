"""Index a completed calendar month's periodic/current filings, including untagged 6-Ks."""

import argparse
import logging
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from sec_mcp.ingest_all import _get_sb
from sec_mcp.sec_client import SEC_BASE, get_sec_client

FORMS = {"10-K", "10-Q", "8-K", "20-F", "40-F", "6-K"}
ACCESSION = re.compile(r"edgar/data/\d+/(\d{10}-\d{2}-\d{6})\.txt")


def month_filings(index: str, month: str) -> list[dict]:
    rows = {}
    for line in index.splitlines():
        fields = line.split("|")
        if len(fields) != 5:
            continue
        cik, _, form, filed, path = fields
        if not filed.startswith(month + "-") or form.removesuffix("/A") not in FORMS:
            continue
        match = ACCESSION.fullmatch(path.strip())
        if not match or not cik.isdigit():
            raise ValueError(f"Malformed SEC index record for {form} on {filed}")
        accession = match[1]
        rows.setdefault(
            accession,
            {
                "accession": accession,
                "cik": int(cik),
                "form": form,
                "filed": filed,
                "is_amendment": form.endswith("/A"),
            },
        )
    if not rows:
        raise ValueError(f"No covered filings in the SEC master index for {month}")
    return list(rows.values())


def ingest_month(month: str, today: date) -> None:
    first = date.fromisoformat(month + "-01")
    if first >= today.replace(day=1):
        raise ValueError("Choose a completed calendar month")
    quarter = (first.month - 1) // 3 + 1
    url = f"{SEC_BASE}/Archives/edgar/full-index/{first.year}/QTR{quarter}/master.idx"
    index = get_sec_client()._request(url, timeout=90).content.decode("latin-1")
    rows = month_filings(index, month)
    table = _get_sb().table("sec_filings")
    for offset in range(0, len(rows), 500):
        # A later graph extraction may have enriched the filing. Preserve those fields.
        table.upsert(
            rows[offset : offset + 500], on_conflict="accession", ignore_duplicates=True
        ).execute()
    logging.info(
        "%s indexed %d distinct accessions: %s",
        month,
        len(rows),
        dict(Counter(row["form"] for row in rows)),
    )


if __name__ == "__main__":
    today = datetime.now(timezone.utc).date()
    previous = today.replace(day=1) - timedelta(days=1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default=previous.strftime("%Y-%m"))
    ingest_month(parser.parse_args().month, today)
