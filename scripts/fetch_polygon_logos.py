#!/usr/bin/env python3
"""Fetch high-quality company logos from Polygon's branding endpoint.

For each ticker:
  1. GET /v3/reference/tickers/{t}  -> results.branding.{logo_url, icon_url}
  2. download both assets (URLs require ?apiKey= appended)

Saves:
  <out>/<TICKER>_icon.<ext>   square branded mark (rows/avatars)
  <out>/<TICKER>_logo.<ext>   full wordmark (headers); often .svg

Coverage: tickers with no branding block are recorded in _no_branding.json.
Polygon paid tier has high rate limits; default 16 workers is safe.

Usage:
  python fetch_polygon_logos.py --out /tmp/poly_logos --limit 50
  python fetch_polygon_logos.py --out /tmp/poly_logos            # all in --tickers
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REF = "https://api.polygon.io/v3/reference/tickers/{t}"
_lock = threading.Lock()
_n = 0


def _ext_from(url: str, ctype: str) -> str:
    for e in (".svg", ".png", ".jpeg", ".jpg", ".webp"):
        if url.lower().split("?")[0].endswith(e):
            return ".jpg" if e == ".jpeg" else e
    if "svg" in ctype:
        return ".svg"
    if "png" in ctype:
        return ".png"
    if "webp" in ctype:
        return ".webp"
    return ".jpg"


def _dl(url: str, key: str, dest_base: Path, kind: str) -> str | None:
    """Download one branding asset; returns the saved filename or None."""
    sep = "&" if "?" in url else "?"
    try:
        r = requests.get(f"{url}{sep}apiKey={key}", timeout=20)
        if r.status_code != 200 or len(r.content) < 200:
            return None
        ext = _ext_from(url, r.headers.get("content-type", ""))
        out = dest_base.with_name(f"{dest_base.name}_{kind}{ext}")
        out.write_bytes(r.content)
        return out.name
    except Exception:
        return None


def process(ticker: str, out_dir: Path, key: str, total: int, icon_only: bool = False) -> dict:
    global _n
    res = {"ticker": ticker, "icon": None, "logo": None, "branding": False}
    try:
        r = requests.get(REF.format(t=ticker), params={"apiKey": key}, timeout=20)
        if r.status_code == 200:
            b = (r.json().get("results") or {}).get("branding") or {}
            base = out_dir / ticker
            if b.get("icon_url"):
                res["icon"] = _dl(b["icon_url"], key, base, "icon")
            if not icon_only and b.get("logo_url"):
                res["logo"] = _dl(b["logo_url"], key, base, "logo")
            res["branding"] = bool(res["icon"] or res["logo"])
    except Exception:
        pass
    with _lock:
        _n += 1
        if _n % 250 == 0 or _n == total:
            print(f"  [{_n}/{total}] ...{ticker} branding={res['branding']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tickers", default="/tmp/us_tickers.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--icon-only", action="store_true", help="skip wordmark logos")
    args = ap.parse_args()

    key = os.environ.get("POLYGON_API_KEY", "")
    assert key, "POLYGON_API_KEY not set"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = json.load(open(args.tickers))
    if args.limit:
        tickers = tickers[: args.limit]

    total = len(tickers)
    print(f"Polygon branding fetch for {total} tickers -> {out_dir}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, t, out_dir, key, total, args.icon_only) for t in tickers]
        for f in as_completed(futs):
            results.append(f.result())

    have = [r for r in results if r["branding"]]
    no_brand = [r["ticker"] for r in results if not r["branding"]]
    both = sum(1 for r in results if r["icon"] and r["logo"])

    manifest = {r["ticker"]: {"icon": r["icon"], "logo": r["logo"]} for r in have}
    (out_dir / "_polygon_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "_no_branding.json").write_text(json.dumps(sorted(no_brand), indent=2))

    print(f"\nDone. with_branding={len(have)} (both icon+logo={both}) "
          f"no_branding={len(no_brand)} total={total}", flush=True)
    print(f"coverage: {len(have)/total*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
