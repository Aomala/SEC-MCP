#!/usr/bin/env python3
"""Fetch logos for ALL US-listed stocks (~10k) — no API key required.

Sources tried in order (all free):
  1. nvstly/icons               GitHub raw  (community-maintained, broad)
  2. davidepalazzo/ticker-logos GitHub raw  (fallback)
  3. logo.dev via known domains (only for the ~hundred names we have domains for)
  4. generated monogram tile     (guarantees every ticker has *something*)

Ticker universe comes from SEC company_tickers.json (free, no key) — the same
source SEC-MCP's ingest uses. Class-share dots (BRK.B) are saved as BRK-B.

Usage:
  python fetch_all_us_logos.py --out /tmp/logos_build/tickers
  python fetch_all_us_logos.py --out ... --limit 30        # test run
  python fetch_all_us_logos.py --out ... --no-monogram     # skip fallback tiles
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

NVSTLY = "https://raw.githubusercontent.com/nvstly/icons/main/ticker_icons/{t}.png"
DAVIDE = "https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/{t}.png"
LOGODEV_KEY = "pk_WyZRLK_ZSxmHL3ZyyD22-A"  # public key, also used by Fineas script
MONO_SIZE = 256

_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
]
_PALETTE = [
    (99, 102, 241), (16, 185, 129), (239, 68, 68), (245, 158, 11),
    (14, 165, 233), (168, 85, 247), (236, 72, 153), (34, 197, 94),
    (249, 115, 22), (6, 182, 212), (139, 92, 246), (220, 38, 38),
]
_counter_lock = threading.Lock()
_done = 0


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_monogram(ticker: str) -> bytes:
    h = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    bg = _PALETTE[h % len(_PALETTE)]
    img = Image.new("RGBA", (MONO_SIZE, MONO_SIZE), bg + (255,))
    draw = ImageDraw.Draw(img)
    text = ticker[:4] if len(ticker) <= 4 else ticker[:3]
    fsize = 120 if len(text) <= 2 else (90 if len(text) == 3 else 70)
    font = _load_font(fsize)
    b = draw.textbbox((0, 0), text, font=font)
    draw.text(((MONO_SIZE - (b[2] - b[0])) / 2 - b[0], (MONO_SIZE - (b[3] - b[1])) / 2 - b[1]),
              text, fill=(255, 255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _variants(ticker: str) -> list[str]:
    """GitHub repos use '-' not '.' for class shares; try a few forms."""
    out = [ticker]
    if "." in ticker:
        out += [ticker.replace(".", "-"), ticker.replace(".", "")]
    return out


def _try(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 200:
            Image.open(io.BytesIO(r.content)).verify()
            return r.content
    except Exception:
        pass
    return None


def fetch_real(ticker: str, domains: dict) -> tuple[bytes, str] | None:
    for v in _variants(ticker):
        d = _try(NVSTLY.format(t=v))
        if d:
            return d, "nvstly"
    for v in _variants(ticker):
        d = _try(DAVIDE.format(t=v))
        if d:
            return d, "davide"
    dom = domains.get(ticker)
    if dom:
        d = _try(f"https://img.logo.dev/{dom}?token={LOGODEV_KEY}&size=256&format=png")
        if d:
            return d, "logodev"
    return None


def process(ticker: str, out_dir: Path, total: int, domains: dict, monogram: bool) -> str:
    global _done
    fname = ticker.replace(".", "-")  # filename-safe
    out = out_dir / f"{fname}.png"
    status = "skip"
    if not out.exists():
        hit = fetch_real(ticker, domains)
        if hit:
            out.write_bytes(hit[0])
            status = hit[1]
        elif monogram:
            out.write_bytes(make_monogram(fname))
            status = "mono"
        else:
            status = "miss"
    with _counter_lock:
        _done += 1
        if _done % 250 == 0 or _done == total:
            print(f"  [{_done}/{total}] ...{ticker} ({status})", flush=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tickers", default="/tmp/us_tickers.json")
    ap.add_argument("--domains", default="", help="optional JSON {ticker: domain}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--no-monogram", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = json.load(open(args.tickers))
    domains = json.load(open(args.domains)) if args.domains and Path(args.domains).exists() else {}
    if args.limit:
        tickers = tickers[: args.limit]

    total = len(tickers)
    print(f"Fetching logos for {total} tickers → {out_dir}", flush=True)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, t, out_dir, total, domains, not args.no_monogram) for t in tickers]
        for f in as_completed(futs):
            r = f.result()
            counts[r] = counts.get(r, 0) + 1

    # Manifest of real logos (UI prefers real; can decide per-app whether to use monograms)
    manifest = {t: f"/logos/tickers/{t.replace('.', '-')}.png" for t in tickers}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"\nDone. {counts} total={total}", flush=True)
    print(f"manifest: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
