#!/usr/bin/env python3
"""Generate flag-pair composite logos for currency pairs — no API key.

For each pair (e.g. EUR/USD) builds a split tile: base-currency flag on the
left, quote-currency flag on the right, with a thin divider. Currencies that
aren't a country (BTC, ETH, gold XAU, silver XAG) get a symbol tile.

Flags come from flagcdn.com (free, no key); each flag is downloaded once and
cached, then composited locally — so the whole FX matrix costs only ~30 flag
fetches.

Pair universe: if no --forexlist file is given, builds the standard FX matrix
locally (all ordered pairs among majors+minors, plus metals/crypto vs USD).

Usage:
  python fetch_forex_logos.py --out /tmp/logos_build/forex
  python fetch_forex_logos.py --out ... --limit 10   # test run
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

FLAG_URL = "https://flagcdn.com/w320/{cc}.png"
TILE_W, TILE_H = 320, 214  # 3:2 flag aspect

# ISO-4217 currency → ISO-3166-1 alpha-2 country (for flagcdn)
CCY_COUNTRY = {
    "AED": "ae", "AUD": "au", "BRL": "br", "CAD": "ca", "CHF": "ch",
    "CNY": "cn", "CZK": "cz", "DKK": "dk", "EUR": "eu", "GBP": "gb",
    "HKD": "hk", "HUF": "hu", "ILS": "il", "INR": "in", "JPY": "jp",
    "KRW": "kr", "MXN": "mx", "NOK": "no", "NZD": "nz", "PLN": "pl",
    "RUB": "ru", "SAR": "sa", "SEK": "se", "SGD": "sg", "THB": "th",
    "TRY": "tr", "USD": "us", "ZAR": "za",
}
# Non-country currencies → symbol drawn on a colored tile
# Plain ASCII labels — Unicode ₿/Ξ aren't in the macOS system fonts and render
# as empty boxes, so use the ticker letters on a brand-colored tile instead.
SYMBOL_CCY = {
    "BTC": ("BTC", (247, 147, 26)),       # bitcoin orange
    "ETH": ("ETH", (98, 126, 234)),       # ethereum blue
    "XAU": ("Au", (212, 175, 55)),        # gold
    "XAG": ("Ag", (170, 169, 173)),       # silver
}

_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
_flag_cache: dict[str, Image.Image] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _symbol_tile(text: str, color: tuple) -> Image.Image:
    img = Image.new("RGBA", (TILE_W, TILE_H), color + (255,))
    d = ImageDraw.Draw(img)
    f = _font(110 if len(text) == 1 else 90)
    b = d.textbbox((0, 0), text, font=f)
    d.text(((TILE_W - (b[2] - b[0])) / 2 - b[0], (TILE_H - (b[3] - b[1])) / 2 - b[1]),
           text, fill=(255, 255, 255, 255), font=f)
    return img


def _code_tile(code: str) -> Image.Image:
    img = Image.new("RGBA", (TILE_W, TILE_H), (71, 85, 105, 255))
    d = ImageDraw.Draw(img)
    f = _font(80)
    b = d.textbbox((0, 0), code, font=f)
    d.text(((TILE_W - (b[2] - b[0])) / 2 - b[0], (TILE_H - (b[3] - b[1])) / 2 - b[1]),
           code, fill=(255, 255, 255, 255), font=f)
    return img


def _half_tile(text: str, color: tuple) -> Image.Image:
    """Symbol glyph centered within a HALF-width tile (so it survives placement
    into either side of a pair composite without being clipped)."""
    img = Image.new("RGBA", (TILE_W // 2, TILE_H), color + (255,))
    d = ImageDraw.Draw(img)
    f = _font(96 if len(text) == 1 else 72)
    b = d.textbbox((0, 0), text, font=f)
    d.text((((TILE_W // 2) - (b[2] - b[0])) / 2 - b[0], (TILE_H - (b[3] - b[1])) / 2 - b[1]),
           text, fill=(255, 255, 255, 255), font=f)
    return img


def get_ccy_half(ccy: str) -> Image.Image:
    """Return a HALF-width (160x214) image for a currency: cropped flag half,
    or a symbol glyph centered in the half."""
    if ccy in _flag_cache:
        return _flag_cache[ccy]
    img: Image.Image | None = None
    if ccy in SYMBOL_CCY:
        text, color = SYMBOL_CCY[ccy]
        img = _half_tile(text, color)  # already half-width, glyph centered
    elif ccy in CCY_COUNTRY:
        try:
            r = requests.get(FLAG_URL.format(cc=CCY_COUNTRY[ccy]), timeout=12)
            if r.status_code == 200 and len(r.content) > 200:
                full = Image.open(io.BytesIO(r.content)).convert("RGBA").resize((TILE_W, TILE_H))
                img = full.crop((TILE_W // 4, 0, TILE_W // 4 + TILE_W // 2, TILE_H))  # center slice
        except Exception:
            img = None
    if img is None:
        img = _code_tile(ccy).crop((TILE_W // 4, 0, TILE_W // 4 + TILE_W // 2, TILE_H))
    _flag_cache[ccy] = img
    return img


def make_pair(base: str, quote: str) -> bytes:
    left = get_ccy_half(base)
    right = get_ccy_half(quote)
    canvas = Image.new("RGBA", (TILE_W, TILE_H), (0, 0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (TILE_W // 2, 0))
    d = ImageDraw.Draw(canvas)
    d.line([(TILE_W // 2, 0), (TILE_W // 2, TILE_H)], fill=(255, 255, 255, 230), width=3)
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


def build_default_matrix() -> list[str]:
    majors = ["USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"]
    others = ["CNY", "HKD", "SGD", "SEK", "NOK", "DKK", "MXN", "ZAR",
              "TRY", "INR", "BRL", "KRW", "PLN", "THB", "ILS", "CZK",
              "HUF", "RUB", "AED", "SAR"]
    ccys = majors + others
    syms = [b + q for b in ccys for q in ccys if b != q]
    syms += [m + "USD" for m in ("XAU", "XAG")]
    syms += [c + "USD" for c in ("BTC", "ETH")]
    return syms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--forexlist", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.forexlist and Path(args.forexlist).exists():
        data = json.load(open(args.forexlist))
        syms = [(x if isinstance(x, str) else x.get("symbol", "")) for x in data]
    else:
        syms = build_default_matrix()

    pairs = []
    for s in syms:
        s = (s or "").replace("/", "").upper()
        if len(s) == 6:
            pairs.append((s[:3], s[3:]))
    pairs = sorted(set(pairs))
    if args.limit:
        pairs = pairs[: args.limit]

    total = len(pairs)
    print(f"Generating {total} forex flag-pair logos → {out_dir}", flush=True)

    made = 0
    manifest = {}
    for i, (b, q) in enumerate(pairs, 1):
        sym = f"{b}{q}"
        out = out_dir / f"{sym}.png"
        if not out.exists():
            out.write_bytes(make_pair(b, q))
            made += 1
        manifest[f"{b}/{q}"] = f"/logos/forex/{sym}.png"
        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] {sym}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nDone. generated={made} total={total} flags_cached={len(_flag_cache)}", flush=True)


if __name__ == "__main__":
    main()
