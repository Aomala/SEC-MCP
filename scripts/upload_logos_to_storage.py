#!/usr/bin/env python3
"""Upload logo PNGs to Supabase Storage bucket 'logos' (public CDN).

Layout in bucket:
  logos/tickers/<TICKER>.png
  logos/forex/<PAIR>.png

Idempotent: uses upsert, so re-runs overwrite. Parallel via threads.
Reads creds from SEC-MCP/.env (SUPABASE_URL + service_role SUPABASE_KEY).

Usage:
  python upload_logos_to_storage.py --src /tmp/logos_build --limit 5   # test
  python upload_logos_to_storage.py --src /tmp/logos_build             # full
"""
from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BUCKET = "logos"
_lock = threading.Lock()
_done = 0
_fail = 0


def _load_env() -> tuple[str, str]:
    env = Path("/Users/austin/SEC-MCP/.env")
    url = key = ""
    for line in env.read_text().splitlines():
        if line.startswith("SUPABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("SUPABASE_KEY="):
            key = line.split("=", 1)[1].strip().strip('"')
    return url, key


def upload_one(url: str, key: str, local: Path, dest: str, total: int) -> bool:
    global _done, _fail
    api = f"{url}/storage/v1/object/{BUCKET}/{dest}"
    ok = False
    try:
        r = requests.post(
            api,
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
                "Content-Type": "image/png",
                "x-upsert": "true",  # overwrite if exists → idempotent
            },
            data=local.read_bytes(),
            timeout=30,
        )
        ok = r.status_code in (200, 201)
        if not ok and r.status_code not in (200, 201):
            # surface first few failures for debugging
            with _lock:
                if _fail < 5:
                    print(f"  FAIL {dest}: HTTP {r.status_code} {r.text[:120]}", flush=True)
    except Exception as e:
        with _lock:
            if _fail < 5:
                print(f"  ERR {dest}: {e}", flush=True)
    with _lock:
        _done += 1
        if not ok:
            _fail += 1
        if _done % 500 == 0 or _done == total:
            print(f"  [{_done}/{total}] uploaded (fails={_fail})", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/logos_build")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    url, key = _load_env()
    assert url and key, "missing SUPABASE_URL / SUPABASE_KEY"

    src = Path(args.src)
    jobs: list[tuple[Path, str]] = []
    for sub in ("tickers", "forex"):
        d = src / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.png")):
            jobs.append((f, f"{sub}/{f.name}"))
    if args.limit:
        jobs = jobs[: args.limit]

    total = len(jobs)
    print(f"Uploading {total} files to {url}/storage/v1/object/{BUCKET}/", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(upload_one, url, key, lf, dest, total) for lf, dest in jobs]
        for f in as_completed(futs):
            f.result()

    pub = f"{url}/storage/v1/object/public/{BUCKET}"
    print(f"\nDone. uploaded={_done - _fail} failed={_fail} total={total}", flush=True)
    print(f"Public base URL: {pub}/tickers/AAPL.png", flush=True)


if __name__ == "__main__":
    main()
