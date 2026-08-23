"""HTTP-layer caching: ETag/304 revalidation + per-route Cache-Control.

The data layer got fast (stale-while-revalidate in supabase_cache/core.refresh),
but the transport layer was still shipping every byte on every call:
/api/financials-history/AAPL is 176KB raw (10.4KB gzipped) and carried no ETag
and no Cache-Control, so Fineas re-downloaded the whole payload on every mount
and no CDN or browser could hold a copy.

This module adds two things:

1. **ETag + If-None-Match** — a strong ETag over the response body. A repeat
   caller that still holds the payload gets a 34-byte 304 instead of 176KB.
   Cheap to compute (blake2b over bytes we already have in memory) and it costs
   the client one round trip instead of one download.

2. **Cache-Control** — a per-route-class policy so browsers and any CDN in
   front of Railway can serve repeats without touching the app at all. The
   `stale-while-revalidate` directive mirrors what the data layer now does:
   hand over the cached copy immediately, refresh underneath.

Implemented as pure ASGI rather than BaseHTTPMiddleware because the latter
converts every response into a streaming one, which breaks the SSE endpoint
(/api/chatbot-stream) and the CSV/JSON export streams. Here, anything that is
not a small-enough JSON body is passed through untouched.
"""

from __future__ import annotations

import hashlib
import logging

log = logging.getLogger(__name__)

# Bodies larger than this are streamed through without an ETag — buffering a
# huge response to hash it would cost more memory than the 304 saves.
_MAX_ETAG_BYTES = 4 * 1024 * 1024

# Path prefix → Cache-Control. First match wins, so order matters: most
# specific first. `s-maxage` targets a CDN, `max-age` the browser.
#
# The reasoning per class:
#   filings/sections/statements — a filed document is immutable; cache hard.
#   fundamentals/metrics        — rebinds only when the next filing lands;
#                                 minutes of browser cache, hours of CDN, and a
#                                 long stale-while-revalidate window.
#   quotes/news/market          — genuinely live; seconds.
#   everything else             — no policy, left to the route.
_POLICIES: tuple[tuple[str, str], ...] = (
    ("/v1/price/", "public, max-age=15, stale-while-revalidate=60"),
    ("/api/news/", "public, max-age=300, stale-while-revalidate=1800"),
    ("/api/market/", "public, max-age=120, stale-while-revalidate=600"),
    ("/api/index/", "public, max-age=120, stale-while-revalidate=600"),
    ("/api/filing-text/", "public, max-age=86400, s-maxage=604800, immutable"),
    ("/api/section/", "public, max-age=86400, s-maxage=604800, immutable"),
    ("/api/filings/", "public, max-age=3600, s-maxage=86400"),
    ("/api/financials-history/", "public, max-age=600, s-maxage=21600, stale-while-revalidate=604800"),
    ("/api/metrics/", "public, max-age=300, s-maxage=3600, stale-while-revalidate=604800"),
    ("/api/segments/", "public, max-age=600, s-maxage=21600, stale-while-revalidate=604800"),
    ("/api/geo-revenue/", "public, max-age=600, s-maxage=21600, stale-while-revalidate=604800"),
    ("/api/peers/", "public, max-age=3600, s-maxage=86400"),
    ("/api/entity/", "public, max-age=3600, s-maxage=86400"),
    ("/api/etf/", "public, max-age=600, s-maxage=21600"),
    ("/api/tickers/bulk", "public, max-age=3600, s-maxage=86400"),
    ("/v1/", "public, max-age=300, s-maxage=3600, stale-while-revalidate=604800"),
    ("/static/", "public, max-age=3600"),
)

# Never touched: streams, mutations, and anything user-specific.
_SKIP_PREFIXES = (
    "/api/chat",          # also /api/chatbot, /api/chatbot-stream (SSE)
    "/api/export/",       # StreamingResponse
    "/api/cache/",
    "/api/ingest/",
)


def _policy_for(path: str) -> str | None:
    for prefix, policy in _POLICIES:
        if path.startswith(prefix):
            return policy
    return None


class ETagCacheMiddleware:
    """Adds ETag/304 revalidation and Cache-Control to cacheable GET responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.startswith(_SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        policy = _policy_for(path)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        if_none_match = headers.get("if-none-match", "")

        start_message: dict | None = None
        chunks: list[bytes] = []
        overflowed = False

        async def send_wrapper(message):
            nonlocal start_message, overflowed

            if message["type"] == "http.response.start":
                start_message = message
                return  # held back until we know the full body

            if message["type"] != "http.response.body":
                await send(message)
                return

            assert start_message is not None
            body = message.get("body", b"")
            more = message.get("more_body", False)

            if not overflowed:
                chunks.append(body)
                if sum(len(c) for c in chunks) > _MAX_ETAG_BYTES:
                    overflowed = True

            if overflowed:
                # Too big to hash — flush what we held and stream the rest.
                await self._flush_passthrough(send, start_message, chunks, policy)
                chunks.clear()
                await send({"type": "http.response.body", "body": body,
                            "more_body": more})
                return

            if more:
                return  # keep buffering

            full = b"".join(chunks)
            await self._finish(send, start_message, full, policy, if_none_match)

        await self.app(scope, receive, send_wrapper)

    async def _flush_passthrough(self, send, start_message, chunks, policy):
        """Body outgrew the ETag budget — send headers as-is (plus policy)."""
        raw = list(start_message["headers"])
        if policy and not any(k.lower() == b"cache-control" for k, _ in raw):
            raw.append((b"cache-control", policy.encode("latin-1")))
        await send({**start_message, "headers": raw})
        for c in chunks:
            await send({"type": "http.response.body", "body": c, "more_body": True})

    async def _finish(self, send, start_message, body, policy, if_none_match):
        status = start_message["status"]
        raw = list(start_message["headers"])
        have = {k.lower() for k, _ in raw}

        # Only 200s are worth revalidating; errors must not be cached as if valid.
        etag = None
        if status == 200 and body and b"etag" not in have:
            etag = f'"{hashlib.blake2b(body, digest_size=16).hexdigest()}"'
            raw.append((b"etag", etag.encode("latin-1")))
        elif b"etag" in have:
            etag = next((v.decode("latin-1") for k, v in raw if k.lower() == b"etag"), None)

        if policy and status == 200 and b"cache-control" not in have:
            raw.append((b"cache-control", policy.encode("latin-1")))
        # No Vary added here: GZipMiddleware sits OUTSIDE this one and calls
        # add_vary_header("Accept-Encoding") itself whenever it compresses.
        # Adding our own emitted it twice. The only case GZip skips is a client
        # that sent no Accept-Encoding, where the stored copy is uncompressed —
        # acceptable to every client, so no Vary is needed.

        if etag and if_none_match and _etag_matches(if_none_match, etag):
            # 304 carries validators + policy, never a body or a length.
            #
            # The CORS headers MUST survive. CORSMiddleware is installed inside
            # this one, so it stamps Access-Control-Allow-Origin downstream and
            # a keep-list without it strips ACAO from every 304 — which makes
            # the browser reject the revalidation and refetch the full payload,
            # turning this whole layer into a no-op for every cross-origin
            # caller (i.e. all of fineas.ai).
            keep = (
                b"etag", b"cache-control", b"vary", b"date",
                b"access-control-allow-origin",
                b"access-control-allow-credentials",
                b"access-control-expose-headers",
            )
            not_modified = [(k, v) for k, v in raw if k.lower() in keep]
            await send({"type": "http.response.start", "status": 304,
                        "headers": not_modified})
            await send({"type": "http.response.body", "body": b""})
            return

        await send({**start_message, "headers": raw})
        await send({"type": "http.response.body", "body": body})


def _etag_matches(if_none_match: str, etag: str) -> bool:
    """RFC 9110 If-None-Match: `*`, or any listed tag, weak comparison."""
    candidate = if_none_match.strip()
    if candidate == "*":
        return True
    normalized = etag[2:] if etag.startswith("W/") else etag
    for tag in (t.strip() for t in candidate.split(",")):
        bare = tag[2:] if tag.startswith("W/") else tag
        if bare and bare == normalized:
            return True
    return False


def install_http_cache(app) -> None:
    """Install compression + ETag/Cache-Control on a FastAPI app.

    Order matters. Starlette runs the LAST-added middleware outermost, so
    adding ETag first and GZip second puts GZip on the outside: the ETag is
    computed over the raw JSON (stable regardless of whether a given client
    negotiated compression) and the bytes go out compressed.
    """
    from starlette.middleware.gzip import GZipMiddleware

    app.add_middleware(ETagCacheMiddleware)
    # Railway's edge already gzips, but only for traffic that goes through it —
    # direct hits, other hosts, and local runs got none. 176KB → ~10KB.
    app.add_middleware(GZipMiddleware, minimum_size=500)
