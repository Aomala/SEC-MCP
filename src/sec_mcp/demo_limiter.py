"""
Daily usage budget for the public demo site (fineasmcp.vercel.app).

Scoped by Origin/Referer so ONLY browser traffic from the demo site is
metered — Fineas prod calls (Supabase edge functions, fineas.ai) carry a
different Origin or none at all and pass through untouched.

Identity is a per-browser id the frontend persists in localStorage and sends
as X-Demo-Id (20 queries/day). A per-IP backstop (60/day) blunts users who
clear site data to reset the browser budget. Counters are in-memory and reset
daily at UTC midnight — a restart also resets them, which is fine for a demo.

Env knobs:
  DEMO_LIMIT_OFF=1        disable entirely
  DEMO_DAILY_LIMIT        per-browser budget (default 20)
  DEMO_IP_DAILY_LIMIT     per-IP backstop (default 60)
  DEMO_LIMIT_ORIGINS      comma-separated origins to meter
                          (default https://fineasmcp.vercel.app)
"""

import os
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse

# Only expensive, user-initiated actions count as a "try" — browsing filing
# lists and company search stay free so the demo feels alive.
_LIMITED_PREFIXES = (
    "/api/chat",          # also matches /api/chatbot and /api/chatbot-stream
    "/api/enrich/",
    "/api/comps/",
    "/api/filing-text/",
    "/api/export/",
)

_lock = threading.Lock()
_counts: dict = {}
_day: str = ""


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _demo_origins() -> set:
    raw = os.getenv("DEMO_LIMIT_ORIGINS", "https://fineasmcp.vercel.app")
    return {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}


def _is_demo_request(request: Request) -> bool:
    origins = _demo_origins()
    origin = (request.headers.get("origin") or "").rstrip("/")
    if origin:
        return origin in origins
    # Top-level navigations (e.g. the CSV/JSON export links) send no Origin
    # header — fall back to the Referer.
    referer = request.headers.get("referer") or ""
    return any(referer.startswith(o) for o in origins)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for") or ""
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _bump(kind: str, ident: str, limit: int) -> int:
    """Increment (kind, ident) for today; returns remaining (< 0 = over)."""
    global _day
    with _lock:
        today = _today()
        if today != _day:
            _counts.clear()
            _day = today
        key = (kind, ident)
        _counts[key] = _counts.get(key, 0) + 1
        return limit - _counts[key]


def install_demo_limiter(app) -> None:
    browser_limit = int(os.getenv("DEMO_DAILY_LIMIT", "20"))
    ip_limit = int(os.getenv("DEMO_IP_DAILY_LIMIT", "60"))

    @app.middleware("http")
    async def demo_limiter(request: Request, call_next):
        if (
            os.getenv("DEMO_LIMIT_OFF") == "1"
            or request.method == "OPTIONS"
            or not request.url.path.startswith(_LIMITED_PREFIXES)
            or not _is_demo_request(request)
        ):
            return await call_next(request)

        ip_remaining = _bump("ip", _client_ip(request), ip_limit)
        demo_id = (request.headers.get("x-demo-id") or "").strip()[:64]
        browser_remaining = (
            _bump("id", demo_id, browser_limit) if demo_id else browser_limit
        )
        remaining = min(browser_remaining, ip_remaining)

        if remaining < 0:
            # This middleware sits outside CORSMiddleware, so the 429 must
            # carry its own CORS headers or the browser can't read it.
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Demo limit reached",
                    "code": "demo_limit",
                    "detail": (
                        "You've used today's free demo queries. This is the "
                        "live data engine behind Fineas — see it in "
                        "production at https://fineas.ai"
                    ),
                },
                headers={
                    "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                    "Vary": "Origin",
                    "Retry-After": "86400",
                },
            )

        response = await call_next(request)
        response.headers["X-Demo-Remaining"] = str(max(remaining, 0))
        return response
