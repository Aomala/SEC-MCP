"""The `/v1` auth + quota dependency.

One FastAPI dependency does the whole gate:

    key → plan → scope check → quota check → (serve) → meter

Design decisions worth knowing:

* **Anonymous is a plan, not a rejection.** A caller with no key resolves to the
  `anon` plan, which has a small budget and a narrow scope list. That keeps the
  docs page and the demo interactive while making a key the obvious next step —
  we are trying to convert developers, not bounce them.

* **Rejections do not consume quota.** A 429 records the call for analytics with
  `consume=False`. Charging a caller for the request that told them they are out
  of budget is how you get support tickets.

* **Rate-limit headers are always present**, on success and on rejection, so a
  client can back off intelligently instead of hammering.

* **The Authorization: Bearer form is accepted** alongside `X-API-Key`, because
  that is what every HTTP client and AI-agent framework does by default.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse

from sec_mcp.commerce import keys as key_store
from sec_mcp.commerce import plans as plan_catalogue
from sec_mcp.commerce import usage

log = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    """Raised by the dependency; converted to a 429 by the app's handler."""

    def __init__(self, verdict: dict, plan_name: str):
        self.verdict = verdict
        self.plan_name = plan_name
        super().__init__(verdict.get("reason", "quota"))


class ScopeDenied(Exception):
    def __init__(self, scope: str, plan_name: str):
        self.scope = scope
        self.plan_name = plan_name
        super().__init__(f"scope {scope} not in plan {plan_name}")


@dataclass
class Caller:
    """Who is calling, and what they are allowed to do."""

    key_id: str
    key_prefix: str
    plan_name: str
    plan: dict
    owner_email: str = ""
    authenticated: bool = False
    verdict: dict = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)

    @property
    def history_years(self) -> int | None:
        return self.plan.get("history_years")

    def may(self, scope: str) -> bool:
        return plan_catalogue.plan_allows(self.plan, scope)

    def rate_headers(self) -> dict[str, str]:
        v = self.verdict or {}
        headers = {
            "X-RateLimit-Limit": str(v.get("limit", 0)),
            "X-RateLimit-Remaining": str(v.get("remaining", 0)),
            "X-RateLimit-Reset": str(v.get("reset_epoch", 0)),
            "X-Plan": self.plan_name,
        }
        if v.get("rpm_limit"):
            headers["X-RateLimit-Limit-Minute"] = str(v["rpm_limit"])
            headers["X-RateLimit-Remaining-Minute"] = str(v.get("rpm_remaining", 0))
        return headers


def _identify(request: Request) -> tuple[str, str]:
    """Extract the presented secret. Returns (raw_key, source)."""
    raw = (request.headers.get("x-api-key") or "").strip()
    if raw:
        return raw, "header"
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip(), "bearer"
    # Query param is a last resort: it leaks into access logs and Referer, so we
    # accept it only because browser <img>/EventSource callers cannot set headers.
    qp = (request.query_params.get("api_key") or "").strip()
    if qp:
        return qp, "query"
    return "", "none"


def _anon_identity(request: Request) -> str:
    """Quota bucket for keyless callers: their IP, behind Railway's proxy."""
    fwd = request.headers.get("x-forwarded-for") or ""
    ip = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown")
    return f"anon:{ip}"


def resolve_caller(request: Request) -> Caller:
    """Identify the caller and attach their plan. Does not enforce anything."""
    raw, _source = _identify(request)
    record = key_store.lookup(raw) if raw else None

    if record:
        plan_name = (record.get("plan") or "free").lower()
        return Caller(
            key_id=str(record.get("id")),
            key_prefix=record.get("key_prefix", ""),
            plan_name=plan_name,
            plan=plan_catalogue.get_plan(plan_name),
            owner_email=record.get("owner_email", ""),
            authenticated=True,
        )

    # A key was presented but did not resolve — treat as anonymous rather than
    # 401. It is one fewer support email, and the anon budget is small enough
    # that a typo'd key gets noticed fast via X-Plan: anon in the response.
    return Caller(
        key_id=_anon_identity(request),
        key_prefix="",
        plan_name="anon",
        plan=plan_catalogue.get_plan("anon"),
        authenticated=False,
    )


def require(scope: str):
    """Build the dependency for a route in `scope`.

        @router.get("/v1/fundamentals/{ticker}")
        def fundamentals(ticker: str, caller: Caller = Depends(require("fundamentals"))):
    """

    def _dependency(request: Request) -> Caller:
        caller = resolve_caller(request)

        if not caller.may(scope):
            usage.record(caller.key_id, f"{scope}:denied", consume=False, error=True)
            raise ScopeDenied(scope, caller.plan_name)

        verdict = usage.check(caller.key_id, caller.plan)
        caller.verdict = verdict
        if not verdict["allowed"]:
            usage.record(caller.key_id, f"{scope}:limited", consume=False, error=True)
            raise QuotaExceeded(verdict, caller.plan_name)

        usage.record(caller.key_id, _endpoint_label(request))
        # Stashed so the response middleware can attach rate headers without
        # every handler having to remember to return them.
        request.state.caller = caller
        return caller

    return _dependency


def _endpoint_label(request: Request) -> str:
    """Route template, not the concrete path — so /v1/fundamentals/AAPL and
    /v1/fundamentals/MSFT roll up to one row instead of one row per ticker."""
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    return f"{request.method} {path}"


def install_error_handlers(app) -> None:
    """Turn the dependency's exceptions into the documented error envelope."""

    @app.exception_handler(QuotaExceeded)
    async def _quota(request: Request, exc: QuotaExceeded):
        v = exc.verdict
        rate_limited = v.get("reason") == "rate_limited"
        body = {
            "error": ("Rate limit exceeded — too many requests per minute."
                      if rate_limited else
                      "Daily quota exceeded for your plan."),
            "code": v.get("reason"),
            "plan": exc.plan_name,
            "limit": v.get("rpm_limit") if rate_limited else v.get("limit"),
            "resetAt": v.get("reset_epoch"),
            "hint": ("Slow down, or upgrade at /pricing for a higher burst limit."
                     if rate_limited else
                     "Quota resets at 00:00 UTC. Upgrade at /pricing for more calls."),
        }
        headers = {
            "Retry-After": str(v.get("retry_after", 60)),
            "X-RateLimit-Limit": str(v.get("limit", 0)),
            "X-RateLimit-Remaining": str(v.get("remaining", 0)),
            "X-RateLimit-Reset": str(v.get("reset_epoch", 0)),
            "X-Plan": exc.plan_name,
        }
        return JSONResponse(status_code=429, content=body, headers=headers)

    @app.exception_handler(ScopeDenied)
    async def _scope(request: Request, exc: ScopeDenied):
        return JSONResponse(
            status_code=403,
            content={
                "error": f"Your plan does not include '{exc.scope}' data.",
                "code": "scope_denied",
                "plan": exc.plan_name,
                "requiredScope": exc.scope,
                "hint": "See /v1/plans for what each tier includes.",
            },
            headers={"X-Plan": exc.plan_name},
        )


def install_rate_headers(app) -> None:
    """Attach rate-limit headers to every successful /v1 response."""

    @app.middleware("http")
    async def _rate_headers(request: Request, call_next):
        response = await call_next(request)
        caller = getattr(request.state, "caller", None)
        if caller is not None:
            for k, v in caller.rate_headers().items():
                response.headers[k] = v
            # Latency is recorded here, where we actually know it. Metering
            # must never be able to break a response that already succeeded.
            with contextlib.suppress(Exception):
                usage.record(
                    caller.key_id, _endpoint_label(request),
                    duration_ms=(time.monotonic() - caller.started) * 1000,
                    error=response.status_code >= 500,
                    consume=False,
                )
        return response
