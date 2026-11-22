"""Fineas account authentication and shared, atomic request quotas.

Enable only after Fineas's quota migration and internal clients are deployed.
The legacy key store is deliberately not an authentication authority here.
"""

from __future__ import annotations

import hmac
import json
import os
from uuid import UUID

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

from sec_mcp.commerce.auth import Caller
from sec_mcp.commerce.plans import ALL_SCOPES

_PUBLIC = {"/v1", "/v1/plans", "/api/docs", "/api/redoc", "/api/openapi.json"}
_ACCOUNT = {"/v1/me", "/v1/usage", "/v1/keys", "/v1/billing/checkout", "/v1/billing/status"}
_EXPOSE = (
    "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, "
    "X-RateLimit-Limit-Minute, X-RateLimit-Remaining-Minute, X-Plan, Retry-After"
)


def account_gate_enabled() -> bool:
    return os.getenv("FINEAS_ACCOUNT_GATE", "").strip() == "1"


def public_config() -> dict:
    if not account_gate_enabled():
        return {"enabled": False}
    return {
        "enabled": True,
        "supabaseUrl": os.getenv("FINEAS_SUPABASE_URL", "").strip().rstrip("/"),
        "supabaseAnonKey": os.getenv("FINEAS_SUPABASE_ANON_KEY", "").strip(),
        "dailyLimit": 250,
        "minuteLimit": 30,
    }


class GateFailure(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message = status, code, message
        self.headers = headers or {}
        super().__init__(code)


def _settings() -> tuple[str, str]:
    url = os.getenv("FINEAS_SUPABASE_URL", "").strip().rstrip("/")
    anon = os.getenv("FINEAS_SUPABASE_ANON_KEY", "").strip()
    if not url.startswith("https://") or not anon:
        raise GateFailure(503, "configuration_error", "Fineas sign-in is not configured.")
    return url, anon


async def _request(method: str, url: str, *, headers: dict, body: dict | None = None):
    # No retries: a timed-out quota mutation may already have committed.
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
        return await client.request(method, url, headers=headers, json=body)


async def verify_user(request: Request) -> dict:
    auth = request.headers.get("authorization", "")
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise GateFailure(401, "account_required", "Sign in with your Fineas account.")
    url, anon = _settings()
    try:
        response = await _request("GET", f"{url}/auth/v1/user", headers={
            "apikey": anon, "Authorization": f"Bearer {parts[1]}",
        })
    except httpx.HTTPError as exc:
        raise GateFailure(503, "auth_unavailable", "Fineas sign-in is temporarily unavailable.") from exc
    if response.status_code in (401, 403):
        raise GateFailure(401, "invalid_session", "Your session expired. Sign in again.")
    if response.status_code != 200:
        raise GateFailure(503, "auth_unavailable", "Fineas sign-in is temporarily unavailable.")
    try:
        user = response.json()
        user_id = str(UUID(user["id"]))
        # Auth's verified user object is the authority, never decoded JWT claims.
        if user.get("is_anonymous") is not False or user.get("role") != "authenticated":
            raise ValueError("not a registered account")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise GateFailure(401, "invalid_session", "A registered Fineas account is required.") from exc
    return {**user, "id": user_id}


async def consume_quota(user_id: str, units: int = 1) -> dict:
    url, _ = _settings()
    service = os.getenv("FINEAS_SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not service:
        raise GateFailure(503, "configuration_error", "Data access is not configured.")
    try:
        response = await _request(
            "POST", f"{url}/rest/v1/rpc/consume_fineas_data_quota",
            headers={"apikey": service, "Authorization": f"Bearer {service}"},
            body={"p_user_id": user_id, "p_units": units},
        )
        if response.status_code != 200:
            raise ValueError("quota service unavailable")
        rows = response.json()
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("invalid quota response")
        verdict = rows[0]
        if type(verdict.get("allowed")) is not bool:
            raise ValueError("invalid verdict")
        for key in ("limit", "remaining", "reset_epoch", "rpm_limit", "rpm_remaining", "retry_after"):
            if type(verdict.get(key)) is not int or verdict[key] < 0:
                raise ValueError("invalid quota counters")
        if not verdict["allowed"] and verdict.get("reason") not in (
            "quota_exceeded", "rate_limited", "access_disabled",
        ):
            raise ValueError("invalid quota rejection")
        return verdict
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise GateFailure(503, "quota_unavailable", "Usage limits cannot be checked. Try again shortly.") from exc


def _account_caller(user: dict, verdict: dict) -> Caller:
    # Account rollout grants interactive data; large exports remain internal.
    return Caller(
        key_id=f"account:{user['id']}", key_prefix="", plan_name="free",
        plan={"daily_calls": verdict["limit"], "rpm": verdict["rpm_limit"],
              "scopes": list(ALL_SCOPES), "bulk": False, "history_years": None},
        owner_email=user.get("email", ""), authenticated=True, verdict=verdict,
    )


def _internal_only(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in (
        "/api/cache", "/api/ingest", "/api/admin", "/v1/admin",
    ))


def _bulk_export(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in (
        "/api/export", "/api/v1/export", "/v1/bulk",
    ))


async def _mcp_request_units(scope, receive):
    """Read one bounded JSON-RPC call and replay its body to the transport.

    Batches are rejected so one HTTP request cannot execute unmetered tools.
    Handshake, discovery, notifications and SSE connections spend no units.
    """
    if scope["method"] != "POST":
        return 0, receive
    chunks = []
    size = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            raise GateFailure(400, "invalid_mcp_request", "Incomplete MCP request.")
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > 1024 * 1024:
            raise GateFailure(413, "mcp_request_too_large", "MCP requests must be under 1 MiB.")
        chunks.append(chunk)
        if not message.get("more_body"):
            break
    body = b"".join(chunks)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise GateFailure(400, "invalid_mcp_request", "A valid JSON-RPC object is required.") from exc
    if isinstance(payload, list):
        raise GateFailure(400, "mcp_batch_unsupported", "Send MCP tool calls individually.")
    if not isinstance(payload, dict) or not isinstance(payload.get("method"), str):
        raise GateFailure(400, "invalid_mcp_request", "A valid JSON-RPC object is required.")

    pending = True

    async def replay():
        nonlocal pending
        if pending:
            pending = False
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return (1 if payload["method"] == "tools/call" else 0), replay


class FineasAccountMiddleware:
    """Pure ASGI gate, outermost to caches and streaming transports.

    all_paths covers the separate FastMCP HTTP app. Data requests and individual
    MCP tool calls consume one unit; status/account management consume zero.
    """

    def __init__(self, app, all_paths: bool = False):
        self.app, self.all_paths = app, all_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not account_gate_enabled():
            return await self.app(scope, receive, send)
        path = scope.get("path", "").rstrip("/") or "/"
        protected = self.all_paths or path in ("/api", "/v1") or path.startswith(("/api/", "/v1/"))
        public = path in _PUBLIC and scope["method"] in ("GET", "HEAD")
        if scope["method"] == "OPTIONS" or not protected or public:
            return await self.app(scope, receive, send)
        if path == "/v1/billing/webhook" and scope["method"] == "POST":
            return await self.app(scope, receive, send)

        request = Request(scope, receive)
        state = scope.setdefault("state", {})
        state["fineas_account_gate"] = True
        caller = None
        try:
            supplied = request.headers.get("x-fineas-service-key", "")
            secret = os.getenv("FINEAS_SERVICE_KEY", "").strip()
            internal = bool(secret and supplied and hmac.compare_digest(supplied.encode(), secret.encode()))
            account_route = path in _ACCOUNT or path.startswith("/v1/keys/")
            # Account-management routes always prove the human owner, even if
            # an internal service credential was included accidentally.
            if internal and not account_route:
                caller = Caller(key_id="fineas:internal", key_prefix="", plan_name="internal",
                                plan={"scopes": ["*"], "bulk": True, "history_years": None},
                                authenticated=True)
            else:
                if _internal_only(path):
                    raise GateFailure(403, "service_required", "This endpoint is reserved for internal services.")
                user = await verify_user(request)
                state["fineas_user"] = user
                if _bulk_export(path):
                    raise GateFailure(403, "bulk_restricted", "Bulk downloads require a separate data agreement.")
                units = 0 if account_route else 1
                if self.all_paths:
                    units, receive = await _mcp_request_units(scope, receive)
                verdict = await consume_quota(user["id"], units)
                caller = _account_caller(user, verdict)
                if not verdict["allowed"]:
                    headers = {**caller.rate_headers(), "Retry-After": str(verdict["retry_after"])}
                    disabled = verdict["reason"] == "access_disabled"
                    raise GateFailure(403 if disabled else 429, verdict["reason"],
                                      "Data access is disabled." if disabled else "Your free usage limit has been reached.",
                                      headers)
            state["caller"] = caller
        except GateFailure as exc:
            headers = {"Cache-Control": "private, no-store", **exc.headers}
            if exc.status == 401:
                headers["WWW-Authenticate"] = 'Bearer realm="Fineas"'
            if request.headers.get("origin"):
                headers.update({"Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": _EXPOSE})
            response = JSONResponse({"error": exc.message, "code": exc.code}, exc.status, headers=headers)
            return await response(scope, receive, send)

        async def guarded_send(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # Old public cache policy must never let a CDN skip this gate.
                headers["Cache-Control"] = "private, no-store"
                if caller and caller.verdict:
                    headers.update(caller.rate_headers())
            await send(message)

        await self.app(scope, receive, guarded_send)


def install_account_gate(app) -> None:
    @app.get("/auth/config", include_in_schema=False)
    def auth_config():
        return JSONResponse(public_config(), headers={"Cache-Control": "no-store"})

    app.add_middleware(FineasAccountMiddleware)
