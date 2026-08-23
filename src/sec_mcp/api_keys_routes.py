"""Self-serve signup, key management, and billing endpoints.

Kept out of api_v1.py because these are account-management routes, not data
routes: they are not metered by plan scope, and they must keep working when a
customer is out of quota.

Anti-abuse posture: key creation is rate-limited per IP and per email. There is
no email-verification loop yet — the free tier's quota is small enough that
farming keys buys little, and adding a verification step before there is a
single customer is the wrong order of work. Noted as a real gap in the docstring
rather than pretended away.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sec_mcp.commerce import billing, usage
from sec_mcp.commerce import keys as key_store
from sec_mcp.commerce import plans as plan_catalogue

log = logging.getLogger(__name__)

router = APIRouter(tags=["account"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")

# Key-creation throttle: in-memory, per UTC day.
_MAX_KEYS_PER_EMAIL = 5
_MAX_KEYS_PER_IP_DAY = 10
_signup_lock = threading.Lock()
_signups: dict[str, tuple[float, int]] = {}


def _throttle(bucket: str, limit: int, window: float = 86400.0) -> bool:
    """True if allowed. Fixed window, resets `window` seconds after first hit."""
    now = time.time()
    with _signup_lock:
        first, count = _signups.get(bucket, (now, 0))
        if now - first > window:
            first, count = now, 0
        if count >= limit:
            return False
        _signups[bucket] = (first, count + 1)
        return True


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for") or ""
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SignupRequest(BaseModel):
    email: str = Field(..., description="Where the key belongs")
    name: str = Field("default", description="Label for this key")


@router.post("/v1/keys", summary="Create a free API key")
def create_key(body: SignupRequest, request: Request):
    """Issue a free-tier key. The secret is returned ONCE and never again."""
    email = (body.email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return JSONResponse(status_code=422, content={
            "error": "A valid email address is required.",
            "code": "invalid_email",
            "hint": "The key is tied to this address for recovery and billing."})

    if not _throttle(f"ip:{_client_ip(request)}", _MAX_KEYS_PER_IP_DAY):
        return JSONResponse(status_code=429, content={
            "error": "Too many keys created from this address today.",
            "code": "signup_throttled",
            "hint": "Reuse your existing key, or contact us for a higher tier."})

    existing = [k for k in key_store.list_for_owner(email) if k.get("status") == "active"]
    if len(existing) >= _MAX_KEYS_PER_EMAIL:
        return JSONResponse(status_code=429, content={
            "error": f"This email already has {len(existing)} active keys.",
            "code": "key_limit",
            "hint": "Revoke one at DELETE /v1/keys/{prefix} before creating another."})

    created = key_store.create(email, plan="free", name=body.name or "default")
    if not created:
        return JSONResponse(status_code=503, content={
            "error": "Key store unavailable — could not issue a key.",
            "code": "store_unavailable",
            "hint": "Try again shortly; if it persists, contact support."})

    plan = plan_catalogue.get_plan("free")
    return {
        **created,
        "warning": "Store this key now — it is not retrievable later.",
        "limits": {
            "dailyCalls": plan.get("daily_calls"),
            "requestsPerMinute": plan.get("rpm"),
            "historyYears": plan.get("history_years"),
            "scopes": plan.get("scopes"),
        },
        "quickstart": {
            "curl": ("curl -H 'X-API-Key: <your key>' "
                     "https://sec-mcp-production.up.railway.app/v1/fundamentals/AAPL"),
            "docs": "/docs",
        },
    }


@router.get("/v1/keys", summary="List the keys for an email (prefixes only)")
def list_keys(email: str):
    """Returns prefixes and metadata. Secrets are unrecoverable by design."""
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return JSONResponse(status_code=422, content={
            "error": "A valid email address is required.", "code": "invalid_email",
            "hint": "Pass ?email=you@example.com"})
    return {"email": email, "keys": key_store.list_for_owner(email)}


@router.delete("/v1/keys/{key_prefix}", summary="Revoke a key")
def revoke_key(key_prefix: str, request: Request):
    """Revoke by prefix. Requires the key itself, so only its holder can do it."""
    presented = (request.headers.get("x-api-key") or "").strip()
    record = key_store.lookup(presented) if presented else None
    if not record or record.get("key_prefix") != key_prefix:
        return JSONResponse(status_code=403, content={
            "error": "Send the key being revoked in the X-API-Key header.",
            "code": "revoke_forbidden",
            "hint": "A key can only revoke itself."})
    ok = key_store.revoke(key_prefix)
    return {"revoked": ok, "keyPrefix": key_prefix}


@router.get("/v1/usage", summary="Usage history for the presented key")
def key_usage(request: Request, days: int = 30):
    presented = (request.headers.get("x-api-key") or "").strip()
    record = key_store.lookup(presented) if presented else None
    if not record:
        return JSONResponse(status_code=401, content={
            "error": "Send your key in the X-API-Key header.",
            "code": "unauthenticated", "hint": "Get one at POST /v1/keys."})
    return {
        "keyPrefix": record.get("key_prefix"),
        "plan": record.get("plan"),
        **usage.summary_for_key(str(record.get("id")), days=max(1, min(days, 365))),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Billing
# ═══════════════════════════════════════════════════════════════════════════

class CheckoutRequest(BaseModel):
    email: str
    plan: str = Field("pro", description="pro | business | enterprise")


@router.post("/v1/billing/checkout", summary="Start a Stripe Checkout session")
def checkout(body: CheckoutRequest):
    email = (body.email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return JSONResponse(status_code=422, content={
            "error": "A valid email address is required.", "code": "invalid_email",
            "hint": "The subscription and the key are keyed to this address."})
    result = billing.create_checkout_session(email, (body.plan or "pro").lower())
    if "error" in result:
        code = 503 if result.get("code") == "billing_unconfigured" else 400
        return JSONResponse(status_code=code, content=result)
    return result


@router.post("/v1/billing/webhook", summary="Stripe webhook receiver")
async def stripe_webhook(request: Request):
    """Signature-verified. Always 200s on a verified event, even an unhandled
    one — a non-2xx makes Stripe retry the same event indefinitely."""
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    event = billing.verify_webhook(payload, signature)
    if event is None:
        return JSONResponse(status_code=400, content={
            "error": "Invalid webhook signature.", "code": "bad_signature",
            "hint": "Check STRIPE_WEBHOOK_SECRET matches the endpoint in Stripe."})
    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(billing.handle_event, event)
        # A minted key must never reach the log or the HTTP response.
        result.pop("key", None)
        return {"received": True, **result}
    except Exception as exc:
        log.exception("Stripe webhook handling failed")
        return JSONResponse(status_code=500, content={
            "error": "Webhook handling failed.", "code": "webhook_error",
            "hint": str(exc)})


@router.get("/v1/billing/status", summary="Is billing wired up?")
def billing_status():
    return billing.status()
