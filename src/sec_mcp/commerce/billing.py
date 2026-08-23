"""Stripe billing — checkout, webhook, and the plan changes they cause.

Scope is deliberately narrow. Stripe owns subscription state; we own one field,
`api_keys.plan`. The webhook's whole job is to keep that field agreeing with
Stripe. No invoice logic, no proration math, no local mirror of subscription
objects to drift out of sync.

Flow:
  1. POST /v1/billing/checkout {email, plan}  → Stripe Checkout URL
  2. customer pays
  3. Stripe → POST /v1/billing/webhook
  4. webhook maps price_id → plan and updates every key owned by that email,
     minting one if they had none

Configuration (all env, nothing hardcoded):
  STRIPE_SECRET_KEY       sk_live_… / sk_test_…
  STRIPE_WEBHOOK_SECRET   whsec_…  — signature verification, REQUIRED in prod
  STRIPE_PRICE_STARTER    price_…
  STRIPE_PRICE_PRO        price_…
  STRIPE_PRICE_BUSINESS   price_…
  BILLING_SUCCESS_URL / BILLING_CANCEL_URL

Absent STRIPE_SECRET_KEY the module reports unconfigured and the endpoints
return a clean 503 — the rest of the API is unaffected.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def _price_map() -> dict[str, str]:
    """price_id → plan name, built from env so prices can rotate without a deploy."""
    pairs = {
        "starter": os.environ.get("STRIPE_PRICE_STARTER", "").strip(),
        "pro": os.environ.get("STRIPE_PRICE_PRO", "").strip(),
        "business": os.environ.get("STRIPE_PRICE_BUSINESS", "").strip(),
    }
    return {price: plan for plan, price in pairs.items() if price}


def _plan_price(plan: str) -> str | None:
    return {v: k for k, v in _price_map().items()}.get(plan)


def is_configured() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def _stripe():
    """Import + configure the SDK lazily so stripe stays an optional dependency."""
    import stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"].strip()
    return stripe


def create_checkout_session(email: str, plan: str) -> dict:
    """Return {url} for Stripe Checkout, or {error, code, hint}."""
    if not is_configured():
        return {"error": "Billing is not configured on this deployment.",
                "code": "billing_unconfigured",
                "hint": "Set STRIPE_SECRET_KEY to enable self-serve checkout."}

    price_id = _plan_price(plan)
    if not price_id:
        return {"error": f"No Stripe price configured for plan '{plan}'.",
                "code": "unknown_plan",
                "hint": "Paid plans are: starter, pro, business. See /v1/plans."}

    try:
        stripe = _stripe()
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=os.environ.get(
                "BILLING_SUCCESS_URL",
                "https://fineas.ai/api/welcome?session_id={CHECKOUT_SESSION_ID}"),
            cancel_url=os.environ.get("BILLING_CANCEL_URL", "https://fineas.ai/pricing"),
            # Echoed back on every webhook for this subscription, so the handler
            # never has to guess which plan was bought.
            metadata={"plan": plan, "email": email},
            subscription_data={"metadata": {"plan": plan, "email": email}},
            allow_promotion_codes=True,
        )
        return {"url": session.url, "sessionId": session.id, "plan": plan}
    except Exception as exc:
        log.error("Stripe checkout failed: %s", exc)
        return {"error": "Could not start checkout.", "code": "stripe_error",
                "hint": str(exc)}


def verify_webhook(payload: bytes, signature: str):
    """Verify and parse a webhook. Returns the event, or None if untrusted.

    An unverified webhook is an open door to free plan upgrades, so a missing
    secret is treated as a hard failure in production rather than being waved
    through. Only an explicit STRIPE_WEBHOOK_INSECURE=1 (local testing) skips it.
    """
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        if os.environ.get("STRIPE_WEBHOOK_INSECURE") == "1":
            import json
            log.warning("Stripe webhook signature check DISABLED (dev mode)")
            return json.loads(payload)
        log.error("STRIPE_WEBHOOK_SECRET is not set — rejecting webhook")
        return None
    try:
        stripe = _stripe()
        return stripe.Webhook.construct_event(payload, signature, secret)
    except Exception as exc:
        log.error("Stripe webhook verification failed: %s", exc)
        return None


def _apply_plan(email: str, plan: str, **extra) -> dict:
    """Move every key owned by `email` to `plan`, minting one if they have none."""
    from sec_mcp.commerce import keys as key_store

    email = (email or "").strip().lower()
    if not email:
        return {"ok": False, "reason": "no_email"}

    existing = key_store.list_for_owner(email)
    active = [k for k in existing if k.get("status") == "active"]

    if not active:
        # Paid before they ever took a free key — mint one now. It is returned
        # to the caller of this function, never logged, and the welcome email
        # is what actually delivers it.
        created = key_store.create(email, plan=plan, name="subscription")
        return {"ok": bool(created), "created": True,
                "keyPrefix": (created or {}).get("keyPrefix"),
                "key": (created or {}).get("key"), "plan": plan}

    for k in active:
        key_store.set_plan(k["key_prefix"], plan, **extra)
    return {"ok": True, "created": False, "plan": plan,
            "updated": [k["key_prefix"] for k in active]}


def handle_event(event) -> dict:
    """Apply one Stripe event. Unknown types are acknowledged and ignored —
    returning an error would make Stripe retry forever."""
    etype = event.get("type") if isinstance(event, dict) else getattr(event, "type", "")
    data = (event.get("data") or {}).get("object", {}) if isinstance(event, dict) else {}

    if etype == "checkout.session.completed":
        meta = data.get("metadata") or {}
        email = (meta.get("email") or data.get("customer_email")
                 or (data.get("customer_details") or {}).get("email") or "")
        plan = meta.get("plan") or "starter"
        result = _apply_plan(
            email, plan,
            stripe_customer_id=data.get("customer"),
            stripe_subscription_id=data.get("subscription"),
        )
        log.info("Billing: %s → %s (created=%s)", email, plan, result.get("created"))
        return result

    if etype in ("customer.subscription.updated", "customer.subscription.created"):
        meta = data.get("metadata") or {}
        email = meta.get("email") or ""
        # Trust the live price over the metadata: metadata is written once at
        # checkout, but the price changes on every upgrade and downgrade.
        items = ((data.get("items") or {}).get("data") or [])
        price_id = (items[0].get("price") or {}).get("id") if items else None
        plan = _price_map().get(price_id) or meta.get("plan") or "starter"
        if data.get("status") in ("canceled", "unpaid", "incomplete_expired"):
            plan = "free"
        return _apply_plan(email, plan)

    if etype == "customer.subscription.deleted":
        meta = data.get("metadata") or {}
        # Downgrade, never revoke: the key keeps working on the free tier. A
        # dead key on cancellation generates support load and kills win-backs.
        return _apply_plan(meta.get("email") or "", "free")

    log.debug("Billing: ignoring event %s", etype)
    return {"ok": True, "ignored": etype}


def status() -> dict:
    """Config health — surfaced on /v1/billing/status so misconfiguration is
    visible before a customer hits it, not after."""
    prices = _price_map()
    return {
        "configured": is_configured(),
        "webhookSecretSet": bool(os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()),
        "pricesConfigured": sorted(set(prices.values())),
        "missingPrices": [p for p in ("starter", "pro", "business")
                          if p not in set(prices.values())],
    }
