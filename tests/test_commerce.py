"""Commerce layer: plans, key hashing, quota, scope gating, HTTP caching.

Offline by design — the key store and Supabase are stubbed, so these run in CI
without network or a database. What they pin down is the logic that decides
whether a paying customer gets served, which is the part that must not regress
silently.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sec_mcp.commerce import keys as key_store
from sec_mcp.commerce import plans as plan_catalogue
from sec_mcp.commerce import usage


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Every test starts with empty counters and no plan overrides."""
    plan_catalogue.reset_cache()
    key_store.invalidate()
    usage._daily.clear()
    usage._minute.clear()
    usage._pending.clear()
    # Never let a test flush to a real database.
    monkeypatch.setattr(usage, "_client", lambda: None)
    monkeypatch.setattr(key_store, "_client", lambda: None)
    yield
    plan_catalogue.reset_cache()


# ── Plans ──────────────────────────────────────────────────────────────────

def test_unknown_plan_falls_back_to_free_never_errors():
    assert plan_catalogue.get_plan("nonsense")["label"] == "Free"
    assert plan_catalogue.get_plan(None)["label"] == "Free"
    assert plan_catalogue.get_plan("")["label"] == "Free"


def test_plan_ladder_is_monotonic():
    """A higher price must never buy a smaller quota — an easy config typo."""
    ladder = ["free", "starter", "pro", "business"]
    prev_calls = prev_rpm = -1
    for name in ladder:
        plan = plan_catalogue.get_plan(name)
        assert plan["daily_calls"] > prev_calls, f"{name} daily_calls not increasing"
        assert plan["rpm"] > prev_rpm, f"{name} rpm not increasing"
        prev_calls, prev_rpm = plan["daily_calls"], plan["rpm"]


def test_wildcard_scope_allows_everything():
    pro = plan_catalogue.get_plan("pro")
    for scope in plan_catalogue.ALL_SCOPES:
        assert plan_catalogue.plan_allows(pro, scope)


def test_anon_cannot_reach_paid_scopes():
    anon = plan_catalogue.get_plan("anon")
    assert not plan_catalogue.plan_allows(anon, "ownership")
    assert not plan_catalogue.plan_allows(anon, "bulk")
    assert plan_catalogue.plan_allows(anon, "fundamentals")


def test_env_override_changes_pricing_without_code_change(monkeypatch):
    monkeypatch.setenv("API_PLANS", '{"starter": {"price_usd_month": 79}}')
    plan_catalogue.reset_cache()
    assert plan_catalogue.get_plan("starter")["price_usd_month"] == 79
    # Untouched fields survive the merge.
    assert plan_catalogue.get_plan("starter")["daily_calls"] == 10_000


def test_malformed_override_does_not_break_the_ladder(monkeypatch):
    monkeypatch.setenv("API_PLANS", "{not json at all")
    plan_catalogue.reset_cache()
    assert plan_catalogue.get_plan("pro")["daily_calls"] == 100_000


def test_public_catalogue_hides_internal_tiers():
    names = {p["plan"] for p in plan_catalogue.public_catalogue()}
    assert "internal" not in names and "anon" not in names
    assert {"free", "starter", "pro"} <= names


# ── Keys ───────────────────────────────────────────────────────────────────

def test_generated_key_is_never_recoverable_from_stored_fields():
    full, visible, digest = key_store.generate()
    assert full.startswith("fin_live_")
    assert digest == key_store.hash_key(full)
    # The two things we persist must not reconstruct the secret.
    assert full not in visible
    assert full not in digest
    assert len(visible) < len(full)


def test_keys_are_unique_and_high_entropy():
    seen = {key_store.generate()[0] for _ in range(500)}
    assert len(seen) == 500


def test_test_mode_keys_are_visually_distinct():
    assert key_store.generate("test")[0].startswith("fin_test_")


def test_lookup_without_a_backend_returns_none_not_an_exception():
    assert key_store.lookup("fin_live_whatever") is None
    assert key_store.lookup("") is None


# ── Quota + rate limiting ──────────────────────────────────────────────────

def test_quota_blocks_only_after_the_limit_is_reached():
    plan = {"daily_calls": 3, "rpm": 100}
    for _ in range(3):
        assert usage.check("k1", plan)["allowed"]
        usage.record("k1", "GET /v1/x")
    verdict = usage.check("k1", plan)
    assert not verdict["allowed"]
    assert verdict["reason"] == "quota_exceeded"
    assert verdict["retry_after"] > 0


def test_rate_limit_trips_before_the_daily_quota():
    plan = {"daily_calls": 1_000, "rpm": 2}
    for _ in range(2):
        assert usage.check("k2", plan)["allowed"]
        usage.record("k2", "GET /v1/x")
    verdict = usage.check("k2", plan)
    assert verdict["reason"] == "rate_limited"
    # A minute-limited caller still has daily budget left — the message must
    # not tell them to wait until tomorrow.
    assert verdict["remaining"] > 0
    assert verdict["retry_after"] <= 60


def test_rejected_calls_do_not_consume_quota():
    plan = {"daily_calls": 10, "rpm": 100}
    usage.record("k3", "denied", consume=False)
    usage.record("k3", "denied", consume=False)
    assert usage.check("k3", plan)["remaining"] == 9  # only this check's own call


def test_check_is_read_only():
    plan = {"daily_calls": 5, "rpm": 100}
    before = usage.check("k4", plan)["remaining"]
    usage.check("k4", plan)
    usage.check("k4", plan)
    assert usage.check("k4", plan)["remaining"] == before


def test_keys_have_independent_budgets():
    plan = {"daily_calls": 2, "rpm": 100}
    usage.record("a", "x")
    usage.record("a", "x")
    assert not usage.check("a", plan)["allowed"]
    assert usage.check("b", plan)["allowed"]


def test_failed_flush_retains_counts_for_retry():
    usage.record("k5", "GET /v1/x")
    assert usage._pending  # queued
    usage.flush()          # no client configured → returns 0, keeps the delta
    assert usage.check("k5", {"daily_calls": 10, "rpm": 10})["remaining"] == 8


# ── HTTP surface ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from sec_mcp.chat_app import app
    return TestClient(app)


def test_plans_endpoint_is_public(client):
    r = client.get("/v1/plans")
    assert r.status_code == 200
    assert len(r.json()["plans"]) >= 3


def test_unauthenticated_caller_gets_the_anon_plan(client):
    r = client.get("/v1/me")
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is False
    assert body["plan"] == "anon"


def test_every_metered_response_carries_rate_headers(client):
    r = client.get("/v1/me")
    for header in ("X-RateLimit-Limit", "X-RateLimit-Remaining",
                   "X-RateLimit-Reset", "X-Plan"):
        assert header in r.headers, f"missing {header}"


def test_out_of_scope_endpoint_returns_403_with_the_scope_named(client):
    r = client.get("/v1/ownership/AAPL/insiders")
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "scope_denied"
    assert body["requiredScope"] == "ownership"


def test_bulk_requires_a_paid_plan(client):
    r = client.get("/v1/bulk/fundamentals?universe=dow30")
    assert r.status_code == 403


def test_signup_rejects_a_malformed_email(client):
    r = client.post("/v1/keys", json={"email": "not-an-email"})
    assert r.status_code == 422
    assert r.json()["code"] == "invalid_email"


def test_revoking_without_the_key_is_forbidden(client):
    r = client.delete("/v1/keys/fin_live_abcd1234")
    assert r.status_code == 403


def test_usage_endpoint_requires_a_key(client):
    assert client.get("/v1/usage").status_code == 401


def test_billing_status_reports_configuration_honestly(client):
    body = client.get("/v1/billing/status").json()
    assert "configured" in body and "missingPrices" in body


def test_checkout_without_stripe_returns_503_not_a_crash(client, monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    r = client.post("/v1/billing/checkout", json={"email": "a@b.com", "plan": "pro"})
    assert r.status_code == 503
    assert r.json()["code"] == "billing_unconfigured"


def test_webhook_rejects_an_unsigned_payload(client, monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_INSECURE", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    r = client.post("/v1/billing/webhook", json={"type": "checkout.session.completed"})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_signature"


# ── HTTP caching layer ─────────────────────────────────────────────────────

def test_cacheable_response_carries_an_etag_and_policy(client):
    r = client.get("/v1/plans")
    assert r.headers.get("etag")
    # /v1/ prefix policy
    assert "max-age" in r.headers.get("cache-control", "")


def test_matching_if_none_match_returns_304_with_no_body(client):
    first = client.get("/v1/plans")
    etag = first.headers["etag"]
    second = client.get("/v1/plans", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert not second.content


def test_stale_etag_returns_the_full_body(client):
    r = client.get("/v1/plans", headers={"If-None-Match": '"not-the-right-hash"'})
    assert r.status_code == 200
    assert r.content


def test_if_none_match_wildcard_matches(client):
    client.get("/v1/plans")
    assert client.get("/v1/plans", headers={"If-None-Match": "*"}).status_code == 304


def test_error_responses_are_not_given_an_etag(client):
    """A 403 must never be cached as though it were the resource."""
    r = client.get("/v1/ownership/AAPL/insiders")
    assert r.status_code == 403
    assert not r.headers.get("etag")


# ── Concurrency contract ───────────────────────────────────────────────────

def test_no_route_handler_blocks_the_event_loop():
    """The regression guard for the 45s server-wide stall.

    A handler that does blocking work must be a plain `def` so FastAPI runs it
    on the threadpool. Only handlers that genuinely await an ASGI primitive may
    be `async def`.
    """
    import inspect

    from sec_mcp.chat_app import app

    allowed_async = {
        "validate_metric_endpoint",  # await request.json()
        "run_screener",              # await request.json()
        "v1_screen",                 # await request.json()
        "stripe_webhook",            # await request.body()
    }
    offenders = []
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None or not inspect.iscoroutinefunction(endpoint):
            continue
        name = endpoint.__name__
        # FastAPI's own built-ins (swagger/openapi/redoc) are genuinely async.
        if name in allowed_async or getattr(route, "path", "").startswith(
                ("/openapi", "/swagger", "/redoc", "/docs/oauth2")):
            continue
        offenders.append(f"{getattr(route, 'path', '?')} ({name})")
    assert not offenders, (
        "These handlers are `async def` but do blocking work — they will stall "
        f"the whole server: {offenders}")


# ── Regressions found by audit — each of these shipped broken once ─────────

def test_legacy_v1_paths_still_resolve(client):
    """/v1/financials and /v1/search predate the router and are still called by
    static/app.js (peer comps) and the docs console. Renaming them 404'd the
    live frontend."""
    assert client.get("/v1/search?q=apple").status_code == 200
    # Route must EXIST (not 404); data availability is a separate concern.
    assert client.get("/v1/financials/AAPL").status_code != 404


def test_304_preserves_cors_headers(client):
    """CORSMiddleware is installed inside the ETag layer, so it stamps ACAO
    downstream. A keep-list without it strips ACAO from every 304, the browser
    rejects the revalidation, and the caching layer becomes a no-op for every
    cross-origin caller."""
    origin = {"Origin": "https://fineas.ai"}
    first = client.get("/v1/plans", headers=origin)
    assert "access-control-allow-origin" in first.headers

    second = client.get("/v1/plans",
                        headers={**origin, "If-None-Match": first.headers["etag"]})
    assert second.status_code == 304
    assert "access-control-allow-origin" in second.headers, \
        "304 dropped ACAO — cross-origin revalidation will fail"


def test_rate_limit_headers_are_exposed_to_browsers(client):
    """A custom header a browser cannot read is a header that does not exist."""
    exposed = client.get("/v1/plans", headers={"Origin": "https://fineas.ai"}) \
        .headers.get("access-control-expose-headers", "")
    for header in ("X-RateLimit-Remaining", "X-Plan", "Retry-After"):
        assert header in exposed, f"{header} not exposed to cross-origin JS"


def test_alias_writes_survive_concurrent_handlers(tmp_path, monkeypatch):
    """Handlers run on a 96-thread pool, and learn_company does a
    read-modify-write on one JSON file. Unserialized, that loses entries — it
    wiped 293 aliases in a single local run."""
    import concurrent.futures as cf

    from sec_mcp import intent_parser

    target = tmp_path / "aliases.json"
    target.write_text('{"seed": "SEED"}')
    monkeypatch.setattr(intent_parser, "_ALIASES_PATH", target)

    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(lambda i: intent_parser._save_alias(f"co{i}", f"T{i}"), range(200)))

    final = intent_parser._load_aliases()
    assert len(final) == 201, f"lost entries under concurrency: {len(final)} != 201"
    assert final["seed"] == "SEED"


def test_history_worker_default_is_one():
    """Fanning out the history build is slower, not faster: workers stampede a
    lockless submissions cache and then serialize behind the SEC rate lock
    (measured 1w 9.3-10.0s vs 8w 12.5-14.5s)."""
    import os

    from sec_mcp import financials
    assert hasattr(financials, "os"), "financials.py must import os"
    assert int(os.environ.get("SEC_HISTORY_WORKERS", "1")) == 1
