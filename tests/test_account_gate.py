"""Offline security-boundary tests; no production keys or network access."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from sec_mcp.commerce import account_gate as gate
from sec_mcp.commerce import keys, usage
from sec_mcp.commerce.auth import Caller, install_error_handlers, install_rate_headers, require

USER_ID = "9ef07cc2-62ab-4caf-a897-6f1ea8107c7c"
OTHER_ID = "101707cc-62ab-4caf-a897-6f1ea8107c7c"
USER = {"id": USER_ID, "email": "owner@example.com", "role": "authenticated",
        "is_anonymous": False, "email_confirmed_at": "2026-01-01T00:00:00Z"}
VERDICT = {"allowed": True, "reason": None, "limit": 250, "remaining": 249,
           "reset_epoch": 1900000000, "rpm_limit": 30, "rpm_remaining": 29,
           "retry_after": 0}
AUTH = {"Authorization": "Bearer verified-user-token"}


@pytest.fixture
def network(monkeypatch):
    monkeypatch.setenv("FINEAS_ACCOUNT_GATE", "1")
    monkeypatch.setenv("FINEAS_SUPABASE_URL", "https://fineas-test.supabase.co")
    monkeypatch.setenv("FINEAS_SUPABASE_ANON_KEY", "public-anon")
    monkeypatch.setenv("FINEAS_SUPABASE_SERVICE_ROLE_KEY", "private-database-service")
    monkeypatch.setenv("FINEAS_SERVICE_KEY", "private-origin-service")

    async def answer(method, url, *, headers, body=None):
        if url.endswith("/auth/v1/user"):
            assert method == "GET" and headers["apikey"] == "public-anon"
            if headers["Authorization"] != AUTH["Authorization"]:
                return httpx.Response(401, json={"error": "invalid token"})
            return httpx.Response(200, json=USER)
        assert method == "POST"
        assert url == "https://fineas-test.supabase.co/rest/v1/rpc/consume_fineas_data_quota"
        assert headers["Authorization"] == "Bearer private-database-service"
        assert headers["apikey"] == "private-database-service"
        assert body["p_user_id"] == USER_ID
        return httpx.Response(200, json=[VERDICT])

    mock = AsyncMock(side_effect=answer)
    monkeypatch.setattr(gate, "_request", mock)
    return mock


@pytest.fixture
def client(network):
    from sec_mcp.api_keys_routes import router

    app = FastAPI()

    @app.get("/api/data")
    @app.get("/api/cache/stats")
    @app.get("/api/export/AAPL/csv")
    @app.get("/v1/bulk/fundamentals")
    def data():
        return JSONResponse({"data": "protected"}, headers={"Cache-Control": "public, max-age=3600"})

    needs_fundamentals = Depends(require("fundamentals"))

    @app.get("/v1/data")
    @app.get("/v1/me")
    def v1(caller: Caller = needs_fundamentals):
        return {"plan": caller.plan_name, "remaining": caller.verdict.get("remaining")}

    for path in ("/health", "/docs", "/static/example.js", "/embed/youtube/abcdefghijk", "/v1/plans"):
        app.add_api_route(path, lambda: {"public": True})
    app.include_router(router)
    install_error_handlers(app)
    install_rate_headers(app)
    gate.install_account_gate(app)
    return TestClient(app)


@pytest.mark.parametrize("path", ["/api/data", "/v1/data", "/api/data/", "/v1/me", "/v1/usage"])
def test_anonymous_and_legacy_key_callers_are_rejected(client, network, path):
    for headers in ({}, {"X-API-Key": "legacy-key"}, {"Origin": "https://fineas.ai", "X-Demo-Id": "foo"}):
        response = client.get(path, headers=headers)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "private, no-store"
    network.assert_not_called()


def test_invalid_bearer_never_reaches_quota(client, network):
    response = client.get("/api/data", headers={"Authorization": "Bearer invented-jwt"})
    assert response.status_code == 401
    assert network.await_count == 1


@pytest.mark.parametrize("patch", [{"is_anonymous": True}, {"is_anonymous": None}, {"role": "service_role"}, {"id": "not-a-uuid"}])
def test_rejects_unregistered_or_malformed_auth_users(client, network, patch):
    network.side_effect = None
    network.return_value = httpx.Response(200, json={**USER, **patch})
    assert client.get("/api/data", headers=AUTH).status_code == 401
    assert network.await_count == 1


@pytest.mark.parametrize("path", ["/api/data", "/v1/data"])
def test_both_data_surfaces_share_one_account_quota_and_never_public_cache(client, network, monkeypatch, path):
    monkeypatch.setattr(usage, "check", lambda *a: pytest.fail("legacy quota must not run"))
    monkeypatch.setattr(usage, "record", lambda *a, **kw: pytest.fail("legacy metering must not run"))
    response = client.get(path, headers=AUTH)
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Remaining"] == "249"
    assert response.headers["X-RateLimit-Remaining-Minute"] == "29"
    assert response.headers["Cache-Control"] == "private, no-store"
    assert network.await_args_list[-1].kwargs["body"] == {"p_user_id": USER_ID, "p_units": 1}
    assert network.await_count == 2


@pytest.mark.parametrize("path", ["/v1/me", "/v1/usage"])
def test_status_does_not_spend_quota(client, network, path):
    assert client.get(path, headers=AUTH).status_code == 200
    assert network.await_args_list[-1].kwargs["body"]["p_units"] == 0


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.Response(500, json={}), httpx.Response(200, json=[]), httpx.Response(200, json=[{**VERDICT, "allowed": "true"}])])
def test_quota_errors_fail_closed(client, network, failure):
    network.side_effect = [httpx.Response(200, json=USER), failure]
    response = client.get("/api/data", headers=AUTH)
    assert response.status_code == 503
    assert response.json()["code"] == "quota_unavailable"


def test_auth_outage_fails_closed(client, network):
    network.side_effect = httpx.ConnectError("offline")
    assert client.get("/api/data", headers=AUTH).status_code == 503


@pytest.mark.parametrize("missing", ["FINEAS_SUPABASE_URL", "FINEAS_SUPABASE_ANON_KEY", "FINEAS_SUPABASE_SERVICE_ROLE_KEY"])
def test_incomplete_configuration_never_becomes_anonymous_access(client, monkeypatch, missing):
    monkeypatch.delenv(missing)
    assert client.get("/api/data", headers=AUTH).status_code == 503


@pytest.mark.parametrize("reason,status", [("quota_exceeded", 429), ("rate_limited", 429), ("access_disabled", 403)])
def test_quota_denials_include_backoff_and_cors(client, network, reason, status):
    network.side_effect = [httpx.Response(200, json=USER), httpx.Response(200, json=[{
        **VERDICT, "allowed": False, "reason": reason, "retry_after": 60,
    }])]
    response = client.get("/api/data", headers={**AUTH, "Origin": "https://fineasmcp.vercel.app"})
    assert response.status_code == status
    assert response.headers["Retry-After"] == "60"
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert "X-RateLimit-Remaining" in response.headers["Access-Control-Expose-Headers"]


def test_internal_bypass_only_accepts_separate_service_header(client, network):
    response = client.get("/api/cache/stats", headers={"X-Fineas-Service-Key": "private-origin-service"})
    assert response.status_code == 200
    network.assert_not_called()
    for header in ("Authorization", "X-API-Key"):
        response = client.get("/api/data", headers={header: "Bearer private-origin-service"})
        assert response.status_code == 401


def test_account_management_cannot_use_internal_bypass(client, network):
    response = client.get("/v1/keys", headers={"X-Fineas-Service-Key": "private-origin-service"})
    assert response.status_code == 401
    network.assert_not_called()


def test_maintenance_and_bulk_are_not_interactive_access(client):
    for path in ("/api/cache/stats", "/api/export/AAPL/csv", "/v1/bulk/fundamentals"):
        assert client.get(path, headers=AUTH).status_code == 403


def test_public_routes_and_config_reveal_no_service_credentials(client, network):
    for path in ("/health", "/static/example.js", "/embed/youtube/abcdefghijk", "/v1/plans"):
        assert client.get(path).status_code == 200
    config = client.get("/auth/config").json()
    assert config == {"enabled": True, "supabaseUrl": "https://fineas-test.supabase.co",
                      "supabaseAnonKey": "public-anon", "dailyLimit": 250, "minuteLimit": 30}
    assert "private-" not in json.dumps(config)
    network.assert_not_called()


def test_webhook_remains_signature_verified_without_account(client, monkeypatch, network):
    from sec_mcp.commerce import billing
    monkeypatch.setattr(billing, "verify_webhook", lambda *a: None)
    assert client.post("/v1/billing/webhook", content=b"{}").status_code == 400
    network.assert_not_called()


def test_key_list_cannot_select_another_email(client, monkeypatch):
    monkeypatch.setattr(keys, "list_for_user", lambda uid: [{"key_prefix": uid}])
    assert client.get("/v1/keys?email=victim@example.com", headers=AUTH).status_code == 403
    response = client.get("/v1/keys", headers=AUTH)
    assert response.json()["keys"] == [{"key_prefix": USER_ID}]


def test_key_signup_derives_owner_from_verified_account(client, monkeypatch):
    monkeypatch.setattr(keys, "list_for_user", lambda uid: [])
    created = []
    def create(email, **kwargs):
        created.append((email, kwargs))
        return {"key": "new-secret"}
    monkeypatch.setattr(keys, "create", create)
    assert client.post("/v1/keys", json={"email": "victim@example.com"}, headers=AUTH).status_code == 403
    assert client.post("/v1/keys", json={"name": "mine"}, headers=AUTH).status_code == 200
    assert created[0][0] == USER["email"]
    assert created[0][1]["metadata"] == {"fineas_user_id": USER_ID}


@pytest.mark.parametrize("metadata", [{}, {"fineas_user_id": OTHER_ID}])
def test_legacy_or_foreign_key_possession_does_not_prove_ownership(client, monkeypatch, metadata):
    monkeypatch.setattr(keys, "lookup", lambda key: {"key_prefix": "abc", "metadata": metadata})
    monkeypatch.setattr(keys, "revoke", lambda *a: pytest.fail("must not revoke"))
    assert client.delete("/v1/keys/abc", headers={**AUTH, "X-API-Key": "stolen-key"}).status_code == 403


def test_legacy_checkout_cannot_sell_unprovisioned_account_upgrade(client, monkeypatch):
    from sec_mcp.commerce import billing
    monkeypatch.setattr(billing, "create_checkout_session", lambda *a: pytest.fail("must not charge"))
    response = client.post("/v1/billing/checkout", json={"plan": "pro"}, headers=AUTH)
    assert response.status_code == 503
    assert response.json()["code"] == "account_billing_unavailable"


def test_disabled_rollout_preserves_existing_routes(client, monkeypatch, network):
    monkeypatch.delenv("FINEAS_ACCOUNT_GATE")
    assert client.get("/api/data").status_code == 200
    assert client.get("/auth/config").json() == {"enabled": False}
    network.assert_not_called()


def test_remote_mcp_app_also_rejects_anonymous_requests(network):
    pytest.importorskip("fastmcp")
    from sec_mcp.commerce.mcp import FineasFastMCP
    app = FineasFastMCP("test").http_app()
    response = TestClient(app).post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert response.status_code == 401
    network.assert_not_called()


@pytest.fixture
def mcp_client(network):
    app = FastAPI()

    @app.post("/mcp")
    async def echo(request: Request):
        return await request.json()

    @app.get("/sse")
    def sse():
        return {"connected": True}

    app.add_middleware(gate.FineasAccountMiddleware, all_paths=True)
    return TestClient(app)


@pytest.mark.parametrize("method,units", [("tools/call", 1), ("tools/list", 0), ("initialize", 0), ("ping", 0)])
def test_mcp_units_and_request_body_replay(mcp_client, network, method, units):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {"name": "get_quote"}}
    response = mcp_client.post("/mcp", json=payload, headers=AUTH)
    assert response.status_code == 200
    assert response.json() == payload
    assert network.await_args_list[-1].kwargs["body"]["p_units"] == units


def test_mcp_sse_connection_spends_no_quota(mcp_client, network):
    assert mcp_client.get("/sse", headers=AUTH).status_code == 200
    assert network.await_args_list[-1].kwargs["body"]["p_units"] == 0


def test_mcp_batch_cannot_execute_many_tools_for_one_unit(mcp_client, network):
    calls = [{"jsonrpc": "2.0", "method": "tools/call", "id": i} for i in range(10)]
    response = mcp_client.post("/mcp", json=calls, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["code"] == "mcp_batch_unsupported"
    assert network.await_count == 1  # identity verified; no quota mutation


def test_mcp_oversized_request_is_rejected(mcp_client, network):
    response = mcp_client.post("/mcp", content=b"x" * (1024 * 1024 + 1), headers=AUTH)
    assert response.status_code == 413
    assert network.await_count == 1
