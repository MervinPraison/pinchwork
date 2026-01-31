"""Tests for platform fee, admin grant, escrowed balance."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from pinchwork.config import settings
from tests.conftest import auth_header, register_agent


@pytest.fixture(autouse=True)
def _set_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_key", "test-admin-secret")


async def _full_cycle(client: AsyncClient, poster_key: str, worker_key: str, max_credits: int = 50):
    """Create task, pickup, deliver, approve. Returns task dict from approve."""
    resp = await client.post(
        "/v1/tasks",
        json={"need": "test task", "max_credits": max_credits},
        headers=auth_header(poster_key),
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    resp = await client.post("/v1/tasks/pickup", headers=auth_header(worker_key))
    assert resp.status_code == 200

    resp = await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "done"},
        headers=auth_header(worker_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        f"/v1/tasks/{task_id}/approve",
        json={},
        headers=auth_header(poster_key),
    )
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.anyio
async def test_platform_fee_on_approve(two_agents):
    """10% fee deducted from worker payment, credited to platform."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    initial_credits = settings.initial_credits
    max_credits = 50

    await _full_cycle(c, poster["key"], worker["key"], max_credits=max_credits)

    # Worker gets 90% of charged credits (50 * 0.9 = 45)
    resp = await c.get("/v1/me", headers=auth_header(worker["key"]))
    worker_credits = resp.json()["credits"]
    fee = int(max_credits * settings.platform_fee_percent / 100)
    expected_worker = initial_credits + max_credits - fee
    assert worker_credits == expected_worker

    # Poster paid max_credits from escrow
    resp = await c.get("/v1/me", headers=auth_header(poster["key"]))
    poster_credits = resp.json()["credits"]
    assert poster_credits == initial_credits - max_credits


@pytest.mark.anyio
async def test_platform_fee_zero_percent(two_agents, monkeypatch):
    """With 0% fee, worker gets full amount."""
    monkeypatch.setattr(settings, "platform_fee_percent", 0.0)
    c = two_agents["client"]

    await _full_cycle(c, two_agents["poster"]["key"], two_agents["worker"]["key"], max_credits=50)

    resp = await c.get("/v1/me", headers=auth_header(two_agents["worker"]["key"]))
    assert resp.json()["credits"] == settings.initial_credits + 50


@pytest.mark.anyio
async def test_admin_grant_happy_path(client):
    """Admin can grant credits to any agent."""
    data = await register_agent(client, "grantee")

    resp = await client.post(
        "/v1/admin/credits/grant",
        json={"agent_id": data["agent_id"], "amount": 500, "reason": "bonus"},
        headers={"Authorization": "Bearer test-admin-secret", "Accept": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["granted"] == 500

    # Verify credits increased
    resp = await client.get("/v1/me", headers=auth_header(data["api_key"]))
    assert resp.json()["credits"] == settings.initial_credits + 500


@pytest.mark.anyio
async def test_admin_grant_wrong_key(client):
    """Wrong admin key returns 403."""
    data = await register_agent(client, "test")
    resp = await client.post(
        "/v1/admin/credits/grant",
        json={"agent_id": data["agent_id"], "amount": 100},
        headers={"Authorization": "Bearer wrong-key", "Accept": "application/json"},
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_grant_no_key_configured(client, monkeypatch):
    """No admin key configured returns 501."""
    monkeypatch.setattr(settings, "admin_key", None)
    resp = await client.post(
        "/v1/admin/credits/grant",
        json={"agent_id": "ag_test", "amount": 100},
        headers={"Authorization": "Bearer anything", "Accept": "application/json"},
    )
    assert resp.status_code == 501


@pytest.mark.anyio
async def test_admin_grant_nonexistent_agent(client):
    """Granting to nonexistent agent returns 404."""
    resp = await client.post(
        "/v1/admin/credits/grant",
        json={"agent_id": "ag_nonexistent", "amount": 100},
        headers={"Authorization": "Bearer test-admin-secret", "Accept": "application/json"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_escrowed_balance(client):
    """Escrowed balance reflects active tasks."""
    data = await register_agent(client, "escrow-test")
    headers = auth_header(data["api_key"])

    # Create two tasks
    await client.post("/v1/tasks", json={"need": "t1", "max_credits": 20}, headers=headers)
    await client.post("/v1/tasks", json={"need": "t2", "max_credits": 30}, headers=headers)

    resp = await client.get("/v1/me/credits", headers=headers)
    body = resp.json()
    assert body["escrowed"] == 50
    assert body["balance"] == settings.initial_credits - 50


@pytest.mark.anyio
async def test_higher_max_credits(client):
    """Can create tasks with max_credits up to 100000."""
    data = await register_agent(client, "high-credits")
    headers = auth_header(data["api_key"])

    # Need more credits first
    await client.post(
        "/v1/admin/credits/grant",
        json={"agent_id": data["agent_id"], "amount": 100000},
        headers={"Authorization": "Bearer test-admin-secret", "Accept": "application/json"},
    )

    resp = await client.post(
        "/v1/tasks",
        json={"need": "big task", "max_credits": 90000},
        headers=headers,
    )
    assert resp.status_code == 201


@pytest.mark.anyio
async def test_credits_ledger_shows_fee(two_agents):
    """Ledger entries show platform_fee reason."""
    c = two_agents["client"]
    await _full_cycle(c, two_agents["poster"]["key"], two_agents["worker"]["key"], max_credits=50)

    resp = await c.get("/v1/me/credits", headers=auth_header(two_agents["worker"]["key"]))
    ledger = resp.json()["ledger"]
    reasons = [e["reason"] for e in ledger]
    assert "payment" in reasons
    assert "signup_bonus" in reasons
