"""Tests for the rating system."""

from __future__ import annotations

import pytest

from pinchwork.config import settings
from tests.conftest import auth_header, register_agent


@pytest.fixture(autouse=True)
def _set_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_key", "test-admin-secret")


async def _create_deliver_approve(client, poster_key, worker_key, max_credits=50, rating=None):
    """Helper: create task, pickup, deliver, approve. Returns task_id."""
    resp = await client.post(
        "/v1/tasks",
        json={"need": "rate me", "max_credits": max_credits},
        headers=auth_header(poster_key),
    )
    task_id = resp.json()["task_id"]

    await client.post("/v1/tasks/pickup", headers=auth_header(worker_key))
    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "done"},
        headers=auth_header(worker_key),
    )

    approve_body = {}
    if rating is not None:
        approve_body["rating"] = rating
    resp = await client.post(
        f"/v1/tasks/{task_id}/approve",
        json=approve_body,
        headers=auth_header(poster_key),
    )
    assert resp.status_code == 200
    return task_id


@pytest.mark.anyio
async def test_approve_with_rating_updates_reputation(two_agents):
    """Approving with a rating updates the worker's reputation."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    await _create_deliver_approve(c, poster["key"], worker["key"], rating=5)

    # Check worker reputation
    resp = await c.get(f"/v1/agents/{worker['id']}", headers=auth_header(poster["key"]))
    body = resp.json()
    assert body["reputation"] == 5.0
    assert body["rating_count"] == 1


@pytest.mark.anyio
async def test_approve_without_rating_backwards_compat(two_agents):
    """Approving without rating still works."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    await _create_deliver_approve(c, poster["key"], worker["key"], rating=None)

    resp = await c.get(f"/v1/agents/{worker['id']}", headers=auth_header(poster["key"]))
    body = resp.json()
    assert body["reputation"] == 0.0
    assert body["rating_count"] == 0


@pytest.mark.anyio
async def test_worker_rates_poster(two_agents):
    """Worker can rate the poster after task approval."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    task_id = await _create_deliver_approve(c, poster["key"], worker["key"])

    resp = await c.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 4, "feedback": "good task description"},
        headers=auth_header(worker["key"]),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["rated_id"] == poster["id"]
    assert body["rating"] == 4


@pytest.mark.anyio
async def test_double_rating_prevented(two_agents):
    """Can't rate the same task twice."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    task_id = await _create_deliver_approve(c, poster["key"], worker["key"])

    await c.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 4},
        headers=auth_header(worker["key"]),
    )

    resp = await c.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 5},
        headers=auth_header(worker["key"]),
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_non_worker_cannot_rate(two_agents):
    """Only the worker can rate the poster."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    task_id = await _create_deliver_approve(c, poster["key"], worker["key"])

    # Poster tries to rate (poster is not the worker)
    d3 = await register_agent(c, "outsider")
    resp = await c.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 1},
        headers=auth_header(d3["api_key"]),
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_rate_unapproved_task(two_agents):
    """Can't rate a task that's not approved."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Create and deliver but don't approve
    resp = await c.post(
        "/v1/tasks",
        json={"need": "no approve", "max_credits": 10},
        headers=auth_header(poster["key"]),
    )
    task_id = resp.json()["task_id"]

    await c.post("/v1/tasks/pickup", headers=auth_header(worker["key"]))
    await c.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "done"},
        headers=auth_header(worker["key"]),
    )

    resp = await c.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 3},
        headers=auth_header(worker["key"]),
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_reputation_is_average(client):
    """Reputation equals the average of all ratings."""
    poster_data = await register_agent(client, "multi-poster")

    # Give poster enough credits
    await client.post(
        "/v1/admin/credits/grant",
        json={"agent_id": poster_data["agent_id"], "amount": 1000},
        headers={"Authorization": "Bearer test-admin-secret", "Accept": "application/json"},
    )

    # Two different workers rate the poster
    for worker_name, rating_score in [("w1", 3), ("w2", 5)]:
        worker_data = await register_agent(client, worker_name)
        task_id = await _create_deliver_approve(
            client, poster_data["api_key"], worker_data["api_key"], max_credits=10, rating=None
        )
        await client.post(
            f"/v1/tasks/{task_id}/rate",
            json={"rating": rating_score},
            headers=auth_header(worker_data["api_key"]),
        )

    # Poster reputation should be average: (3 + 5) / 2 = 4.0
    resp = await client.get(
        f"/v1/agents/{poster_data['agent_id']}",
        headers=auth_header(poster_data["api_key"]),
    )
    assert resp.json()["reputation"] == 4.0
    assert resp.json()["rating_count"] == 2


@pytest.mark.anyio
async def test_rating_count_in_public_profile(two_agents):
    """Public profile shows rating_count."""
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    await _create_deliver_approve(c, poster["key"], worker["key"], rating=4)

    resp = await c.get(f"/v1/agents/{worker['id']}", headers=auth_header(poster["key"]))
    assert resp.json()["rating_count"] == 1
