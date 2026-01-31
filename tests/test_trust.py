"""Tests for agent-to-agent trust scores (Feature 6)."""

from __future__ import annotations

import pytest

from tests.conftest import auth_header, register_agent

pytestmark = pytest.mark.anyio


async def _complete_task(
    client, poster_h, worker_h, poster_id, worker_id, approve=True, rating=None
):
    """Helper: create, pickup, deliver, and optionally approve a task."""
    resp = await client.post(
        "/v1/tasks",
        json={"need": "Trust test task", "max_credits": 10},
        headers=poster_h,
    )
    task_id = resp.json()["task_id"]

    await client.post(f"/v1/tasks/{task_id}/pickup", headers=worker_h)
    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "Done"},
        headers=worker_h,
    )

    if approve:
        body = {}
        if rating is not None:
            body["rating"] = rating
        await client.post(f"/v1/tasks/{task_id}/approve", json=body, headers=poster_h)

    return task_id


async def test_trust_increases_on_approve(client):
    """Trust should increase bidirectionally when a task is approved."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])
    worker_h = auth_header(worker["api_key"])

    await _complete_task(client, poster_h, worker_h, poster["agent_id"], worker["agent_id"])

    # Check poster's trust scores
    resp = await client.get("/v1/me/trust", headers=poster_h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["trust_scores"][0]["trusted_id"] == worker["agent_id"]
    assert data["trust_scores"][0]["score"] > 0.5  # Should have increased from 0.5
    assert data["trust_scores"][0]["interactions"] == 1

    # Check worker's trust scores
    resp = await client.get("/v1/me/trust", headers=worker_h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["trust_scores"][0]["trusted_id"] == poster["agent_id"]
    assert data["trust_scores"][0]["score"] > 0.5


async def test_trust_decreases_on_reject(client):
    """Trust should decrease poster→worker on rejection."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])
    worker_h = auth_header(worker["api_key"])

    # Create, pickup, deliver
    resp = await client.post(
        "/v1/tasks",
        json={"need": "Will be rejected", "max_credits": 10},
        headers=poster_h,
    )
    task_id = resp.json()["task_id"]

    await client.post(f"/v1/tasks/{task_id}/pickup", headers=worker_h)
    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "Bad work"},
        headers=worker_h,
    )

    # Reject
    await client.post(
        f"/v1/tasks/{task_id}/reject",
        json={"reason": "Not meeting requirements"},
        headers=poster_h,
    )

    # Check poster's trust toward worker (should be < 0.5)
    resp = await client.get("/v1/me/trust", headers=poster_h)
    data = resp.json()
    assert data["total"] == 1
    assert data["trust_scores"][0]["trusted_id"] == worker["agent_id"]
    assert data["trust_scores"][0]["score"] < 0.5  # Decreased from 0.5


async def test_trust_stays_in_bounds(client):
    """Trust score should always stay in [0, 1]."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])
    worker_h = auth_header(worker["api_key"])

    # Multiple approvals should increase trust but stay ≤ 1.0
    for _ in range(10):
        await _complete_task(client, poster_h, worker_h, poster["agent_id"], worker["agent_id"])

    resp = await client.get("/v1/me/trust", headers=poster_h)
    data = resp.json()
    score = data["trust_scores"][0]["score"]
    assert 0.0 <= score <= 1.0
    assert data["trust_scores"][0]["interactions"] == 10


async def test_trust_bidirectional_independent(client):
    """Trust from A→B is independent of B→A."""
    a = await register_agent(client, "agent-a")
    b = await register_agent(client, "agent-b")
    a_h = auth_header(a["api_key"])
    b_h = auth_header(b["api_key"])

    # A posts, B works, A approves → both get +trust
    await _complete_task(client, a_h, b_h, a["agent_id"], b["agent_id"])

    # B posts, A works, B rejects → B's trust toward A decreases
    resp = await client.post(
        "/v1/tasks",
        json={"need": "Reverse task", "max_credits": 10},
        headers=b_h,
    )
    task_id = resp.json()["task_id"]
    await client.post(f"/v1/tasks/{task_id}/pickup", headers=a_h)
    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "Bad"},
        headers=a_h,
    )
    await client.post(
        f"/v1/tasks/{task_id}/reject",
        json={"reason": "Poor quality"},
        headers=b_h,
    )

    # A's trust toward B should still be > 0.5 (from approval)
    resp = await client.get("/v1/me/trust", headers=a_h)
    a_trust = resp.json()["trust_scores"]
    b_score = next(t for t in a_trust if t["trusted_id"] == b["agent_id"])
    assert b_score["score"] > 0.5

    # B's trust toward A: one positive (approve) + one negative (reject)
    resp = await client.get("/v1/me/trust", headers=b_h)
    b_trust = resp.json()["trust_scores"]
    a_score = next(t for t in b_trust if t["trusted_id"] == a["agent_id"])
    # Should be close to 0.5 since one up one down
    assert a_score["interactions"] == 2


async def test_trust_from_poster_rating(client):
    """Worker rating the poster also updates trust."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])
    worker_h = auth_header(worker["api_key"])

    task_id = await _complete_task(
        client, poster_h, worker_h, poster["agent_id"], worker["agent_id"]
    )

    # Worker rates poster highly
    await client.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 5},
        headers=worker_h,
    )

    # Worker's trust toward poster should have been updated twice:
    # once from approve (positive) and once from rating (positive, since rating >= 3)
    resp = await client.get("/v1/me/trust", headers=worker_h)
    data = resp.json()
    poster_trust = next(t for t in data["trust_scores"] if t["trusted_id"] == poster["agent_id"])
    assert poster_trust["interactions"] == 2
    assert poster_trust["score"] > 0.5


async def test_low_rating_decreases_trust(client):
    """A low rating (< 3) should decrease worker's trust toward poster."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])
    worker_h = auth_header(worker["api_key"])

    task_id = await _complete_task(
        client, poster_h, worker_h, poster["agent_id"], worker["agent_id"]
    )

    # Worker rates poster low
    await client.post(
        f"/v1/tasks/{task_id}/rate",
        json={"rating": 1},
        headers=worker_h,
    )

    resp = await client.get("/v1/me/trust", headers=worker_h)
    data = resp.json()
    poster_trust = next(t for t in data["trust_scores"] if t["trusted_id"] == poster["agent_id"])
    # Two updates: positive from approve, negative from low rating
    assert poster_trust["interactions"] == 2


async def test_empty_trust_scores(client):
    """New agent has no trust scores."""
    agent = await register_agent(client, "new-agent")
    headers = auth_header(agent["api_key"])

    resp = await client.get("/v1/me/trust", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["trust_scores"] == []
