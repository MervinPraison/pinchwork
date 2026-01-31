"""Tests for mid-task messaging (Feature 5)."""

from __future__ import annotations

import pytest

from tests.conftest import auth_header, register_agent

pytestmark = pytest.mark.anyio


async def _create_claimed_task(client):
    """Helper: create a task and have it claimed by a worker."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])
    worker_h = auth_header(worker["api_key"])

    # Create task
    resp = await client.post(
        "/v1/tasks",
        json={"need": "Test messaging", "max_credits": 10},
        headers=poster_h,
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    # Worker picks up
    pickup = await client.post(f"/v1/tasks/{task_id}/pickup", headers=worker_h)
    assert pickup.status_code == 200

    return task_id, poster, worker, poster_h, worker_h


async def test_send_message_as_worker(client):
    """Worker can send a message on a claimed task."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Working on this now!"},
        headers=worker_h,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["sender_id"] == worker["agent_id"]
    assert data["message"] == "Working on this now!"
    assert data["id"].startswith("msg-")


async def test_send_message_as_poster(client):
    """Poster can send a message on a claimed task."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Please prioritize section 2"},
        headers=poster_h,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["sender_id"] == poster["agent_id"]


async def test_list_messages_in_order(client):
    """Messages are listed in chronological order."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    # Send multiple messages
    await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "First message"},
        headers=worker_h,
    )
    await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Second message"},
        headers=poster_h,
    )
    await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Third message"},
        headers=worker_h,
    )

    # List messages
    resp = await client.get(f"/v1/tasks/{task_id}/messages", headers=worker_h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert [m["message"] for m in data["messages"]] == [
        "First message",
        "Second message",
        "Third message",
    ]


async def test_outsider_cannot_send_message(client):
    """A third agent cannot send messages on a task."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)
    outsider = await register_agent(client, "outsider")
    outsider_h = auth_header(outsider["api_key"])

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Sneaky message"},
        headers=outsider_h,
    )
    assert resp.status_code == 403


async def test_outsider_cannot_list_messages(client):
    """A third agent cannot list messages on a task."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)
    outsider = await register_agent(client, "outsider")
    outsider_h = auth_header(outsider["api_key"])

    resp = await client.get(f"/v1/tasks/{task_id}/messages", headers=outsider_h)
    assert resp.status_code == 403


async def test_message_on_posted_task_rejected(client):
    """Messages are not allowed on posted (unclaimed) tasks."""
    poster = await register_agent(client, "poster")
    await register_agent(client, "worker")
    poster_h = auth_header(poster["api_key"])

    resp = await client.post(
        "/v1/tasks",
        json={"need": "Not yet claimed", "max_credits": 10},
        headers=poster_h,
    )
    task_id = resp.json()["task_id"]

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Too early"},
        headers=poster_h,
    )
    assert resp.status_code == 409


async def test_message_on_delivered_task(client):
    """Messages are allowed on delivered tasks."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    # Deliver
    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "Done"},
        headers=worker_h,
    )

    # Both can still message
    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Looks good, some feedback"},
        headers=poster_h,
    )
    assert resp.status_code == 201

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Thanks for the feedback"},
        headers=worker_h,
    )
    assert resp.status_code == 201


async def test_message_on_approved_task_rejected(client):
    """Messages are not allowed on approved tasks."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "Done"},
        headers=worker_h,
    )
    await client.post(f"/v1/tasks/{task_id}/approve", json={}, headers=poster_h)

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "Too late"},
        headers=poster_h,
    )
    assert resp.status_code == 409


async def test_messages_readable_on_approved_task(client):
    """Messages can be read even after task is approved."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    # Send a message while claimed
    await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={"message": "During work"},
        headers=worker_h,
    )

    # Deliver and approve
    await client.post(
        f"/v1/tasks/{task_id}/deliver",
        json={"result": "Done"},
        headers=worker_h,
    )
    await client.post(f"/v1/tasks/{task_id}/approve", json={}, headers=poster_h)

    # Messages are still readable
    resp = await client.get(f"/v1/tasks/{task_id}/messages", headers=poster_h)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


async def test_message_missing_field(client):
    """Missing message field returns 400."""
    task_id, poster, worker, poster_h, worker_h = await _create_claimed_task(client)

    resp = await client.post(
        f"/v1/tasks/{task_id}/messages",
        json={},
        headers=worker_h,
    )
    assert resp.status_code == 400


async def test_message_on_nonexistent_task(client):
    """Messaging a non-existent task returns 404."""
    agent = await register_agent(client, "agent")
    headers = auth_header(agent["api_key"])

    resp = await client.post(
        "/v1/tasks/tk-nonexistent/messages",
        json={"message": "Hello"},
        headers=headers,
    )
    assert resp.status_code == 404
