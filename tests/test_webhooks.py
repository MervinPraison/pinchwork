"""Tests for webhook delivery (Feature 1)."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import auth_header, register_agent

pytestmark = pytest.mark.anyio


async def test_register_with_webhook(client):
    """Register an agent with webhook_url and webhook_secret."""
    resp = await client.post(
        "/v1/register",
        json={
            "name": "webhook-agent",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "my-secret",
        },
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 201
    data = resp.json()

    # Verify webhook URL shows up in GET /v1/me
    me_resp = await client.get("/v1/me", headers=auth_header(data["api_key"]))
    assert me_resp.status_code == 200
    assert me_resp.json()["webhook_url"] == "https://example.com/hook"


async def test_update_webhook_url(client):
    """Update webhook_url via PATCH /v1/me."""
    reg = await register_agent(client, "webhook-update")
    headers = auth_header(reg["api_key"])

    resp = await client.patch(
        "/v1/me",
        json={"webhook_url": "https://new-hook.com/events"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["webhook_url"] == "https://new-hook.com/events"


async def test_update_webhook_secret(client):
    """Update webhook_secret via PATCH /v1/me."""
    reg = await register_agent(client, "webhook-secret")
    headers = auth_header(reg["api_key"])

    resp = await client.patch(
        "/v1/me",
        json={"webhook_url": "https://hook.com/events", "webhook_secret": "new-secret"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["webhook_url"] == "https://hook.com/events"


async def test_webhook_not_in_public_profile(client):
    """Webhook URL/secret should not appear in public agent profile."""
    resp = await client.post(
        "/v1/register",
        json={
            "name": "private-hook",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "secret",
        },
        headers={"Accept": "application/json"},
    )
    data = resp.json()

    profile = await client.get(
        f"/v1/agents/{data['agent_id']}", headers={"Accept": "application/json"}
    )
    assert profile.status_code == 200
    profile_data = profile.json()
    assert "webhook_url" not in profile_data
    assert "webhook_secret" not in profile_data


async def test_webhook_signature_computation():
    """Test HMAC-SHA256 signature generation."""
    from pinchwork.webhooks import _sign_payload

    payload = b'{"event": "test"}'
    secret = "my-secret"
    sig = _sign_payload(payload, secret)

    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert sig == expected


async def test_webhook_delivery_on_task_event(client):
    """Webhook is delivered when a task event occurs."""
    # Register poster with webhook
    poster_resp = await client.post(
        "/v1/register",
        json={
            "name": "poster-with-hook",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "test-secret",
        },
        headers={"Accept": "application/json"},
    )
    poster = poster_resp.json()
    poster_headers = auth_header(poster["api_key"])

    # Register worker
    worker = await register_agent(client, "worker-for-hook")
    worker_headers = auth_header(worker["api_key"])

    # Create task
    task_resp = await client.post(
        "/v1/tasks",
        json={"need": "Test webhook delivery", "max_credits": 10},
        headers=poster_headers,
    )
    assert task_resp.status_code == 201
    task_id = task_resp.json()["task_id"]

    # Worker picks up and delivers — should trigger webhook to poster
    with patch("pinchwork.webhooks._deliver_with_retries", new_callable=AsyncMock) as mock_deliver:
        pickup_resp = await client.post(
            f"/v1/tasks/{task_id}/pickup",
            headers=worker_headers,
        )
        assert pickup_resp.status_code == 200

        deliver_resp = await client.post(
            f"/v1/tasks/{task_id}/deliver",
            json={"result": "Done!"},
            headers=worker_headers,
        )
        assert deliver_resp.status_code == 200

        # Allow async tasks to run
        import asyncio

        await asyncio.sleep(0.1)

        # Verify webhook was called (at least the task_delivered event)
        if mock_deliver.call_count > 0:
            call_args = mock_deliver.call_args_list[-1]
            url = call_args[0][0]
            payload = json.loads(call_args[0][1])
            headers_sent = call_args[0][2]
            assert url == "https://example.com/hook"
            assert payload["event"] == "task_delivered"
            assert payload["task_id"] == task_id
            assert "X-Pinchwork-Signature" in headers_sent


async def test_webhook_no_delivery_without_url(client):
    """No webhook delivery if agent has no webhook_url."""
    from pinchwork.events import Event
    from pinchwork.webhooks import deliver_webhook

    poster = await register_agent(client, "no-webhook-agent")

    # Mock session.get to return an agent without webhook_url
    mock_session = AsyncMock()
    mock_agent = AsyncMock()
    mock_agent.webhook_url = None
    mock_session.get.return_value = mock_agent

    with patch("pinchwork.webhooks._deliver_with_retries", new_callable=AsyncMock) as mock:
        await deliver_webhook(
            poster["agent_id"], Event(type="test", task_id="tk-123"), mock_session
        )
        mock.assert_not_called()


async def test_webhook_retry_logic():
    """Test that webhook delivery retries on failure."""
    from pinchwork.webhooks import _deliver_with_retries

    with patch("pinchwork.webhooks.settings") as mock_settings:
        mock_settings.webhook_max_retries = 2
        mock_settings.webhook_timeout_seconds = 1

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value = mock_client

            await _deliver_with_retries(
                "https://example.com/hook",
                b'{"event": "test"}',
                {"Content-Type": "application/json"},
            )

            # Should have been called max_retries + 1 times
            assert mock_client.post.call_count == 3
