"""Webhook delivery with HMAC-SHA256 signatures."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from pinchwork.config import settings
from pinchwork.db_models import Agent
from pinchwork.events import Event

logger = logging.getLogger("pinchwork.webhooks")


def _sign_payload(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for a payload."""
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


async def _deliver_with_retries(url: str, payload: bytes, headers: dict) -> None:
    """POST payload to URL with exponential backoff retries."""
    max_retries = settings.webhook_max_retries
    timeout = settings.webhook_timeout_seconds

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, content=payload, headers=headers)
                if resp.status_code < 400:
                    logger.info("Webhook delivered to %s (status %d)", url, resp.status_code)
                    return
                logger.warning(
                    "Webhook to %s returned %d (attempt %d/%d)",
                    url,
                    resp.status_code,
                    attempt + 1,
                    max_retries + 1,
                )
        except Exception:
            logger.warning(
                "Webhook to %s failed (attempt %d/%d)",
                url,
                attempt + 1,
                max_retries + 1,
                exc_info=True,
            )

        if attempt < max_retries:
            await asyncio.sleep(2**attempt)

    logger.error("Webhook delivery to %s failed after %d attempts", url, max_retries + 1)


async def deliver_webhook(agent_id: str, event: Event, session: AsyncSession) -> None:
    """Look up agent's webhook config and deliver event."""
    agent = await session.get(Agent, agent_id)
    if not agent or not agent.webhook_url:
        return

    payload_dict = {
        "event": event.type,
        "task_id": event.task_id,
        "data": event.data,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    payload = json.dumps(payload_dict).encode()

    headers = {"Content-Type": "application/json"}
    if agent.webhook_secret:
        headers["X-Pinchwork-Signature"] = _sign_payload(payload, agent.webhook_secret)

    asyncio.create_task(_deliver_with_retries(agent.webhook_url, payload, headers))
