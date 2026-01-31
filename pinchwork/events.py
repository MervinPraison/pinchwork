"""SSE event bus for real-time notifications."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pinchwork.events")


@dataclass
class Event:
    type: str
    task_id: str
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """In-memory pub/sub for agent-scoped events."""

    def __init__(self, max_queue_size: int = 100):
        self._subscribers: dict[str, list[asyncio.Queue[Event | None]]] = {}
        self._max_queue_size = max_queue_size

    def subscribe(self, agent_id: str) -> asyncio.Queue[Event | None]:
        q: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.setdefault(agent_id, []).append(q)
        return q

    def unsubscribe(self, agent_id: str, queue: asyncio.Queue[Event | None]) -> None:
        queues = self._subscribers.get(agent_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(agent_id, None)

    def publish(self, agent_id: str, event: Event) -> None:
        for q in self._subscribers.get(agent_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full for agent %s, dropping event", agent_id)

    def publish_many(self, agent_ids: list[str], event: Event) -> None:
        for aid in agent_ids:
            self.publish(aid, event)


event_bus = EventBus()
