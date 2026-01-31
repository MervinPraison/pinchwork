"""Tests for the SSE event bus and event publishing."""

from __future__ import annotations

import pytest

from pinchwork.events import Event, EventBus


@pytest.mark.anyio
async def test_event_bus_subscribe_publish():
    """EventBus delivers events to subscribers."""
    bus = EventBus()
    q = bus.subscribe("agent1")

    bus.publish("agent1", Event(type="test", task_id="tk_1"))

    event = q.get_nowait()
    assert event.type == "test"
    assert event.task_id == "tk_1"


@pytest.mark.anyio
async def test_event_bus_unsubscribe():
    """After unsubscribe, no more events are received."""
    bus = EventBus()
    q = bus.subscribe("agent1")
    bus.unsubscribe("agent1", q)

    bus.publish("agent1", Event(type="test", task_id="tk_1"))

    assert q.empty()


@pytest.mark.anyio
async def test_event_bus_publish_many():
    """publish_many sends to all specified agents."""
    bus = EventBus()
    q1 = bus.subscribe("a1")
    q2 = bus.subscribe("a2")
    q3 = bus.subscribe("a3")

    bus.publish_many(["a1", "a3"], Event(type="notify", task_id="tk_x"))

    assert not q1.empty()
    assert q2.empty()
    assert not q3.empty()


@pytest.mark.anyio
async def test_event_bus_queue_full():
    """Queue full drops events gracefully."""
    bus = EventBus(max_queue_size=2)
    q = bus.subscribe("agent1")

    # Fill the queue
    bus.publish("agent1", Event(type="e1", task_id="tk_1"))
    bus.publish("agent1", Event(type="e2", task_id="tk_2"))
    # This should be silently dropped
    bus.publish("agent1", Event(type="e3", task_id="tk_3"))

    assert q.qsize() == 2


@pytest.mark.anyio
async def test_event_bus_multiple_subscribers():
    """Multiple subscribers for same agent each get the event."""
    bus = EventBus()
    q1 = bus.subscribe("agent1")
    q2 = bus.subscribe("agent1")

    bus.publish("agent1", Event(type="test", task_id="tk_1"))

    assert not q1.empty()
    assert not q2.empty()


@pytest.mark.anyio
async def test_event_bus_no_cross_agent():
    """Events for one agent don't reach another."""
    bus = EventBus()
    q1 = bus.subscribe("agent1")
    q2 = bus.subscribe("agent2")

    bus.publish("agent1", Event(type="test", task_id="tk_1"))

    assert not q1.empty()
    assert q2.empty()


@pytest.mark.anyio
async def test_sse_endpoint_requires_auth(client):
    """SSE endpoint requires authentication."""
    resp = await client.get("/v1/events")
    assert resp.status_code == 401
