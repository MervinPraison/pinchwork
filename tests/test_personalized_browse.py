"""Tests for personalized browsing, tag extraction, and enriched responses."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

from pinchwork.db_models import Agent, Task, TaskMatch
from tests.conftest import auth_header, register_agent


async def _register_infra_agent(client: AsyncClient, name: str = "infra") -> dict:
    """Register an agent that accepts system tasks."""
    data = await register_agent(client, name)
    resp = await client.patch(
        "/v1/me",
        json={"accepts_system_tasks": True, "good_at": "matching, verification"},
        headers=auth_header(data["api_key"]),
    )
    assert resp.status_code == 200
    return data


async def _register_skilled_agent(client: AsyncClient, name: str, good_at: str) -> dict:
    """Register an agent with specific skills."""
    data = await register_agent(client, name)
    resp = await client.patch(
        "/v1/me",
        json={"good_at": good_at},
        headers=auth_header(data["api_key"]),
    )
    assert resp.status_code == 200
    return data


async def _drain_non_match_tasks(client: AsyncClient, infra: dict) -> None:
    """Drain any system tasks that aren't match_agents (e.g. extract_capabilities)."""
    for _ in range(10):
        resp = await client.post(
            "/v1/tasks/pickup",
            headers=auth_header(infra["api_key"]),
        )
        if resp.status_code == 204:
            break
        data = resp.json()
        if "Match agents for:" in data["need"]:
            # Put it back by abandoning — actually we can't abandon system tasks
            # easily, so we just deliver a dummy result
            await client.post(
                f"/v1/tasks/{data['task_id']}/deliver",
                json={"result": json.dumps({"ranked_agents": []})},
                headers=auth_header(infra["api_key"]),
            )
            continue
        # Deliver non-match system tasks with dummy result
        await client.post(
            f"/v1/tasks/{data['task_id']}/deliver",
            json={
                "result": json.dumps(
                    {
                        "agent_id": "x",
                        "tags": [],
                    }
                )
            },
            headers=auth_header(infra["api_key"]),
        )


async def _do_matching(
    client: AsyncClient,
    infra: dict,
    ranked_agents: list[str],
    extracted_tags: list[str] | None = None,
) -> str:
    """Have infra agent pick up and deliver a match result. Returns system task ID."""
    # Drain any capability extraction tasks first
    for _ in range(10):
        resp = await client.post(
            "/v1/tasks/pickup",
            headers=auth_header(infra["api_key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        if "Match agents for:" in data["need"]:
            system_task_id = data["task_id"]
            break
        # Not a match task — deliver and continue
        await client.post(
            f"/v1/tasks/{data['task_id']}/deliver",
            json={
                "result": json.dumps(
                    {
                        "agent_id": "x",
                        "tags": [],
                    }
                )
            },
            headers=auth_header(infra["api_key"]),
        )
    else:
        raise AssertionError("No match_agents system task found")

    result = {"ranked_agents": ranked_agents}
    if extracted_tags is not None:
        result["extracted_tags"] = extracted_tags

    resp = await client.post(
        f"/v1/tasks/{system_task_id}/deliver",
        json={"result": json.dumps(result)},
        headers=auth_header(infra["api_key"]),
    )
    assert resp.status_code == 200
    return system_task_id


@pytest.mark.asyncio
async def test_extracted_tags_saved_from_matching(client, db):
    """Match result with extracted_tags populates Task.extracted_tags field."""
    poster = await register_agent(client, "poster")
    infra = await _register_infra_agent(client, "infra")
    worker = await _register_skilled_agent(client, "worker", "Dutch translation")

    resp = await client.post(
        "/v1/tasks",
        json={"need": "Translate document to Dutch", "max_credits": 10},
        headers=auth_header(poster["api_key"]),
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    await _do_matching(client, infra, [worker["agent_id"]], ["translation", "dutch", "nlp"])

    async with db() as session:
        task = await session.get(Task, task_id)
        assert task.extracted_tags is not None
        tags = json.loads(task.extracted_tags)
        assert "translation" in tags
        assert "dutch" in tags


@pytest.mark.asyncio
async def test_browse_matched_tasks_ordered_by_rank(client, db):
    """Matched tasks are returned in rank order, not insertion order."""
    poster = await register_agent(client, "poster")
    infra = await _register_infra_agent(client, "infra")
    alice = await _register_skilled_agent(client, "alice", "Dutch translation")

    # Create two tasks
    resp1 = await client.post(
        "/v1/tasks",
        json={"need": "Task A", "max_credits": 10},
        headers=auth_header(poster["api_key"]),
    )
    task_a = resp1.json()["task_id"]

    resp2 = await client.post(
        "/v1/tasks",
        json={"need": "Task B", "max_credits": 10},
        headers=auth_header(poster["api_key"]),
    )
    task_b = resp2.json()["task_id"]

    # Match both tasks — infra does matching for task A first, then task B
    # Rank alice for task A at rank 0
    await _do_matching(client, infra, [alice["agent_id"]])
    # Rank alice for task B at rank 0
    await _do_matching(client, infra, [alice["agent_id"]])

    # Now manually adjust ranks to make task B rank higher than task A
    async with db() as session:
        from sqlmodel import select

        result = await session.execute(
            select(TaskMatch).where(
                TaskMatch.agent_id == alice["agent_id"],
                TaskMatch.task_id == task_a,
            )
        )
        match_a = result.scalar_one()
        match_a.rank = 5  # Lower priority
        session.add(match_a)

        result = await session.execute(
            select(TaskMatch).where(
                TaskMatch.agent_id == alice["agent_id"],
                TaskMatch.task_id == task_b,
            )
        )
        match_b = result.scalar_one()
        match_b.rank = 0  # Higher priority
        session.add(match_b)
        await session.commit()

    # Browse — task B should come first (rank 0 vs rank 5)
    resp = await client.get(
        "/v1/tasks/available",
        headers=auth_header(alice["api_key"]),
    )
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    matched_tasks = [t for t in tasks if t["is_matched"]]
    assert len(matched_tasks) >= 2
    assert matched_tasks[0]["task_id"] == task_b
    assert matched_tasks[1]["task_id"] == task_a


@pytest.mark.asyncio
async def test_browse_broadcast_scored_by_tag_overlap(client, db):
    """Broadcast tasks are ordered by relevance to agent's capability tags."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")

    # Give the worker capability tags directly (simulating extraction result)
    async with db() as session:
        agent = await session.get(Agent, worker["agent_id"])
        agent.capability_tags = json.dumps(["dutch", "translation"])
        agent.good_at = "Dutch translation"
        session.add(agent)
        await session.commit()

    # Create tasks — one with matching tags, one without
    resp1 = await client.post(
        "/v1/tasks",
        json={"need": "Python coding task", "max_credits": 10, "tags": ["python", "coding"]},
        headers=auth_header(poster["api_key"]),
    )
    assert resp1.status_code == 201
    python_task_id = resp1.json()["task_id"]

    resp2 = await client.post(
        "/v1/tasks",
        json={
            "need": "Dutch translation task",
            "max_credits": 10,
            "tags": ["dutch", "translation"],
        },
        headers=auth_header(poster["api_key"]),
    )
    assert resp2.status_code == 201
    dutch_task_id = resp2.json()["task_id"]

    # Browse — Dutch task should come first because of tag overlap
    resp = await client.get(
        "/v1/tasks/available",
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert len(tasks) >= 2
    # Dutch task should be first (2 tag overlaps vs 0)
    assert tasks[0]["task_id"] == dutch_task_id
    assert tasks[1]["task_id"] == python_task_id


@pytest.mark.asyncio
async def test_capability_extraction_spawned_on_register(client, db):
    """System task created when agent registers with good_at and infra agents exist."""
    infra = await _register_infra_agent(client, "infra")

    # Register a new agent with good_at
    resp = await client.post(
        "/v1/register",
        json={"name": "skilled-agent", "good_at": "Python data analysis"},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 201

    # Infra agent should be able to pick up an extract_capabilities system task
    # (there may also be a match or other system task from infra's own registration)
    # Keep picking up until we find extract_capabilities or run out
    found = False
    for _ in range(5):
        resp = await client.post(
            "/v1/tasks/pickup",
            headers=auth_header(infra["api_key"]),
        )
        if resp.status_code == 204:
            break
        data = resp.json()
        if "Extract capability tags" in data["need"]:
            found = True
            # Deliver result
            agent_id = data["need"].split("Agent ID: ")[1].split("\n")[0]
            result = json.dumps(
                {
                    "agent_id": agent_id,
                    "tags": ["python", "data-analysis"],
                }
            )
            await client.post(
                f"/v1/tasks/{data['task_id']}/deliver",
                json={"result": result},
                headers=auth_header(infra["api_key"]),
            )
            break
        else:
            # Deliver dummy result for other system tasks
            await client.post(
                f"/v1/tasks/{data['task_id']}/deliver",
                json={"result": json.dumps({"ranked_agents": []})},
                headers=auth_header(infra["api_key"]),
            )

    assert found, "No extract_capabilities system task was spawned"


@pytest.mark.asyncio
async def test_capability_extraction_spawned_on_update(client, db):
    """System task created on PATCH /v1/me when good_at changes."""
    infra = await _register_infra_agent(client, "infra")
    worker = await register_agent(client, "worker")

    # Drain any existing system tasks from infra registration
    for _ in range(5):
        resp = await client.post(
            "/v1/tasks/pickup",
            headers=auth_header(infra["api_key"]),
        )
        if resp.status_code == 204:
            break
        data = resp.json()
        await client.post(
            f"/v1/tasks/{data['task_id']}/deliver",
            json={"result": json.dumps({"ranked_agents": [], "agent_id": "x", "tags": []})},
            headers=auth_header(infra["api_key"]),
        )

    # Update worker's good_at
    resp = await client.patch(
        "/v1/me",
        json={"good_at": "French translation, legal text"},
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200

    # Infra should get an extract_capabilities task
    resp = await client.post(
        "/v1/tasks/pickup",
        headers=auth_header(infra["api_key"]),
    )
    assert resp.status_code == 200
    assert "Extract capability tags" in resp.json()["need"]


@pytest.mark.asyncio
async def test_capability_result_processed(client, db):
    """Delivering extraction result saves Agent.capability_tags."""
    infra = await _register_infra_agent(client, "infra")
    worker = await register_agent(client, "worker")

    # Drain existing system tasks
    for _ in range(5):
        resp = await client.post(
            "/v1/tasks/pickup",
            headers=auth_header(infra["api_key"]),
        )
        if resp.status_code == 204:
            break
        data = resp.json()
        await client.post(
            f"/v1/tasks/{data['task_id']}/deliver",
            json={"result": json.dumps({"ranked_agents": [], "agent_id": "x", "tags": []})},
            headers=auth_header(infra["api_key"]),
        )

    # Update worker to trigger extraction
    resp = await client.patch(
        "/v1/me",
        json={"good_at": "Python, machine learning, data science"},
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200

    # Infra picks up extraction task
    resp = await client.post(
        "/v1/tasks/pickup",
        headers=auth_header(infra["api_key"]),
    )
    assert resp.status_code == 200
    system_task_id = resp.json()["task_id"]

    # Deliver capability extraction result
    result = json.dumps(
        {
            "agent_id": worker["agent_id"],
            "tags": ["python", "machine-learning", "data-science"],
        }
    )
    resp = await client.post(
        f"/v1/tasks/{system_task_id}/deliver",
        json={"result": result},
        headers=auth_header(infra["api_key"]),
    )
    assert resp.status_code == 200

    # Verify tags saved
    async with db() as session:
        agent = await session.get(Agent, worker["agent_id"])
        assert agent.capability_tags is not None
        tags = json.loads(agent.capability_tags)
        assert "python" in tags
        assert "machine-learning" in tags


@pytest.mark.asyncio
async def test_garbage_capability_result_handled(client, db):
    """Invalid JSON in capability extraction doesn't crash, tags stays None."""
    infra = await _register_infra_agent(client, "infra")
    worker = await register_agent(client, "worker")

    # Drain existing system tasks
    for _ in range(5):
        resp = await client.post(
            "/v1/tasks/pickup",
            headers=auth_header(infra["api_key"]),
        )
        if resp.status_code == 204:
            break
        data = resp.json()
        await client.post(
            f"/v1/tasks/{data['task_id']}/deliver",
            json={"result": json.dumps({"ranked_agents": [], "agent_id": "x", "tags": []})},
            headers=auth_header(infra["api_key"]),
        )

    # Trigger extraction
    resp = await client.patch(
        "/v1/me",
        json={"good_at": "testing garbage"},
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200

    # Infra picks up and delivers garbage
    resp = await client.post(
        "/v1/tasks/pickup",
        headers=auth_header(infra["api_key"]),
    )
    assert resp.status_code == 200
    system_task_id = resp.json()["task_id"]

    resp = await client.post(
        f"/v1/tasks/{system_task_id}/deliver",
        json={"result": "not valid json at all"},
        headers=auth_header(infra["api_key"]),
    )
    assert resp.status_code == 200  # Should not crash

    # Tags should remain None
    async with db() as session:
        agent = await session.get(Agent, worker["agent_id"])
        assert agent.capability_tags is None


@pytest.mark.asyncio
async def test_browse_without_capability_tags_is_fifo(client, db):
    """Agents without capability tags see broadcast tasks in FIFO order."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")  # No good_at, no capability_tags

    # Create tasks in order
    resp1 = await client.post(
        "/v1/tasks",
        json={"need": "First task", "max_credits": 10, "tags": ["python"]},
        headers=auth_header(poster["api_key"]),
    )
    first_id = resp1.json()["task_id"]

    resp2 = await client.post(
        "/v1/tasks",
        json={"need": "Second task", "max_credits": 10, "tags": ["dutch"]},
        headers=auth_header(poster["api_key"]),
    )
    second_id = resp2.json()["task_id"]

    # Browse — should be FIFO since worker has no capability tags
    resp = await client.get(
        "/v1/tasks/available",
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert len(tasks) == 2
    assert tasks[0]["task_id"] == first_id
    assert tasks[1]["task_id"] == second_id


@pytest.mark.asyncio
async def test_pickup_respects_rank_order(client, db):
    """Pickup claims the highest-ranked matched task first."""
    poster = await register_agent(client, "poster")
    infra = await _register_infra_agent(client, "infra")
    worker = await _register_skilled_agent(client, "worker", "Dutch translation")

    # Create two tasks
    resp1 = await client.post(
        "/v1/tasks",
        json={"need": "Low priority task", "max_credits": 10},
        headers=auth_header(poster["api_key"]),
    )
    task_low = resp1.json()["task_id"]

    resp2 = await client.post(
        "/v1/tasks",
        json={"need": "High priority task", "max_credits": 10},
        headers=auth_header(poster["api_key"]),
    )
    task_high = resp2.json()["task_id"]

    # Match both tasks — give worker rank 5 for low, rank 0 for high
    await _do_matching(client, infra, [worker["agent_id"]])
    await _do_matching(client, infra, [worker["agent_id"]])

    async with db() as session:
        from sqlmodel import select

        result = await session.execute(
            select(TaskMatch).where(
                TaskMatch.agent_id == worker["agent_id"],
                TaskMatch.task_id == task_low,
            )
        )
        match_low = result.scalar_one()
        match_low.rank = 5
        session.add(match_low)

        result = await session.execute(
            select(TaskMatch).where(
                TaskMatch.agent_id == worker["agent_id"],
                TaskMatch.task_id == task_high,
            )
        )
        match_high = result.scalar_one()
        match_high.rank = 0
        session.add(match_high)
        await session.commit()

    # Pickup — should get the high-priority task
    resp = await client.post(
        "/v1/tasks/pickup",
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200
    assert resp.json()["task_id"] == task_high


@pytest.mark.asyncio
async def test_enriched_browse_response(client, db):
    """Browse response includes poster_reputation, is_matched, and match_rank."""
    poster = await register_agent(client, "poster")
    infra = await _register_infra_agent(client, "infra")
    worker = await _register_skilled_agent(client, "worker", "Dutch translation")

    resp = await client.post(
        "/v1/tasks",
        json={"need": "Test enriched browse", "max_credits": 10},
        headers=auth_header(poster["api_key"]),
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    await _do_matching(client, infra, [worker["agent_id"]], ["test"])

    resp = await client.get(
        "/v1/tasks/available",
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert len(tasks) >= 1

    task = next(t for t in tasks if t["task_id"] == task_id)
    assert "poster_reputation" in task
    assert task["is_matched"] is True
    assert task["match_rank"] == 0


@pytest.mark.asyncio
async def test_enriched_pickup_response(client, db):
    """Pickup response includes tags, created_at, and poster_reputation."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")

    resp = await client.post(
        "/v1/tasks",
        json={"need": "Test enriched pickup", "max_credits": 10, "tags": ["test-tag"]},
        headers=auth_header(poster["api_key"]),
    )
    assert resp.status_code == 201

    resp = await client.post(
        "/v1/tasks/pickup",
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "tags" in data
    assert data["tags"] == ["test-tag"]
    assert "created_at" in data
    assert data["created_at"] is not None
    assert "poster_reputation" in data


@pytest.mark.asyncio
async def test_no_capability_extraction_without_infra(client, db):
    """No system task spawned when no infra agents exist."""
    # Register agent with good_at but no infra agents
    resp = await client.post(
        "/v1/register",
        json={"name": "lonely-agent", "good_at": "Python coding"},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 201

    # No system tasks should exist (check DB directly)
    async with db() as session:
        from sqlmodel import select

        result = await session.execute(
            select(Task).where(
                Task.is_system == True,  # noqa: E712
                Task.system_task_type == "extract_capabilities",
            )
        )
        tasks = result.scalars().all()
        assert len(tasks) == 0
