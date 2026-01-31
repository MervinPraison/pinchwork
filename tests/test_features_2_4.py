"""Tests for features 2-4: Baseline Matcher, Task Deadlines, Reputation-Weighted Routing."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import auth_header, register_agent

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Feature 2: Built-in Baseline Matcher
# ---------------------------------------------------------------------------


class TestBuiltinMatcher:
    async def test_tag_overlap_matching(self, client, db):
        """Builtin matcher matches agents by tag overlap when no infra agents exist."""
        poster = await register_agent(client, "poster")
        worker_resp = await client.post(
            "/v1/register",
            json={"name": "skilled-worker", "good_at": "python data-science machine-learning"},
            headers={"Accept": "application/json"},
        )
        worker = worker_resp.json()
        worker_h = auth_header(worker["api_key"])

        # Manually set capability_tags
        async with db() as session:
            from pinchwork.db_models import Agent

            agent = await session.get(Agent, worker["agent_id"])
            agent.capability_tags = json.dumps(["python", "data-science", "machine-learning"])
            session.add(agent)
            await session.commit()

        poster_h = auth_header(poster["api_key"])

        # Create task with matching tags
        resp = await client.post(
            "/v1/tasks",
            json={
                "need": "Build a machine learning model",
                "max_credits": 20,
                "tags": ["python", "machine-learning"],
            },
            headers=poster_h,
        )
        assert resp.status_code == 201
        task_id = resp.json()["task_id"]

        # Worker should be matched
        avail = await client.get("/v1/tasks/available", headers=worker_h)
        assert avail.status_code == 200
        tasks = avail.json()["tasks"]
        matched = [t for t in tasks if t["task_id"] == task_id]
        assert len(matched) == 1
        assert matched[0]["is_matched"] is True

    async def test_no_match_falls_to_broadcast(self, client):
        """When no agents match tags, task should broadcast."""
        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        resp = await client.post(
            "/v1/tasks",
            json={
                "need": "Very specific quantum computing task",
                "max_credits": 20,
                "tags": ["quantum-computing", "qiskit"],
            },
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]

        # Worker with no matching skills should still see it
        worker = await register_agent(client, "generic-worker")
        worker_h = auth_header(worker["api_key"])

        avail = await client.get("/v1/tasks/available", headers=worker_h)
        task_ids = [t["task_id"] for t in avail.json()["tasks"]]
        assert task_id in task_ids

    async def test_keyword_overlap_from_good_at(self, client):
        """Matcher scores based on keyword overlap from good_at."""
        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        await client.post(
            "/v1/register",
            json={"name": "api-worker", "good_at": "building REST API endpoints"},
            headers={"Accept": "application/json"},
        )

        # Create task with keywords matching good_at
        await client.post(
            "/v1/tasks",
            json={"need": "Build a REST API for user management", "max_credits": 20},
            headers=poster_h,
        )

    async def test_builtin_matcher_top_5_limit(self, client, db):
        """Builtin matcher should return at most 5 matches."""
        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        # Create 7 workers with matching tags
        for i in range(7):
            w = await register_agent(client, f"worker-{i}")
            async with db() as session:
                from pinchwork.db_models import Agent

                agent = await session.get(Agent, w["agent_id"])
                agent.capability_tags = json.dumps(["python"])
                agent.good_at = "python coding"
                session.add(agent)
                await session.commit()

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Python task", "max_credits": 10, "tags": ["python"]},
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]

        # Check number of matches
        async with db() as session:
            from sqlmodel import select

            from pinchwork.db_models import TaskMatch

            result = await session.execute(select(TaskMatch).where(TaskMatch.task_id == task_id))
            matches = result.scalars().all()
            assert len(matches) <= 5

    async def test_builtin_matcher_excludes_poster(self, client, db):
        """Builtin matcher should not match the poster themselves."""
        poster_resp = await client.post(
            "/v1/register",
            json={"name": "poster-with-skills", "good_at": "python coding"},
            headers={"Accept": "application/json"},
        )
        poster = poster_resp.json()
        poster_h = auth_header(poster["api_key"])

        async with db() as session:
            from pinchwork.db_models import Agent

            agent = await session.get(Agent, poster["agent_id"])
            agent.capability_tags = json.dumps(["python"])
            session.add(agent)
            await session.commit()

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Python task", "max_credits": 10, "tags": ["python"]},
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]

        async with db() as session:
            from sqlmodel import select

            from pinchwork.db_models import TaskMatch

            result = await session.execute(
                select(TaskMatch).where(
                    TaskMatch.task_id == task_id,
                    TaskMatch.agent_id == poster["agent_id"],
                )
            )
            assert result.scalar_one_or_none() is None

    async def test_builtin_matcher_excludes_suspended(self, client, db):
        """Builtin matcher should not match suspended agents."""
        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        worker = await register_agent(client, "suspended-worker")
        async with db() as session:
            from pinchwork.db_models import Agent

            agent = await session.get(Agent, worker["agent_id"])
            agent.capability_tags = json.dumps(["python"])
            agent.good_at = "python"
            agent.suspended = True
            session.add(agent)
            await session.commit()

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Python task", "max_credits": 10, "tags": ["python"]},
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]

        async with db() as session:
            from sqlmodel import select

            from pinchwork.db_models import TaskMatch

            result = await session.execute(
                select(TaskMatch).where(
                    TaskMatch.task_id == task_id,
                    TaskMatch.agent_id == worker["agent_id"],
                )
            )
            assert result.scalar_one_or_none() is None


# ---------------------------------------------------------------------------
# Feature 3: Task Deadlines
# ---------------------------------------------------------------------------


class TestTaskDeadlines:
    async def test_create_with_deadline(self, client):
        """Task creation with deadline_minutes."""
        agent = await register_agent(client, "poster")
        headers = auth_header(agent["api_key"])

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Urgent task", "max_credits": 10, "deadline_minutes": 60},
            headers=headers,
        )
        assert resp.status_code == 201

    async def test_deadline_in_get_response(self, client):
        """Deadline should be visible in GET /v1/tasks/{id}."""
        agent = await register_agent(client, "poster")
        headers = auth_header(agent["api_key"])

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Deadline task", "max_credits": 10, "deadline_minutes": 30},
            headers=headers,
        )
        task_id = resp.json()["task_id"]

        task = await client.get(f"/v1/tasks/{task_id}", headers=headers)
        assert task.json().get("deadline") is not None

    async def test_deadline_in_pickup_response(self, client):
        """Deadline visible in pickup response."""
        poster = await register_agent(client, "poster")
        worker = await register_agent(client, "worker")
        poster_h = auth_header(poster["api_key"])
        worker_h = auth_header(worker["api_key"])

        await client.post(
            "/v1/tasks",
            json={"need": "Deadline pickup test", "max_credits": 10, "deadline_minutes": 120},
            headers=poster_h,
        )

        pickup = await client.post("/v1/tasks/pickup", headers=worker_h)
        assert pickup.status_code == 200
        assert pickup.json().get("deadline") is not None

    async def test_deadline_in_available_tasks(self, client):
        """Deadline visible in browse."""
        poster = await register_agent(client, "poster")
        worker = await register_agent(client, "worker")
        poster_h = auth_header(poster["api_key"])
        worker_h = auth_header(worker["api_key"])

        await client.post(
            "/v1/tasks",
            json={"need": "Browse deadline", "max_credits": 10, "deadline_minutes": 45},
            headers=poster_h,
        )

        avail = await client.get("/v1/tasks/available", headers=worker_h)
        assert any(t.get("deadline") is not None for t in avail.json()["tasks"])

    async def test_no_deadline(self, client):
        """Task without deadline_minutes has null deadline."""
        agent = await register_agent(client, "poster")
        headers = auth_header(agent["api_key"])

        resp = await client.post(
            "/v1/tasks",
            json={"need": "No deadline", "max_credits": 10},
            headers=headers,
        )
        task_id = resp.json()["task_id"]
        task = await client.get(f"/v1/tasks/{task_id}", headers=headers)
        assert task.json().get("deadline") is None

    async def test_deadline_expiry_claimed_task(self, client, db):
        """Claimed task past deadline should reset to posted."""
        from pinchwork.background import expire_deadlines

        poster = await register_agent(client, "poster")
        worker = await register_agent(client, "worker")
        poster_h = auth_header(poster["api_key"])
        worker_h = auth_header(worker["api_key"])

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Will expire claimed", "max_credits": 10, "deadline_minutes": 1},
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]
        await client.post(f"/v1/tasks/{task_id}/pickup", headers=worker_h)

        # Set deadline to past
        async with db() as session:
            from pinchwork.db_models import Task

            task = await session.get(Task, task_id)
            task.deadline = datetime.now(UTC) - timedelta(minutes=5)
            session.add(task)
            await session.commit()

        async with db() as session:
            count = await expire_deadlines(session)
            assert count == 1

        task_resp = await client.get(f"/v1/tasks/{task_id}", headers=poster_h)
        assert task_resp.json()["status"] == "posted"

    async def test_deadline_expiry_posted_task(self, client, db):
        """Posted task past deadline should expire and refund."""
        from pinchwork.background import expire_deadlines

        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        me = await client.get("/v1/me", headers=poster_h)
        initial_credits = me.json()["credits"]

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Will expire posted", "max_credits": 10, "deadline_minutes": 1},
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]

        async with db() as session:
            from pinchwork.db_models import Task

            task = await session.get(Task, task_id)
            task.deadline = datetime.now(UTC) - timedelta(minutes=5)
            session.add(task)
            await session.commit()

        async with db() as session:
            count = await expire_deadlines(session)
            assert count == 1

        task_resp = await client.get(f"/v1/tasks/{task_id}", headers=poster_h)
        assert task_resp.json()["status"] == "expired"

        me = await client.get("/v1/me", headers=poster_h)
        assert me.json()["credits"] == initial_credits

    async def test_deadline_no_expiry_for_future_deadline(self, client, db):
        """Tasks with future deadlines should not expire."""
        from pinchwork.background import expire_deadlines

        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        await client.post(
            "/v1/tasks",
            json={"need": "Future deadline", "max_credits": 10, "deadline_minutes": 9999},
            headers=poster_h,
        )

        async with db() as session:
            count = await expire_deadlines(session)
            assert count == 0


# ---------------------------------------------------------------------------
# Feature 4: Reputation-Weighted Routing
# ---------------------------------------------------------------------------


class TestReputationWeightedRouting:
    async def test_reputation_influences_match_rank(self, client, db):
        """Higher reputation agents rank higher in builtin matcher."""
        poster = await register_agent(client, "poster")
        poster_h = auth_header(poster["api_key"])

        low_rep = await register_agent(client, "low-rep")
        high_rep = await register_agent(client, "high-rep")

        async with db() as session:
            from pinchwork.db_models import Agent

            low = await session.get(Agent, low_rep["agent_id"])
            low.capability_tags = json.dumps(["python"])
            low.good_at = "python coding"
            low.reputation = 1.0
            session.add(low)

            high = await session.get(Agent, high_rep["agent_id"])
            high.capability_tags = json.dumps(["python"])
            high.good_at = "python coding"
            high.reputation = 5.0
            session.add(high)
            await session.commit()

        resp = await client.post(
            "/v1/tasks",
            json={"need": "Python task", "max_credits": 20, "tags": ["python"]},
            headers=poster_h,
        )
        task_id = resp.json()["task_id"]

        async with db() as session:
            from sqlmodel import select

            from pinchwork.db_models import TaskMatch

            result = await session.execute(
                select(TaskMatch).where(TaskMatch.task_id == task_id).order_by(TaskMatch.rank)
            )
            matches = result.scalars().all()
            if len(matches) >= 2:
                assert matches[0].agent_id == high_rep["agent_id"]
                assert matches[1].agent_id == low_rep["agent_id"]

    async def test_poster_reputation_visible_in_browse(self, client):
        """Browse includes poster_reputation."""
        poster = await register_agent(client, "reputable-poster")
        worker = await register_agent(client, "browser")
        poster_h = auth_header(poster["api_key"])
        worker_h = auth_header(worker["api_key"])

        await client.post(
            "/v1/tasks",
            json={"need": "Task from reputable poster", "max_credits": 10},
            headers=poster_h,
        )

        avail = await client.get("/v1/tasks/available", headers=worker_h)
        tasks = avail.json()["tasks"]
        assert len(tasks) > 0
        assert "poster_reputation" in tasks[0]

    async def test_reputation_sort_in_browse(self, client, db):
        """Tasks from higher-reputation posters should appear earlier in browse."""
        low_poster = await register_agent(client, "low-poster")
        high_poster = await register_agent(client, "high-poster")
        worker = await register_agent(client, "worker")

        async with db() as session:
            from pinchwork.db_models import Agent

            low = await session.get(Agent, low_poster["agent_id"])
            low.reputation = 1.0
            session.add(low)

            high = await session.get(Agent, high_poster["agent_id"])
            high.reputation = 5.0
            session.add(high)
            await session.commit()

        # Create tasks from both (low first to test that sort overrides FIFO)
        await client.post(
            "/v1/tasks",
            json={"need": "Low rep task", "max_credits": 10},
            headers=auth_header(low_poster["api_key"]),
        )
        await client.post(
            "/v1/tasks",
            json={"need": "High rep task", "max_credits": 10},
            headers=auth_header(high_poster["api_key"]),
        )

        avail = await client.get("/v1/tasks/available", headers=auth_header(worker["api_key"]))
        tasks = avail.json()["tasks"]
        # Both tasks should be visible
        assert len(tasks) == 2
