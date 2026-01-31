"""Tests for all 8 new features: rejection reasons, questions, search,
stats, agent discovery, reputation breakdown, batch pickup, capabilities."""

import pytest

from tests.conftest import auth_header, register_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def create_task(
    client, api_key, need="Do something", max_credits=50, tags=None, context=None
):
    """Create a task and return the response JSON."""
    body = {"need": need, "max_credits": max_credits}
    if tags:
        body["tags"] = tags
    if context:
        body["context"] = context
    resp = await client.post("/v1/tasks", json=body, headers=auth_header(api_key))
    assert resp.status_code == 201
    return resp.json()


async def pickup(client, api_key, **kwargs):
    """Pick up a task and return response."""
    return await client.post("/v1/tasks/pickup", headers=auth_header(api_key), params=kwargs)


async def deliver(client, api_key, task_id, result="Done", credits_claimed=None):
    """Deliver a task."""
    body = {"result": result}
    if credits_claimed is not None:
        body["credits_claimed"] = credits_claimed
    return await client.post(
        f"/v1/tasks/{task_id}/deliver", json=body, headers=auth_header(api_key)
    )


async def approve(client, api_key, task_id, rating=None):
    """Approve a task."""
    body = {}
    if rating is not None:
        body["rating"] = rating
    return await client.post(
        f"/v1/tasks/{task_id}/approve", json=body, headers=auth_header(api_key)
    )


async def full_task_cycle(
    client, poster_key, worker_key, need="Do X", tags=None, rating=None, credits=30
):
    """Create → pickup → deliver → approve a task. Returns task_id."""
    task = await create_task(client, poster_key, need=need, max_credits=credits, tags=tags)
    task_id = task["task_id"]
    pickup_resp = await pickup(client, worker_key)
    assert pickup_resp.status_code == 200
    picked = pickup_resp.json()
    assert picked["task_id"] == task_id
    await deliver(client, worker_key, task_id, result="Completed")
    await approve(client, poster_key, task_id, rating=rating)
    return task_id


# ===========================================================================
# Feature 1: Required Rejection Reason + Rejection History
# ===========================================================================


class TestRejectionReason:
    @pytest.mark.asyncio
    async def test_reject_without_reason_returns_400(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], task["task_id"])

        # Reject with empty body
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/reject", json={}, headers=auth_header(poster["key"])
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_reject_with_reason_succeeds(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], task["task_id"])

        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/reject",
            json={"reason": "Output was incomplete"},
            headers=auth_header(poster["key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rejection_reason"] == "Output was incomplete"
        assert data["rejection_count"] == 1
        assert data["status"] == "claimed"  # grace period keeps claim

    @pytest.mark.asyncio
    async def test_rejection_count_increments(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        # First cycle
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], task["task_id"])
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/reject",
            json={"reason": "First rejection"},
            headers=auth_header(poster["key"]),
        )
        assert resp.json()["rejection_count"] == 1

        # Second cycle — worker keeps claim due to grace period, no re-pickup needed
        await deliver(c, worker["key"], task["task_id"])
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/reject",
            json={"reason": "Second rejection"},
            headers=auth_header(poster["key"]),
        )
        assert resp.json()["rejection_count"] == 2

    @pytest.mark.asyncio
    async def test_rejection_count_visible_in_browse(self, two_agents, db):
        """After grace period expires, rejection count is visible in browse."""
        from datetime import UTC, datetime, timedelta

        from pinchwork.background import expire_rejection_grace

        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], task["task_id"])
        await c.post(
            f"/v1/tasks/{task['task_id']}/reject",
            json={"reason": "Bad quality"},
            headers=auth_header(poster["key"]),
        )

        # Expire the grace period so task goes back to posted
        async with db() as session:
            from pinchwork.db_models import Task

            t = await session.get(Task, task["task_id"])
            t.rejection_grace_deadline = datetime.now(UTC) - timedelta(minutes=1)
            session.add(t)
            await session.commit()

        async with db() as session:
            await expire_rejection_grace(session)

        browse = await c.get("/v1/tasks/available", headers=auth_header(worker["key"]))
        assert browse.status_code == 200
        tasks = browse.json()["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["rejection_count"] == 1


# ===========================================================================
# Feature 2: Task Questions (Pre-Pickup Clarification)
# ===========================================================================


class TestTaskQuestions:
    @pytest.mark.asyncio
    async def test_ask_and_answer_question(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"], need="Write a parser")

        # Worker asks a question
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "What input format?"},
            headers=auth_header(worker["key"]),
        )
        assert resp.status_code == 201
        q = resp.json()
        assert q["question"] == "What input format?"
        assert q["answer"] is None
        qid = q["id"]

        # Poster answers
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions/{qid}/answer",
            json={"answer": "JSON lines"},
            headers=auth_header(poster["key"]),
        )
        assert resp.status_code == 200
        a = resp.json()
        assert a["answer"] == "JSON lines"
        assert a["answered_at"] is not None

    @pytest.mark.asyncio
    async def test_list_questions(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])

        await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "Q1"},
            headers=auth_header(worker["key"]),
        )
        await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "Q2"},
            headers=auth_header(worker["key"]),
        )

        resp = await c.get(
            f"/v1/tasks/{task['task_id']}/questions",
            headers=auth_header(worker["key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_poster_cannot_ask_own_task(self, two_agents):
        c = two_agents["client"]
        poster = two_agents["poster"]

        task = await create_task(c, poster["key"])
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "Self-question?"},
            headers=auth_header(poster["key"]),
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_max_unanswered_questions(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])

        for i in range(5):
            resp = await c.post(
                f"/v1/tasks/{task['task_id']}/questions",
                json={"question": f"Q{i}"},
                headers=auth_header(worker["key"]),
            )
            assert resp.status_code == 201

        # 6th should fail
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "Q6 too many"},
            headers=auth_header(worker["key"]),
        )
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_only_poster_can_answer(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "Q?"},
            headers=auth_header(worker["key"]),
        )
        qid = resp.json()["id"]

        # Worker tries to answer
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions/{qid}/answer",
            json={"answer": "I'll answer myself"},
            headers=auth_header(worker["key"]),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_cannot_answer_twice(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions",
            json={"question": "Q?"},
            headers=auth_header(worker["key"]),
        )
        qid = resp.json()["id"]

        await c.post(
            f"/v1/tasks/{task['task_id']}/questions/{qid}/answer",
            json={"answer": "First answer"},
            headers=auth_header(poster["key"]),
        )
        resp = await c.post(
            f"/v1/tasks/{task['task_id']}/questions/{qid}/answer",
            json={"answer": "Second answer"},
            headers=auth_header(poster["key"]),
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_question_on_nonexistent_task(self, client):
        agent = await register_agent(client)
        resp = await client.post(
            "/v1/tasks/tk-nonexistent/questions",
            json={"question": "Q?"},
            headers=auth_header(agent["api_key"]),
        )
        assert resp.status_code == 404


# ===========================================================================
# Feature 3: Full-Text Search
# ===========================================================================


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_by_keyword_in_need(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Deploy Kubernetes cluster")
        await create_task(c, poster["key"], need="Write Python script")

        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": "kubernetes"},
        )
        assert resp.status_code == 200
        tasks = resp.json()["tasks"]
        assert len(tasks) == 1
        assert "Kubernetes" in tasks[0]["need"]

    @pytest.mark.asyncio
    async def test_search_by_keyword_in_context(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(
            c, poster["key"], need="Fix the issue", context="Docker container hardening"
        )

        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": "hardening"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="URGENT: Fix Database")

        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": "database"},
        )
        assert len(resp.json()["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_search_with_tags_combined(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Fix K8s bug", tags=["devops"])
        await create_task(c, poster["key"], need="Fix K8s config", tags=["backend"])

        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": "K8s", "tags": "devops"},
        )
        tasks = resp.json()["tasks"]
        assert len(tasks) == 1
        assert "bug" in tasks[0]["need"]

    @pytest.mark.asyncio
    async def test_search_empty_returns_all(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Task A")
        await create_task(c, poster["key"], need="Task B")

        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": ""},
        )
        assert len(resp.json()["tasks"]) == 2

    @pytest.mark.asyncio
    async def test_search_on_pickup(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Kubernetes deployment task")
        await create_task(c, poster["key"], need="Python linting task")

        resp = await c.post(
            "/v1/tasks/pickup",
            headers=auth_header(worker["key"]),
            params={"search": "kubernetes"},
        )
        assert resp.status_code == 200
        assert "Kubernetes" in resp.json()["need"]

    @pytest.mark.asyncio
    async def test_search_no_results(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Regular task")

        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": "nonexistentthing"},
        )
        assert resp.json()["total"] == 0


# ===========================================================================
# Feature 4: Earnings Dashboard (GET /v1/me/stats)
# ===========================================================================


class TestEarningsStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, registered_agent):
        c, _, api_key = registered_agent
        resp = await c.get("/v1/me/stats", headers=auth_header(api_key))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_earned"] == 0
        assert data["total_spent"] == 0
        assert data["approval_rate"] is None

    @pytest.mark.asyncio
    async def test_stats_after_task_completion(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await full_task_cycle(c, poster["key"], worker["key"], tags=["python"], credits=30)

        # Worker stats
        resp = await c.get("/v1/me/stats", headers=auth_header(worker["key"]))
        data = resp.json()
        assert data["total_earned"] > 0
        assert data["approval_rate"] == 1.0
        assert data["recent_7d_earned"] > 0
        assert data["recent_30d_earned"] > 0

    @pytest.mark.asyncio
    async def test_stats_tasks_by_tag(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await full_task_cycle(c, poster["key"], worker["key"], tags=["python"], credits=20)
        await full_task_cycle(c, poster["key"], worker["key"], tags=["rust"], credits=30)

        resp = await c.get("/v1/me/stats", headers=auth_header(worker["key"]))
        data = resp.json()
        tags_earned = {t["tag"]: t["earned"] for t in data["tasks_by_tag"]}
        assert "python" in tags_earned
        assert "rust" in tags_earned

    @pytest.mark.asyncio
    async def test_stats_poster_spent(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await full_task_cycle(c, poster["key"], worker["key"], credits=20)

        resp = await c.get("/v1/me/stats", headers=auth_header(poster["key"]))
        data = resp.json()
        assert data["total_spent"] == 20


# ===========================================================================
# Feature 5: Agent Discovery (GET /v1/agents)
# ===========================================================================


class TestAgentDiscovery:
    @pytest.mark.asyncio
    async def test_list_agents(self, client):
        await register_agent(client, "agent-alpha")
        await register_agent(client, "agent-beta")

        resp = await client.get("/v1/agents", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2

    @pytest.mark.asyncio
    async def test_search_agents_by_skill(self, client):
        a = await register_agent(client, "py-expert")
        # Update good_at
        await client.patch(
            "/v1/me",
            json={"good_at": "Python data analysis and machine learning"},
            headers=auth_header(a["api_key"]),
        )

        resp = await client.get(
            "/v1/agents",
            headers={"Accept": "application/json"},
            params={"search": "machine learning"},
        )
        data = resp.json()
        assert data["total"] >= 1
        assert any("machine learning" in (ag.get("good_at") or "").lower() for ag in data["agents"])

    @pytest.mark.asyncio
    async def test_filter_by_min_reputation(self, client):
        await register_agent(client, "new-agent")

        resp = await client.get(
            "/v1/agents",
            headers={"Accept": "application/json"},
            params={"min_reputation": "4.0"},
        )
        data = resp.json()
        for a in data["agents"]:
            assert a["reputation"] >= 4.0

    @pytest.mark.asyncio
    async def test_sort_by_tasks_completed(self, client):
        await register_agent(client, "a1")
        await register_agent(client, "a2")

        resp = await client.get(
            "/v1/agents",
            headers={"Accept": "application/json"},
            params={"sort_by": "tasks_completed"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_agents_pagination(self, client):
        for i in range(3):
            await register_agent(client, f"page-agent-{i}")

        resp = await client.get(
            "/v1/agents",
            headers={"Accept": "application/json"},
            params={"limit": "2", "offset": "0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) <= 2

    @pytest.mark.asyncio
    async def test_agents_excludes_platform(self, client):
        resp = await client.get("/v1/agents", headers={"Accept": "application/json"})
        data = resp.json()
        ids = [a["id"] for a in data["agents"]]
        assert "ag-platform" not in ids

    @pytest.mark.asyncio
    async def test_no_auth_required(self, client):
        resp = await client.get("/v1/agents", headers={"Accept": "application/json"})
        assert resp.status_code == 200


# ===========================================================================
# Feature 6: Richer Reputation (Per-Tag Breakdown)
# ===========================================================================


class TestReputationBreakdown:
    @pytest.mark.asyncio
    async def test_agent_profile_includes_breakdown(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await full_task_cycle(c, poster["key"], worker["key"], tags=["python"], rating=5)
        await full_task_cycle(c, poster["key"], worker["key"], tags=["rust"], rating=3)

        resp = await c.get(f"/v1/agents/{worker['id']}", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("reputation_by_tag") is not None
        tags_map = {t["tag"]: t for t in data["reputation_by_tag"]}
        assert tags_map["python"]["avg_rating"] == 5.0
        assert tags_map["rust"]["avg_rating"] == 3.0

    @pytest.mark.asyncio
    async def test_agent_profile_includes_good_at(self, client):
        a = await register_agent(client, "skilled-agent")
        await client.patch(
            "/v1/me",
            json={"good_at": "Expert in K8s"},
            headers=auth_header(a["api_key"]),
        )

        resp = await client.get(
            f"/v1/agents/{a['agent_id']}", headers={"Accept": "application/json"}
        )
        data = resp.json()
        assert data["good_at"] == "Expert in K8s"

    @pytest.mark.asyncio
    async def test_no_ratings_empty_breakdown(self, registered_agent):
        c, agent_id, _ = registered_agent
        resp = await c.get(f"/v1/agents/{agent_id}", headers={"Accept": "application/json"})
        data = resp.json()
        assert data.get("reputation_by_tag") is None


# ===========================================================================
# Feature 7: Batch Pickup
# ===========================================================================


class TestBatchPickup:
    @pytest.mark.asyncio
    async def test_batch_pickup_multiple(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        for i in range(5):
            await create_task(c, poster["key"], need=f"Task {i}", max_credits=10)

        resp = await c.post(
            "/v1/tasks/pickup/batch",
            json={"count": 3},
            headers=auth_header(worker["key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["tasks"]) == 3

    @pytest.mark.asyncio
    async def test_batch_pickup_fewer_available(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Only one task")

        resp = await c.post(
            "/v1/tasks/pickup/batch",
            json={"count": 10},
            headers=auth_header(worker["key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_batch_pickup_none_available(self, registered_agent):
        c, _, api_key = registered_agent
        resp = await c.post(
            "/v1/tasks/pickup/batch",
            json={"count": 5},
            headers=auth_header(api_key),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_batch_pickup_with_tags(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Python task", tags=["python"])
        await create_task(c, poster["key"], need="Rust task", tags=["rust"])

        resp = await c.post(
            "/v1/tasks/pickup/batch",
            json={"count": 5, "tags": ["python"]},
            headers=auth_header(worker["key"]),
        )
        data = resp.json()
        assert data["total"] == 1
        assert "Python" in data["tasks"][0]["need"]

    @pytest.mark.asyncio
    async def test_batch_pickup_with_search(self, two_agents):
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Deploy Kubernetes")
        await create_task(c, poster["key"], need="Write docs")

        resp = await c.post(
            "/v1/tasks/pickup/batch",
            json={"count": 5, "search": "kubernetes"},
            headers=auth_header(worker["key"]),
        )
        data = resp.json()
        assert data["total"] == 1


# ===========================================================================
# Feature 8: Capabilities Endpoint + Skill.md Section
# ===========================================================================


class TestCapabilities:
    @pytest.mark.asyncio
    async def test_capabilities_returns_json(self, client):
        resp = await client.get("/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "endpoints" in data
        assert "quick_start" in data
        assert "docs_url" in data
        assert "openapi_url" in data

    @pytest.mark.asyncio
    async def test_capabilities_includes_all_endpoints(self, client):
        resp = await client.get("/v1/capabilities")
        data = resp.json()
        paths = [e["path"] for e in data["endpoints"]]
        assert "/v1/register" in paths
        assert "/v1/tasks" in paths
        assert "/v1/tasks/pickup" in paths

    @pytest.mark.asyncio
    async def test_quick_start_sequence(self, client):
        resp = await client.get("/v1/capabilities")
        qs = resp.json()["quick_start"]
        assert qs[0] == "POST /v1/register"
        assert qs[-1] == "POST /v1/tasks/{id}/deliver"

    @pytest.mark.asyncio
    async def test_skill_md_section_param(self, client):
        resp = await client.get("/skill.md")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_skill_md_section_not_found(self, client):
        resp = await client.get("/skill.md", params={"section": "nonexistent_xyz"})
        assert resp.status_code == 404


# ===========================================================================
# Integration: Multiple features working together
# ===========================================================================


class TestIntegration:
    @pytest.mark.asyncio
    async def test_reject_then_question_then_redeliver(self, two_agents):
        """Full workflow: reject → ask question → answer → re-deliver (grace) → approve."""
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"], need="Complex task")
        tid = task["task_id"]

        # Worker picks up and delivers
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], tid)

        # Poster rejects with reason — worker keeps the claim (grace period)
        await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "Needs more detail"},
            headers=auth_header(poster["key"]),
        )

        # Worker asks a question (task is claimed, but questions should still work)
        # Note: questions require posted status, so we skip the question step here
        # and just re-deliver directly using the grace period

        # Worker re-delivers directly (no pickup needed, grace period)
        await deliver(c, worker["key"], tid, result="Improved version")
        await approve(c, poster["key"], tid, rating=5)

        # Check stats
        resp = await c.get("/v1/me/stats", headers=auth_header(worker["key"]))
        assert resp.json()["total_earned"] > 0

    @pytest.mark.asyncio
    async def test_search_then_batch_pickup(self, two_agents):
        """Search, then batch pickup matching tasks."""
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        await create_task(c, poster["key"], need="Kubernetes deploy", max_credits=10)
        await create_task(c, poster["key"], need="Kubernetes monitoring", max_credits=10)
        await create_task(c, poster["key"], need="Python script", max_credits=10)

        # Search first
        resp = await c.get(
            "/v1/tasks/available",
            headers=auth_header(worker["key"]),
            params={"search": "kubernetes"},
        )
        assert resp.json()["total"] == 2

        # Batch pickup with search filter
        resp = await c.post(
            "/v1/tasks/pickup/batch",
            json={"count": 10, "search": "kubernetes"},
            headers=auth_header(worker["key"]),
        )
        assert resp.json()["total"] == 2


# ===========================================================================
# Feature 9: Rejection Grace Period
# ===========================================================================


class TestRejectionGracePeriod:
    @pytest.mark.asyncio
    async def test_worker_can_redeliver_after_rejection(self, two_agents):
        """Worker keeps the claim after rejection and can re-deliver without re-pickup."""
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        tid = task["task_id"]
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], tid, result="First attempt")

        # Reject
        resp = await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "Incomplete"},
            headers=auth_header(poster["key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "claimed"
        assert data["worker_id"] == worker["id"]
        assert "rejection_grace_deadline" in data

        # Worker re-delivers directly (no pickup needed)
        resp = await deliver(c, worker["key"], tid, result="Second attempt")
        assert resp.status_code == 200
        assert resp.json()["status"] == "delivered"

    @pytest.mark.asyncio
    async def test_grace_period_expiry_resets_to_posted(self, two_agents, db):
        """After grace period expires, background loop resets task to posted."""
        from datetime import UTC, datetime, timedelta

        from pinchwork.background import expire_rejection_grace

        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        tid = task["task_id"]
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], tid)

        # Reject
        await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "Bad"},
            headers=auth_header(poster["key"]),
        )

        # Manually set grace deadline to the past
        async with db() as session:
            from pinchwork.db_models import Task

            t = await session.get(Task, tid)
            t.rejection_grace_deadline = datetime.now(UTC) - timedelta(minutes=1)
            session.add(t)
            await session.commit()

        # Run the background function
        async with db() as session:
            count = await expire_rejection_grace(session)
            assert count == 1

        # Check task is now posted with no worker
        resp = await c.get(f"/v1/tasks/{tid}", headers=auth_header(poster["key"]))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "posted"
        assert data["worker_id"] is None

    @pytest.mark.asyncio
    async def test_other_agent_cannot_pickup_during_grace(self, two_agents):
        """During grace period, the task is claimed — other agents can't pick it up."""
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        tid = task["task_id"]
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], tid)

        await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "Not good enough"},
            headers=auth_header(poster["key"]),
        )

        # Register a third agent who tries to pick up
        third = await register_agent(c, "third-agent")
        resp = await pickup(c, third["api_key"])
        # Should get 204 (no task available) since the only task is claimed
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_rejection_response_includes_grace_deadline(self, two_agents):
        """The rejection response includes the grace deadline."""
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        tid = task["task_id"]
        await pickup(c, worker["key"])
        await deliver(c, worker["key"], tid)

        resp = await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "Missing details"},
            headers=auth_header(poster["key"]),
        )
        data = resp.json()
        assert "rejection_grace_deadline" in data
        # Deadline should be parseable as ISO datetime
        from datetime import datetime

        deadline = datetime.fromisoformat(data["rejection_grace_deadline"])
        assert deadline is not None

    @pytest.mark.asyncio
    async def test_multiple_rejections_with_grace(self, two_agents):
        """Worker can be rejected multiple times, each with a fresh grace period."""
        c = two_agents["client"]
        poster, worker = two_agents["poster"], two_agents["worker"]

        task = await create_task(c, poster["key"])
        tid = task["task_id"]
        await pickup(c, worker["key"])

        # First rejection cycle
        await deliver(c, worker["key"], tid, result="Attempt 1")
        resp = await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "First rejection"},
            headers=auth_header(poster["key"]),
        )
        assert resp.json()["rejection_count"] == 1
        assert resp.json()["status"] == "claimed"
        first_deadline = resp.json()["rejection_grace_deadline"]

        # Re-deliver (no pickup needed) and second rejection
        await deliver(c, worker["key"], tid, result="Attempt 2")
        resp = await c.post(
            f"/v1/tasks/{tid}/reject",
            json={"reason": "Second rejection"},
            headers=auth_header(poster["key"]),
        )
        assert resp.json()["rejection_count"] == 2
        assert resp.json()["status"] == "claimed"
        second_deadline = resp.json()["rejection_grace_deadline"]
        assert second_deadline >= first_deadline

        # Worker can still re-deliver a third time
        resp = await deliver(c, worker["key"], tid, result="Attempt 3")
        assert resp.status_code == 200
