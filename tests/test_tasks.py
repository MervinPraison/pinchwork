import pytest


def hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def jhdr(key: str) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_full_cycle_json(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Delegate
    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={
            "need": "Send an SMS to +31612345678: Your deployment succeeded at 14:32 UTC",
            "max_credits": 10,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "posted"
    task_id = data["task_id"]

    # Check poster credits decreased
    resp = await c.get("/v1/me", headers=hdr(poster["key"]))
    assert resp.json()["credits"] == 90  # 100 - 10 escrowed

    # Pickup
    resp = await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    assert resp.status_code == 200
    picked = resp.json()
    assert picked["task_id"] == task_id

    # Deliver (credits_claimed defaults to max_credits=10)
    resp = await c.post(
        f"/v1/tasks/{task_id}/deliver",
        headers=jhdr(worker["key"]),
        json={
            "result": (
                "SMS sent. SID: SM1234567890abcdef,"
                " status: delivered, timestamp: 2025-01-15T14:32:05Z"
            ),
        },
    )
    assert resp.status_code == 200
    delivered = resp.json()
    assert delivered["status"] == "delivered"

    # Poll
    resp = await c.get(f"/v1/tasks/{task_id}", headers=hdr(poster["key"]))
    assert resp.status_code == 200
    assert resp.json()["status"] == "delivered"

    # Approve
    resp = await c.post(f"/v1/tasks/{task_id}/approve", headers=hdr(poster["key"]))
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"

    # Check worker earned credits
    resp = await c.get("/v1/me", headers=hdr(worker["key"]))
    assert resp.json()["credits"] == 109  # 100 + 10 earned - 1 platform fee (10%)


@pytest.mark.asyncio
async def test_full_cycle_markdown(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Delegate with markdown
    md_body = (
        "---\nmax_credits: 5\n---\n"
        "Review this SaaS ToS for red flags: liability caps, notice periods, dispute resolution"
    )
    resp = await c.post(
        "/v1/tasks",
        headers={**hdr(poster["key"])},
        content=md_body.encode(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "posted"
    task_id = data["task_id"]

    # Pickup
    resp = await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    assert resp.status_code == 200

    # Deliver with plain text
    resp = await c.post(
        f"/v1/tasks/{task_id}/deliver",
        headers=hdr(worker["key"]),
        content=(
            b"No red flags found. Liability cap is 12 months fees (standard)."
            b" 30-day pricing change notice. Arbitration in NL."
        ),
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_reject_resets_to_posted(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Delegate
    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Do something", "max_credits": 15},
    )
    task_id = resp.json()["task_id"]

    # Pickup + deliver
    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    await c.post(
        f"/v1/tasks/{task_id}/deliver",
        headers=jhdr(worker["key"]),
        json={"result": "Bad result"},
    )

    # Reject
    resp = await c.post(
        f"/v1/tasks/{task_id}/reject",
        json={"reason": "Not good enough"},
        headers=hdr(poster["key"]),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "claimed"  # Worker keeps claim during grace period


@pytest.mark.asyncio
async def test_no_tasks_available(registered_agent):
    client, _, api_key = registered_agent
    resp = await client.post("/v1/tasks/pickup", headers=hdr(api_key))
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_cant_pickup_own_task(registered_agent):
    client, _, api_key = registered_agent
    # Post a task
    resp = await client.post(
        "/v1/tasks",
        headers=jhdr(api_key),
        json={"need": "Self task", "max_credits": 5},
    )
    assert resp.status_code == 201

    # Try to pick up own task
    resp = await client.post("/v1/tasks/pickup", headers=hdr(api_key))
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_insufficient_credits(registered_agent):
    client, _, api_key = registered_agent
    resp = await client.post(
        "/v1/tasks",
        headers=jhdr(api_key),
        json={"need": "Expensive task", "max_credits": 999},
    )
    assert resp.status_code == 402


@pytest.mark.asyncio
async def test_wrong_worker_cant_deliver(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Register a third agent
    resp = await c.post(
        "/v1/register", json={"name": "intruder"}, headers={"Accept": "application/json"}
    )
    intruder_key = resp.json()["api_key"]

    # Post and pickup
    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Secret task", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]
    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))

    # Intruder tries to deliver
    resp = await c.post(
        f"/v1/tasks/{task_id}/deliver",
        headers=jhdr(intruder_key),
        json={"result": "Hacked"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_task_visibility(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]

    # Register third agent
    resp = await c.post(
        "/v1/register", json={"name": "outsider"}, headers={"Accept": "application/json"}
    )
    outsider_key = resp.json()["api_key"]

    # Post task
    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Private task", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]

    # Outsider can't see it (returns 404 to prevent task ID enumeration)
    resp = await c.get(f"/v1/tasks/{task_id}", headers=hdr(outsider_key))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Context field tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_with_context(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={
            "need": "Review this endpoint for security vulnerabilities",
            "context": (
                "FastAPI app, Python 3.12. Endpoint handles user settings updates."
                " Focus on injection and auth bypass."
            ),
            "max_credits": 10,
        },
    )
    assert resp.status_code == 201

    # Pickup should include context
    resp = await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    assert resp.status_code == 200
    data = resp.json()
    expected_ctx = (
        "FastAPI app, Python 3.12. Endpoint handles user settings updates."
        " Focus on injection and auth bypass."
    )
    assert data["context"] == expected_ctx


@pytest.mark.asyncio
async def test_pickup_includes_context(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Do work", "context": "Important background", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]

    # Poll should include context
    resp = await c.get(f"/v1/tasks/{task_id}", headers=hdr(poster["key"]))
    assert resp.json()["context"] == "Important background"


@pytest.mark.asyncio
async def test_context_optional(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "No context task", "max_credits": 5},
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    resp = await c.get(f"/v1/tasks/{task_id}", headers=hdr(poster["key"]))
    assert resp.json()["context"] is None


# ---------------------------------------------------------------------------
# Browse available tasks tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_empty(registered_agent):
    client, _, api_key = registered_agent
    resp = await client.get("/v1/tasks/available", headers=hdr(api_key))
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_browse_basic(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Post a task
    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Browse me", "max_credits": 5},
    )
    assert resp.status_code == 201

    # Worker browses
    resp = await c.get("/v1/tasks/available", headers=hdr(worker["key"]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["tasks"][0]["need"] == "Browse me"


@pytest.mark.asyncio
async def test_browse_excludes_own_tasks(registered_agent):
    client, _, api_key = registered_agent

    resp = await client.post(
        "/v1/tasks",
        headers=jhdr(api_key),
        json={"need": "My own task", "max_credits": 5},
    )
    assert resp.status_code == 201

    resp = await client.get("/v1/tasks/available", headers=hdr(api_key))
    assert resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_browse_excludes_system_tasks(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Register an infra agent so system tasks get created
    resp = await c.post(
        "/v1/register",
        json={"name": "infra", "accepts_system_tasks": True},
        headers={"Accept": "application/json"},
    )
    # Post a task (will spawn a system matching task)
    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Regular task", "max_credits": 5},
    )
    assert resp.status_code == 201

    # Worker browse should not show system tasks
    resp = await c.get("/v1/tasks/available", headers=hdr(worker["key"]))
    data = resp.json()
    for task in data["tasks"]:
        assert "Match agents" not in task["need"]


@pytest.mark.asyncio
async def test_browse_pagination(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Post 3 tasks
    for i in range(3):
        await c.post(
            "/v1/tasks",
            headers=jhdr(poster["key"]),
            json={"need": f"Task {i}", "max_credits": 5},
        )

    resp = await c.get("/v1/tasks/available?limit=2&offset=0", headers=hdr(worker["key"]))
    data = resp.json()
    assert data["total"] == 3
    assert len(data["tasks"]) == 2

    resp = await c.get("/v1/tasks/available?limit=2&offset=2", headers=hdr(worker["key"]))
    data = resp.json()
    assert data["total"] == 3
    assert len(data["tasks"]) == 1


@pytest.mark.asyncio
async def test_browse_tag_filtering(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={
            "need": "Review API endpoint for OWASP Top 10 vulnerabilities",
            "max_credits": 5,
            "tags": ["security-audit"],
        },
    )
    await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={
            "need": "Send SMS delivery confirmation to +31612345678",
            "max_credits": 5,
            "tags": ["sms-delivery"],
        },
    )

    resp = await c.get("/v1/tasks/available?tags=security-audit", headers=hdr(worker["key"]))
    data = resp.json()
    assert data["total"] == 1
    assert data["tasks"][0]["need"] == "Review API endpoint for OWASP Top 10 vulnerabilities"


@pytest.mark.asyncio
async def test_browse_includes_context(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Task with ctx", "context": "Some background", "max_credits": 5},
    )

    resp = await c.get("/v1/tasks/available", headers=hdr(worker["key"]))
    data = resp.json()
    assert data["tasks"][0]["context"] == "Some background"


# ---------------------------------------------------------------------------
# Abandon task tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abandon_happy_path(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Abandon me", "max_credits": 10},
    )
    task_id = resp.json()["task_id"]

    # Pickup
    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))

    # Abandon
    resp = await c.post(f"/v1/tasks/{task_id}/abandon", headers=hdr(worker["key"]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "posted"
    assert data["worker_id"] is None


@pytest.mark.asyncio
async def test_abandon_sets_broadcast(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Broadcast after abandon", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]

    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    await c.post(f"/v1/tasks/{task_id}/abandon", headers=hdr(worker["key"]))

    # Task should be available again for browse
    resp = await c.get("/v1/tasks/available", headers=hdr(worker["key"]))
    data = resp.json()
    assert data["total"] == 1
    assert data["tasks"][0]["task_id"] == task_id


@pytest.mark.asyncio
async def test_abandon_wrong_agent(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Not yours", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]

    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))

    # Poster tries to abandon (not the worker)
    resp = await c.post(f"/v1/tasks/{task_id}/abandon", headers=hdr(poster["key"]))
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_abandon_wrong_status(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Still posted", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]

    # Try to abandon a posted (not claimed) task
    resp = await c.post(f"/v1/tasks/{task_id}/abandon", headers=hdr(worker["key"]))
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_abandon_no_credit_penalty(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    # Check worker starting credits
    resp = await c.get("/v1/me", headers=hdr(worker["key"]))
    start_credits = resp.json()["credits"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Free abandon", "max_credits": 10},
    )
    task_id = resp.json()["task_id"]

    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    await c.post(f"/v1/tasks/{task_id}/abandon", headers=hdr(worker["key"]))

    # Worker credits unchanged
    resp = await c.get("/v1/me", headers=hdr(worker["key"]))
    assert resp.json()["credits"] == start_credits


@pytest.mark.asyncio
async def test_abandon_refreshes_expiry(two_agents):
    c = two_agents["client"]
    poster = two_agents["poster"]
    worker = two_agents["worker"]

    resp = await c.post(
        "/v1/tasks",
        headers=jhdr(poster["key"]),
        json={"need": "Expiry test", "max_credits": 5},
    )
    task_id = resp.json()["task_id"]

    await c.post("/v1/tasks/pickup", headers=hdr(worker["key"]))
    await c.post(f"/v1/tasks/{task_id}/abandon", headers=hdr(worker["key"]))

    # Task should be back to posted and available for pickup
    resp = await c.get(f"/v1/tasks/{task_id}", headers=hdr(poster["key"]))
    assert resp.status_code == 200
    assert resp.json()["status"] == "posted"
