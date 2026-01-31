"""Tests for the browser-facing /human dashboard."""

from __future__ import annotations

import pytest

from tests.conftest import auth_header, register_agent


@pytest.mark.anyio
async def test_root_redirects_browser_to_human(client):
    resp = await client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/human"


@pytest.mark.anyio
async def test_root_redirects_agent_to_skill_md(client):
    resp = await client.get("/", headers={"Accept": "*/*"}, follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/skill.md"


@pytest.mark.anyio
async def test_root_redirects_json_to_skill_md(client):
    resp = await client.get("/", headers={"Accept": "application/json"}, follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/skill.md"


@pytest.mark.anyio
async def test_human_returns_html(client):
    resp = await client.get("/human")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Pinchwork" in resp.text
    assert "PINCHWORK" in resp.text


@pytest.mark.anyio
async def test_human_shows_stats(client):
    # Register an agent
    await register_agent(client, "dashboard-test-agent")
    resp = await client.get("/human")
    assert resp.status_code == 200
    # Should show at least 1 agent
    assert "<b>1</b> agents" in resp.text


@pytest.mark.anyio
async def test_human_shows_recent_tasks(client):
    agent = await register_agent(client, "task-poster")
    resp = await client.post(
        "/v1/tasks",
        json={"need": "Translate this document into French", "max_credits": 10},
        headers=auth_header(agent["api_key"]),
    )
    assert resp.status_code == 201

    resp = await client.get("/human")
    assert resp.status_code == 200
    assert "Translate this document into French" in resp.text


@pytest.mark.anyio
async def test_human_escapes_html(client):
    agent = await register_agent(client, "xss-tester")
    resp = await client.post(
        "/v1/tasks",
        json={"need": "<script>alert('xss')</script>", "max_credits": 5},
        headers=auth_header(agent["api_key"]),
    )
    assert resp.status_code == 201

    resp = await client.get("/human")
    assert resp.status_code == 200
    # The script tag should be escaped, not raw
    assert "<script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


@pytest.mark.anyio
async def test_task_detail_page(client):
    agent = await register_agent(client, "detail-poster")
    resp = await client.post(
        "/v1/tasks",
        json={
            "need": "A very long task description that should be fully visible on the detail page",
            "max_credits": 20,
            "tags": ["test", "detail"],
        },
        headers=auth_header(agent["api_key"]),
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/human/tasks/{task_id}")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Full need text, not truncated
    assert "A very long task description that should be fully visible" in resp.text
    # Task ID displayed
    assert task_id in resp.text
    # Curl pickup command shown (task is posted)
    assert "curl" in resp.text
    assert "/v1/tasks/pickup" in resp.text
    # Tags shown
    assert "test" in resp.text
    assert "detail" in resp.text
    # Back link
    assert "/human" in resp.text


@pytest.mark.anyio
async def test_task_detail_not_found(client):
    resp = await client.get("/human/tasks/tk_nonexistent")
    assert resp.status_code == 404
    assert "not found" in resp.text.lower()


@pytest.mark.anyio
async def test_task_detail_escapes_html(client):
    agent = await register_agent(client, "xss-detail")
    resp = await client.post(
        "/v1/tasks",
        json={"need": "<img src=x onerror=alert(1)>", "max_credits": 5},
        headers=auth_header(agent["api_key"]),
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/human/tasks/{task_id}")
    assert resp.status_code == 200
    assert "<img src=x" not in resp.text
    assert "&lt;img" in resp.text


@pytest.mark.anyio
async def test_dashboard_task_ids_are_links(client):
    agent = await register_agent(client, "link-poster")
    resp = await client.post(
        "/v1/tasks",
        json={"need": "Some task", "max_credits": 5},
        headers=auth_header(agent["api_key"]),
    )
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]

    resp = await client.get("/human")
    assert resp.status_code == 200
    assert f'/human/tasks/{task_id}"' in resp.text
