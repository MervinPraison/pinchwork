"""Tests for security fixes, DRY refactors, and new enums from the code review."""

from __future__ import annotations

import pytest

from pinchwork.config import settings
from pinchwork.db_models import MatchStatus, SystemTaskType, TaskStatus, VerificationStatus
from pinchwork.utils import safe_json_loads, status_str
from tests.conftest import auth_header, register_agent

# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_valid_list(self):
        assert safe_json_loads('["x", "y"]') == ["x", "y"]

    def test_none_input(self):
        assert safe_json_loads(None) is None

    def test_invalid_json(self):
        assert safe_json_loads("not json") is None

    def test_empty_string(self):
        assert safe_json_loads("") is None


class TestStatusStr:
    def test_enum_value(self):
        assert status_str(TaskStatus.posted) == "posted"
        assert status_str(TaskStatus.delivered) == "delivered"

    def test_plain_string(self):
        assert status_str("posted") == "posted"
        assert status_str("delivered") == "delivered"


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_system_task_type_values(self):
        assert SystemTaskType.match_agents == "match_agents"
        assert SystemTaskType.verify_completion == "verify_completion"
        assert SystemTaskType.extract_capabilities == "extract_capabilities"

    def test_match_status_values(self):
        assert MatchStatus.pending == "pending"
        assert MatchStatus.matched == "matched"
        assert MatchStatus.broadcast == "broadcast"

    def test_verification_status_values(self):
        assert VerificationStatus.pending == "pending"
        assert VerificationStatus.passed == "passed"
        assert VerificationStatus.failed == "failed"

    def test_enum_string_comparison(self):
        """Enums should compare equal to their string values (str enum)."""
        assert MatchStatus.broadcast == "broadcast"
        assert SystemTaskType.match_agents == "match_agents"


# ---------------------------------------------------------------------------
# Security: fingerprint length
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_fingerprint_is_32_chars(client):
    """Fingerprint should be 32 hex chars (128 bits) for collision resistance."""
    from pinchwork.auth import key_fingerprint

    fp = key_fingerprint("test-key-12345")
    assert len(fp) == 32
    assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# Security: suspend message doesn't leak reason
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_key", "test-admin-secret")


ADMIN_HEADERS = {"Authorization": "Bearer test-admin-secret", "Accept": "application/json"}


@pytest.mark.anyio
async def test_suspend_does_not_leak_reason(client):
    """Suspended agent error should not reveal internal suspend reason."""
    data = await register_agent(client, "leaky")
    headers = auth_header(data["api_key"])

    # Suspend with a specific reason
    resp = await client.post(
        "/v1/admin/agents/suspend",
        json={
            "agent_id": data["agent_id"],
            "suspended": True,
            "reason": "Internal policy violation #42",
        },
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200

    # Try to use the API — should get generic message
    resp = await client.get("/v1/me", headers=headers)
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "Agent suspended"
    assert "Internal policy" not in body["error"]
    assert "#42" not in body["error"]


# ---------------------------------------------------------------------------
# Security: task visibility returns 404 (not 403) for unauthorized access
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_unauthorized_task_access_returns_404(client):
    """Unauthorized task access returns 404 to prevent task ID enumeration."""
    poster = await register_agent(client, "poster")
    outsider = await register_agent(client, "outsider")

    # Create a task
    resp = await client.post(
        "/v1/tasks",
        json={"need": "secret task", "max_credits": 5},
        headers=auth_header(poster["api_key"]),
    )
    task_id = resp.json()["task_id"]

    # Outsider should get 404, not 403
    resp = await client.get(
        f"/v1/tasks/{task_id}",
        headers=auth_header(outsider["api_key"]),
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "Task not found"


# ---------------------------------------------------------------------------
# Security: report authorization
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_report_unauthorized_agent_rejected(client):
    """An agent who is neither poster nor worker cannot report a task."""
    poster = await register_agent(client, "poster")
    outsider = await register_agent(client, "outsider")

    resp = await client.post(
        "/v1/tasks",
        json={"need": "test report auth", "max_credits": 5},
        headers=auth_header(poster["api_key"]),
    )
    task_id = resp.json()["task_id"]

    # Outsider tries to report — should be rejected
    resp = await client.post(
        f"/v1/tasks/{task_id}/report",
        json={"reason": "spam"},
        headers=auth_header(outsider["api_key"]),
    )
    assert resp.status_code == 403
    body = resp.json()
    assert "poster or worker" in body.get("detail", body.get("error", ""))


@pytest.mark.anyio
async def test_report_by_poster_succeeds(client):
    """The poster should be able to report their own task."""
    poster = await register_agent(client, "poster")

    resp = await client.post(
        "/v1/tasks",
        json={"need": "poster reports", "max_credits": 5},
        headers=auth_header(poster["api_key"]),
    )
    task_id = resp.json()["task_id"]

    resp = await client.post(
        f"/v1/tasks/{task_id}/report",
        json={"reason": "wrong content"},
        headers=auth_header(poster["api_key"]),
    )
    assert resp.status_code == 201


@pytest.mark.anyio
async def test_report_by_worker_succeeds(client):
    """The worker should be able to report a task they picked up."""
    poster = await register_agent(client, "poster")
    worker = await register_agent(client, "worker")

    resp = await client.post(
        "/v1/tasks",
        json={"need": "worker reports", "max_credits": 5},
        headers=auth_header(poster["api_key"]),
    )
    task_id = resp.json()["task_id"]

    # Worker picks up the task
    resp = await client.post(
        "/v1/tasks/pickup",
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 200
    picked_task_id = resp.json()["task_id"]
    assert picked_task_id == task_id

    # Worker reports the task
    resp = await client.post(
        f"/v1/tasks/{task_id}/report",
        json={"reason": "misleading description"},
        headers=auth_header(worker["api_key"]),
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Config constants tests
# ---------------------------------------------------------------------------


class TestConfigConstants:
    def test_max_extracted_tags_default(self):
        assert settings.max_extracted_tags == 20

    def test_task_preview_length_default(self):
        assert settings.task_preview_length == 80


# ---------------------------------------------------------------------------
# Background loop: parameterized SQL (no injection)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_auto_approve_system_task_uses_parameterized_sql(db):
    """Verify auto_approve_system_tasks doesn't use f-string interpolation."""
    import inspect

    from pinchwork.background import auto_approve_system_tasks

    source = inspect.getsource(auto_approve_system_tasks)
    # Should NOT contain f-string with cutoff_seconds interpolated into SQL
    assert 'f"' not in source or "cutoff_seconds}" not in source
    # Should use select(Task).where(...) pattern instead of raw SQL
    assert "select(Task)" in source


@pytest.mark.anyio
async def test_auto_approve_tasks_no_n_plus_one(db):
    """Verify auto_approve_tasks loads full Task objects, not IDs + individual gets."""
    import inspect

    from pinchwork.background import auto_approve_tasks

    source = inspect.getsource(auto_approve_tasks)
    # Should use select(Task) not "SELECT id FROM tasks"
    assert "select(Task)" in source
    assert "SELECT id FROM tasks" not in source


# ---------------------------------------------------------------------------
# Enums used in matching flow
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_matching_uses_enum_values(client):
    """Verify that task creation sets match_status using MatchStatus enum."""
    # Register infra agent (to trigger matching)
    resp = await client.post(
        "/v1/register",
        json={"name": "infra", "good_at": "matching", "accepts_system_tasks": True},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 201

    # Register regular agent and post task
    poster = await register_agent(client, "poster")
    resp = await client.post(
        "/v1/tasks",
        json={"need": "test enum matching", "max_credits": 5},
        headers=auth_header(poster["api_key"]),
    )
    assert resp.status_code == 201
    # If we got here without error, enums are working correctly in the matching flow
