"""Shared utility functions for Pinchwork."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from pinchwork.db_models import TaskStatus


def safe_json_loads(value: str | None) -> Any:
    """Parse a JSON string, returning None on failure."""
    if value is None:
        return None
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        return json.loads(value)
    return None


def status_str(status: TaskStatus | str) -> str:
    """Coerce a TaskStatus enum (or plain string) to its string value."""
    return status.value if isinstance(status, TaskStatus) else status
