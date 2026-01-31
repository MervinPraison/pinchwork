"""Background tasks: expire old tasks, auto-approve delivered tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import select

from pinchwork.config import settings
from pinchwork.db_models import MatchStatus, SystemTaskType, Task, TaskMatch, TaskStatus
from pinchwork.events import Event, event_bus
from pinchwork.services.credits import refund
from pinchwork.services.tasks import (
    cleanup_task_event,
    finalize_system_task_approval,
    finalize_task_approval,
)

logger = logging.getLogger("pinchwork.background")


async def expire_tasks(session: AsyncSession) -> int:
    now = datetime.now(UTC).isoformat()
    result = await session.execute(
        select(Task).where(Task.status == TaskStatus.posted, Task.expires_at < now)
    )
    tasks = result.scalars().all()

    for task in tasks:
        task.status = TaskStatus.expired
        session.add(task)
        await refund(session, task.id, task.poster_id, task.max_credits)
        logger.info(
            "Expired task %s, refunded %d credits to %s", task.id, task.max_credits, task.poster_id
        )

    if tasks:
        await session.commit()
        for task in tasks:
            cleanup_task_event(task.id)
            event_bus.publish(task.poster_id, Event(type="task_expired", task_id=task.id))
    return len(tasks)


async def auto_approve_tasks(session: AsyncSession) -> int:
    if settings.disable_auto_approve:
        return 0

    cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    result = await session.execute(
        select(Task).where(
            Task.status == TaskStatus.delivered,
            Task.delivered_at < cutoff,
        )
    )
    tasks = result.scalars().all()

    for task in tasks:
        await finalize_task_approval(session, task, settings.platform_fee_percent)
        logger.info(
            "Auto-approved task %s, paid %d to %s",
            task.id,
            task.credits_charged or 0,
            task.worker_id,
        )

    if tasks:
        await session.commit()
        # SSE: notify workers their tasks were auto-approved
        for task in tasks:
            cleanup_task_event(task.id)
            if task.worker_id:
                event_bus.publish(task.worker_id, Event(type="task_approved", task_id=task.id))
    return len(tasks)


async def expire_matching(session: AsyncSession) -> int:
    """Expire pending matches that passed their deadline, fall back to broadcast."""
    now = datetime.now(UTC).isoformat()
    result = await session.execute(
        select(Task).where(
            Task.match_status == MatchStatus.pending,
            Task.match_deadline < now,
            Task.status == TaskStatus.posted,
        )
    )
    tasks = result.scalars().all()

    for task in tasks:
        task.match_status = MatchStatus.broadcast
        session.add(task)

        # Cancel the associated system task if still posted
        sys_result = await session.execute(
            select(Task).where(
                Task.is_system == True,  # noqa: E712
                Task.system_task_type == SystemTaskType.match_agents,
                Task.parent_task_id == task.id,
                Task.status == TaskStatus.posted,
            )
        )
        sys_task = sys_result.scalar_one_or_none()
        if sys_task:
            sys_task.status = TaskStatus.cancelled
            session.add(sys_task)

        logger.info("Match expired for task %s, fell back to broadcast", task.id)

    if tasks:
        await session.commit()
    return len(tasks)


async def expire_rejection_grace(session: AsyncSession) -> int:
    """Reset tasks whose rejection grace period has expired back to posted."""
    now = datetime.now(UTC).isoformat()
    result = await session.execute(
        select(Task).where(
            Task.status == TaskStatus.claimed,
            Task.rejection_grace_deadline != None,  # noqa: E711
            Task.rejection_grace_deadline < now,
        )
    )
    tasks = result.scalars().all()

    for task in tasks:
        expired_worker_id = task.worker_id
        task.status = TaskStatus.posted
        task.worker_id = None
        task.claimed_at = None
        task.rejection_grace_deadline = None
        task.expires_at = datetime.now(UTC) + timedelta(hours=settings.task_expire_hours)
        task.match_status = MatchStatus.broadcast
        session.add(task)

        logger.info(
            "Rejection grace expired for task %s, reset to posted",
            task.id,
        )

        if expired_worker_id:
            event_bus.publish(
                expired_worker_id,
                Event(type="rejection_grace_expired", task_id=task.id),
            )

    if tasks:
        await session.commit()
    return len(tasks)


async def auto_approve_system_tasks(session: AsyncSession) -> int:
    """Auto-approve delivered system tasks after a short window."""
    cutoff_seconds = settings.system_task_auto_approve_seconds
    cutoff = (datetime.now(UTC) - timedelta(seconds=cutoff_seconds)).isoformat()
    result = await session.execute(
        select(Task).where(
            Task.is_system == True,  # noqa: E712
            Task.status == TaskStatus.delivered,
            Task.delivered_at < cutoff,
        )
    )
    tasks = result.scalars().all()

    for task in tasks:
        await finalize_system_task_approval(session, task)
        logger.info(
            "Auto-approved system task %s, paid %d to %s",
            task.id,
            task.credits_charged or 0,
            task.worker_id,
        )

    if tasks:
        await session.commit()
    return len(tasks)


async def expire_deadlines(session: AsyncSession) -> int:
    """Handle tasks that have passed their deadline.

    - Claimed tasks past deadline: reset to posted (worker ran out of time).
    - Posted tasks past deadline: expire and refund.

    We handle claimed and posted tasks separately with commits in between
    to prevent a claimed→posted reset from immediately being expired.
    """
    now = datetime.now(UTC).isoformat()
    count = 0

    # Claimed tasks past deadline → reset to posted
    result = await session.execute(
        select(Task).where(
            Task.status == TaskStatus.claimed,
            Task.deadline != None,  # noqa: E711
            Task.deadline < now,
        )
    )
    claimed_tasks = result.scalars().all()

    for task in claimed_tasks:
        expired_worker_id = task.worker_id
        task.status = TaskStatus.posted
        task.worker_id = None
        task.claimed_at = None
        task.deadline = None  # Clear deadline after reset so it doesn't immediately expire
        task.expires_at = datetime.now(UTC) + timedelta(hours=settings.task_expire_hours)
        task.match_status = MatchStatus.broadcast
        session.add(task)
        count += 1

        logger.info("Deadline expired for claimed task %s, reset to posted", task.id)
        if expired_worker_id:
            event_bus.publish(
                expired_worker_id,
                Event(type="deadline_expired", task_id=task.id),
            )

    if claimed_tasks:
        await session.commit()

    # Posted tasks past deadline → expire and refund
    result2 = await session.execute(
        select(Task).where(
            Task.status == TaskStatus.posted,
            Task.deadline != None,  # noqa: E711
            Task.deadline < now,
        )
    )
    posted_tasks = result2.scalars().all()

    for task in posted_tasks:
        # Collect matched agent IDs for notification
        match_result = await session.execute(
            select(TaskMatch.agent_id).where(TaskMatch.task_id == task.id)
        )
        matched_agent_ids = [row[0] for row in match_result.fetchall()]

        task.status = TaskStatus.expired
        session.add(task)
        await refund(session, task.id, task.poster_id, task.max_credits)
        count += 1

        logger.info("Deadline expired for posted task %s, expired and refunded", task.id)
        event_bus.publish(task.poster_id, Event(type="task_expired", task_id=task.id))
        event_bus.publish_many(matched_agent_ids, Event(type="task_expired", task_id=task.id))

    if posted_tasks:
        await session.commit()
    return count


async def background_loop(session_factory: sessionmaker) -> None:
    """Run background maintenance every 60 seconds."""
    while True:
        try:
            async with session_factory() as session:
                expired = await expire_tasks(session)
                approved = await auto_approve_tasks(session)
                match_expired = await expire_matching(session)
                sys_approved = await auto_approve_system_tasks(session)
                grace_expired = await expire_rejection_grace(session)
                deadline_expired = await expire_deadlines(session)
                any_work = (
                    expired or approved or match_expired
                    or sys_approved or grace_expired or deadline_expired
                )
                if any_work:
                    logger.info(
                        "BG: exp=%d, app=%d, mexp=%d, sys=%d, gexp=%d, dl=%d",
                        expired,
                        approved,
                        match_expired,
                        sys_approved,
                        grace_expired,
                        deadline_expired,
                    )
        except Exception:
            logger.exception("Background task error")
        await asyncio.sleep(60)
