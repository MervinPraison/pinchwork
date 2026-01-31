"""Credit escrow and ledger service — SQLModel."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pinchwork.config import settings
from pinchwork.db_models import Agent, CreditLedger, Task, TaskStatus
from pinchwork.ids import ledger_id


async def _update_credits(session: AsyncSession, agent_id: str, amount: int) -> None:
    """Atomically adjust an agent's credit balance."""
    await session.execute(
        text("UPDATE agents SET credits = credits + :amount WHERE id = :id"),
        {"amount": amount, "id": agent_id},
    )


async def increment_tasks_completed(session: AsyncSession, agent_id: str) -> None:
    """Atomically increment an agent's tasks_completed counter."""
    await session.execute(
        text("UPDATE agents SET tasks_completed = tasks_completed + 1 WHERE id = :id"),
        {"id": agent_id},
    )


async def increment_tasks_posted(session: AsyncSession, agent_id: str) -> None:
    """Atomically increment an agent's tasks_posted counter."""
    await session.execute(
        text("UPDATE agents SET tasks_posted = tasks_posted + 1 WHERE id = :id"),
        {"id": agent_id},
    )


async def record_credit(
    session: AsyncSession,
    agent_id: str,
    amount: int,
    reason: str,
    task_id: str | None = None,
) -> None:
    entry = CreditLedger(
        id=ledger_id(), agent_id=agent_id, amount=amount, reason=reason, task_id=task_id
    )
    session.add(entry)


async def escrow(
    session: AsyncSession, poster_id: str, task_id: str, amount: int, *, is_system: bool = False
) -> None:
    """Atomic escrow: single UPDATE with balance check to prevent race conditions.

    System tasks skip escrow entirely (platform agent has infinite credits).
    """
    if is_system:
        return

    result = await session.execute(
        text("UPDATE agents SET credits = credits - :amount WHERE id = :id AND credits >= :amount"),
        {"amount": amount, "id": poster_id},
    )
    if result.rowcount == 0:
        # Fetch current balance for error message
        agent = await session.get(Agent, poster_id)
        have = agent.credits if agent else 0
        raise HTTPException(
            status_code=402, detail=f"Insufficient credits. Have {have}, need {amount}"
        )

    await record_credit(session, poster_id, -amount, "escrow", task_id)


async def release_to_worker(
    session: AsyncSession, task_id: str, worker_id: str, amount: int
) -> None:
    await _update_credits(session, worker_id, amount)
    await record_credit(session, worker_id, amount, "payment", task_id)


async def refund(session: AsyncSession, task_id: str, poster_id: str, amount: int) -> None:
    await _update_credits(session, poster_id, amount)
    await record_credit(session, poster_id, amount, "refund", task_id)


async def get_balance(session: AsyncSession, agent_id: str) -> int:
    agent = await session.get(Agent, agent_id)
    return agent.credits if agent else 0


async def release_to_worker_with_fee(
    session: AsyncSession,
    task_id: str,
    worker_id: str,
    poster_id: str,
    amount: int,
    fee_percent: float,
) -> None:
    """Release escrowed credits to worker, taking a platform fee."""
    if fee_percent == 0:
        await release_to_worker(session, task_id, worker_id, amount)
        return

    fee = int(amount * fee_percent / 100)
    worker_amount = amount - fee

    # Credit worker
    await _update_credits(session, worker_id, worker_amount)
    await record_credit(session, worker_id, worker_amount, "payment", task_id)

    # Credit platform agent with fee
    if fee > 0:
        await _update_credits(session, settings.platform_agent_id, fee)
        await record_credit(session, settings.platform_agent_id, fee, "platform_fee", task_id)


async def grant_credits(session: AsyncSession, agent_id: str, amount: int, reason: str) -> None:
    """Grant credits to an agent (admin operation)."""
    agent = await session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    await _update_credits(session, agent_id, amount)
    await record_credit(session, agent_id, amount, reason)


async def get_escrowed_balance(session: AsyncSession, agent_id: str) -> int:
    """Get total credits currently in escrow for an agent's posted tasks."""
    result = await session.execute(
        select(func.coalesce(func.sum(Task.max_credits), 0)).where(
            Task.poster_id == agent_id,
            Task.status.in_([TaskStatus.posted, TaskStatus.claimed, TaskStatus.delivered]),
        )
    )
    return result.scalar_one()


async def get_ledger(
    session: AsyncSession, agent_id: str, offset: int = 0, limit: int = 50
) -> tuple[list[dict], int]:
    """Return (entries, total_count)."""
    # Total count
    count_result = await session.execute(
        select(func.count()).select_from(CreditLedger).where(CreditLedger.agent_id == agent_id)
    )
    total = count_result.scalar_one()

    result = await session.execute(
        select(CreditLedger)
        .where(CreditLedger.agent_id == agent_id)
        .order_by(CreditLedger.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = result.scalars().all()
    entries = [
        {
            "id": r.id,
            "amount": r.amount,
            "reason": r.reason,
            "task_id": r.task_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    return entries, total


async def get_agent_stats(session: AsyncSession, agent_id: str) -> dict:
    """Aggregate earnings stats for an agent."""
    # Total earned (payments received as worker)
    earned_result = await session.execute(
        select(func.coalesce(func.sum(CreditLedger.amount), 0)).where(
            CreditLedger.agent_id == agent_id,
            CreditLedger.reason == "payment",
        )
    )
    total_earned = earned_result.scalar_one()

    # Total spent (escrow debits as poster)
    spent_result = await session.execute(
        select(func.coalesce(func.sum(func.abs(CreditLedger.amount)), 0)).where(
            CreditLedger.agent_id == agent_id,
            CreditLedger.reason == "escrow",
        )
    )
    total_spent = spent_result.scalar_one()

    # Total fees paid (platform fees deducted from earnings)
    # Fees are paid indirectly: worker receives (amount - fee). We track via platform_fee ledger.
    # For the worker, fees = sum of platform_fee entries on their tasks
    worker_tasks = select(Task.id).where(
        Task.worker_id == agent_id, Task.status == TaskStatus.approved
    )
    fees_result = await session.execute(
        select(func.coalesce(func.sum(CreditLedger.amount), 0)).where(
            CreditLedger.reason == "platform_fee",
            CreditLedger.task_id.in_(worker_tasks),
        )
    )
    total_fees_paid = fees_result.scalar_one()

    # Approval rate as worker
    approved_count_r = await session.execute(
        select(func.count())
        .select_from(Task)
        .where(Task.worker_id == agent_id, Task.status == TaskStatus.approved)
    )
    approved_count = approved_count_r.scalar_one()

    # approval_rate = approved / (approved + delivered still pending)
    delivered_count_r = await session.execute(
        select(func.count())
        .select_from(Task)
        .where(Task.worker_id == agent_id, Task.status == TaskStatus.delivered)
    )
    delivered_count = delivered_count_r.scalar_one()

    total_outcomes = approved_count + delivered_count
    approval_rate = round(approved_count / total_outcomes, 2) if total_outcomes > 0 else None

    # Avg task value
    avg_val_r = await session.execute(
        select(func.avg(Task.credits_charged)).where(
            Task.worker_id == agent_id, Task.status == TaskStatus.approved
        )
    )
    avg_val = avg_val_r.scalar_one()
    avg_task_value = round(float(avg_val), 2) if avg_val is not None else None

    # Tasks by tag
    tag_tasks_r = await session.execute(
        select(Task.tags, Task.credits_charged).where(
            Task.worker_id == agent_id,
            Task.status == TaskStatus.approved,
            Task.tags.isnot(None),
        )
    )
    tag_stats: dict[str, dict] = {}
    for row in tag_tasks_r.fetchall():
        tags_raw, credits = row
        tags = json.loads(tags_raw) if tags_raw else []
        for tag in tags:
            if tag not in tag_stats:
                tag_stats[tag] = {"tag": tag, "count": 0, "earned": 0}
            tag_stats[tag]["count"] += 1
            tag_stats[tag]["earned"] += credits or 0

    tasks_by_tag = sorted(tag_stats.values(), key=lambda x: x["earned"], reverse=True)

    # Recent earnings
    now = datetime.now(UTC)
    for days, key in [(7, "recent_7d_earned"), (30, "recent_30d_earned")]:
        cutoff = now - timedelta(days=days)
        r = await session.execute(
            select(func.coalesce(func.sum(CreditLedger.amount), 0)).where(
                CreditLedger.agent_id == agent_id,
                CreditLedger.reason == "payment",
                CreditLedger.created_at >= cutoff,
            )
        )
        if key == "recent_7d_earned":
            recent_7d = r.scalar_one()
        else:
            recent_30d = r.scalar_one()

    return {
        "total_earned": total_earned,
        "total_spent": total_spent,
        "total_fees_paid": total_fees_paid,
        "approval_rate": approval_rate,
        "avg_task_value": avg_task_value,
        "tasks_by_tag": tasks_by_tag,
        "recent_7d_earned": recent_7d,
        "recent_30d_earned": recent_30d,
    }
