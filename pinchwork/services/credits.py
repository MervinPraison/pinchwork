"""Credit escrow and ledger service — SQLModel."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pinchwork.config import settings
from pinchwork.db_models import Agent, CreditLedger, Task, TaskStatus
from pinchwork.ids import ledger_id


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
    await session.execute(
        text("UPDATE agents SET credits = credits + :amount WHERE id = :id"),
        {"amount": amount, "id": worker_id},
    )
    await record_credit(session, worker_id, amount, "payment", task_id)


async def refund(session: AsyncSession, task_id: str, poster_id: str, amount: int) -> None:
    await session.execute(
        text("UPDATE agents SET credits = credits + :amount WHERE id = :id"),
        {"amount": amount, "id": poster_id},
    )
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
    await session.execute(
        text("UPDATE agents SET credits = credits + :amount WHERE id = :id"),
        {"amount": worker_amount, "id": worker_id},
    )
    await record_credit(session, worker_id, worker_amount, "payment", task_id)

    # Credit platform agent with fee
    if fee > 0:
        await session.execute(
            text("UPDATE agents SET credits = credits + :amount WHERE id = :id"),
            {"amount": fee, "id": settings.platform_agent_id},
        )
        await record_credit(session, settings.platform_agent_id, fee, "platform_fee", task_id)


async def grant_credits(
    session: AsyncSession, agent_id: str, amount: int, reason: str
) -> None:
    """Grant credits to an agent (admin operation)."""
    agent = await session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    await session.execute(
        text("UPDATE agents SET credits = credits + :amount WHERE id = :id"),
        {"amount": amount, "id": agent_id},
    )
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
