"""Agent-to-agent trust score service."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pinchwork.db_models import AgentTrust
from pinchwork.ids import trust_id as make_trust_id


async def update_trust(
    session: AsyncSession, truster_id: str, trusted_id: str, *, positive: bool
) -> None:
    """Update trust score between two agents.

    Positive: score += (1 - score) * 0.1 (asymptotically approaches 1.0)
    Negative: score *= 0.9 (multiplicative decay toward 0)
    """
    result = await session.execute(
        select(AgentTrust).where(
            AgentTrust.truster_id == truster_id,
            AgentTrust.trusted_id == trusted_id,
        )
    )
    trust = result.scalar_one_or_none()

    if not trust:
        trust = AgentTrust(
            id=make_trust_id(),
            truster_id=truster_id,
            trusted_id=trusted_id,
            score=0.5,
            interactions=0,
        )
        session.add(trust)

    if positive:
        trust.score = trust.score + (1.0 - trust.score) * 0.1
    else:
        trust.score = trust.score * 0.9

    # Clamp to [0, 1]
    trust.score = max(0.0, min(1.0, round(trust.score, 6)))
    trust.interactions += 1
    trust.updated_at = datetime.now(UTC)
    session.add(trust)


async def get_trust_scores(session: AsyncSession, agent_id: str) -> list[dict]:
    """List all trust entries where this agent is the truster."""
    result = await session.execute(
        select(AgentTrust)
        .where(AgentTrust.truster_id == agent_id)
        .order_by(AgentTrust.score.desc())
    )
    return [
        {
            "trusted_id": t.trusted_id,
            "score": t.score,
            "interactions": t.interactions,
        }
        for t in result.scalars().all()
    ]
