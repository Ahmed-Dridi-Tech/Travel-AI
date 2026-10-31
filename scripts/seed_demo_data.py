"""Populate the database with 5 realistic demo conversations so the dashboard is not empty.

    python scripts/seed_demo_data.py            # (re)seed: removes previous demo data first
    python scripts/seed_demo_data.py --reset    # remove demo data only

Each conversation is run through the real agent (NLP -> tools -> reply), so inquiries and replies are
genuine, then back-dated over the last 3 days. Demo sessions are named "demo-N" and are the only rows
this script ever deletes. With USE_MOCK_NLP=true no ML packages are needed (the Railway setup).
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":  # psycopg async cannot use the default Proactor loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.database.models import Conversation, Inquiry  # noqa: E402
from app.database.repositories import conversation_repo, inquiry_repo, message_repo  # noqa: E402

DEMO_PREFIX = "demo-"

# (session id, customer message, hours ago, intent we expect the agent to find)
DEMO_CONVERSATIONS: list[tuple[str, str, float, str]] = [
    ("demo-1", "météo Djerba", 66, "weather"),
    ("demo-2", "nheb vol Tunis Paris", 50, "flight_inquiry"),
    ("demo-3", "hotel in Sousse for 3 nights", 33, "hotel_inquiry"),
    ("demo-4", "شنو نزور في جربة؟", 17, "destination"),
    ("demo-5", "taux euro dinar", 4, "currency"),
]


async def reset_demo(db: AsyncSession) -> int:
    """Delete demo conversations (messages and inquiries cascade). Returns how many were removed."""
    count = (
        await db.execute(select(func.count()).select_from(Conversation).where(Conversation.session_id.like(f"{DEMO_PREFIX}%")))
    ).scalar_one()
    await db.execute(delete(Conversation).where(Conversation.session_id.like(f"{DEMO_PREFIX}%")))
    await db.flush()
    return int(count)


async def seed_demo(db: AsyncSession, orchestrator: Any, now: datetime | None = None) -> list[dict[str, Any]]:
    """Recreate the 5 demo conversations through `orchestrator`, back-dated from `now`."""
    now = now or datetime.now(timezone.utc)
    await reset_demo(db)  # idempotent: running twice never doubles the data
    report: list[dict[str, Any]] = []
    for session_id, text, hours_ago, expected_intent in DEMO_CONVERSATIONS:
        result = await orchestrator.process(text, session_id, db)
        started = now - timedelta(hours=hours_ago)
        conversation = await conversation_repo.get_by_session_id(db, session_id)
        assert conversation is not None
        conversation.created_at = conversation.updated_at = started
        for offset, message in enumerate(await message_repo.get_by_conversation(db, conversation.id, limit=50)):
            message.created_at = started + timedelta(seconds=20 + 15 * offset)
        for inquiry in await inquiry_repo.list_by_conversation(db, conversation.id):
            inquiry.created_at = inquiry.updated_at = started + timedelta(seconds=40)
        await db.flush()
        report.append({
            "session_id": session_id, "text": text, "language": result.language, "intent": result.intent,
            "expected_intent": expected_intent, "tool": result.tool_used, "tool_success": result.tool_success,
        })
    return report


async def count_demo(db: AsyncSession) -> tuple[int, int]:
    """(demo conversations, inquiries belonging to them)."""
    conversations = (
        await db.execute(select(func.count()).select_from(Conversation).where(Conversation.session_id.like(f"{DEMO_PREFIX}%")))
    ).scalar_one()
    inquiries = (
        await db.execute(
            select(func.count()).select_from(Inquiry).join(Conversation, Inquiry.conversation_id == Conversation.id)
            .where(Conversation.session_id.like(f"{DEMO_PREFIX}%"))
        )
    ).scalar_one()
    return int(conversations), int(inquiries)


async def main(reset_only: bool) -> int:
    from app.database.session import AsyncSessionLocal, engine

    async with AsyncSessionLocal() as db:
        if reset_only:
            removed = await reset_demo(db)
            await db.commit()
            print(f"Removed {removed} demo conversation(s).")
        else:
            from app.agent.orchestrator import AgentOrchestrator

            report = await seed_demo(db, AgentOrchestrator())
            await db.commit()
            for row in report:
                flag = "" if row["intent"] == row["expected_intent"] else f"   <-- expected {row['expected_intent']}"
                print(f"  {row['session_id']}  {row['language']:<8} {row['intent']:<15} tool={row['tool']:<22} {row['text']}{flag}")
            conversations, inquiries = await count_demo(db)
            print(f"Seeded {conversations} demo conversations with {inquiries} inquiries.")
    await engine.dispose()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="remove demo data and exit")
    raise SystemExit(asyncio.run(main(parser.parse_args().reset)))
