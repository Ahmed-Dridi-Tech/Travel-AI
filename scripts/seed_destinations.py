"""Seed/refresh the destinations table from data/destinations_gazetteer.json (idempotent upsert)."""

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":  # psycopg async cannot use the default Proactor loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.database.repositories import destination_repo  # noqa: E402
from app.database.session import AsyncSessionLocal, engine  # noqa: E402

GAZETTEER = ROOT / "data" / "destinations_gazetteer.json"
DESCRIPTIONS = ROOT / "data" / "destinations.json"  # curated content: slug -> description_fr / description_en
_ARABIC = re.compile("[؀-ۿ]")


def to_row(entry: dict, descriptions: dict | None = None) -> dict:  # type: ignore[type-arg]
    """Map a gazetteer entry (+ optional curated descriptions) to Destination columns.
    Destinations without a curated description keep the column untouched: the assistant may only
    state facts that were written into the catalogue."""
    latin = next(n for n in entry["names"] if not _ARABIC.search(n))
    arabic = next(n for n in entry["names"] if _ARABIC.search(n))
    name = latin.title()
    texts = (descriptions or {}).get(entry["slug"], {})
    return {
        **{k: texts[k] for k in ("description_fr", "description_en") if texts.get(k)},
        "slug": entry["slug"],
        "name_fr": name,
        "name_en": name,
        "name_ar": arabic,
        "region": entry["region"],
        "tags": entry["tags"],
        "best_months": entry["best_months"],
        "avg_budget_tnd": entry.get("avg_budget_tnd"),
        "active": True,
    }


async def main() -> int:
    entries = [e for e in json.loads(GAZETTEER.read_text(encoding="utf-8"))["destinations"] if e.get("tunisian", True)]
    descriptions = json.loads(DESCRIPTIONS.read_text(encoding="utf-8")) if DESCRIPTIONS.exists() else {}
    created = updated = 0
    async with AsyncSessionLocal() as db:
        for entry in entries:
            row = to_row(entry, descriptions)
            existing = await destination_repo.get_by_slug(db, row["slug"])
            if existing is None:
                await destination_repo.create(db, **row)
                created += 1
            else:
                await destination_repo.update(db, existing, **{k: v for k, v in row.items() if k != "slug"})
                updated += 1
        await db.commit()
    await engine.dispose()
    print(f"Destinations: {created} created, {updated} updated ({len(entries)} total, {len(descriptions)} with descriptions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
