# travel-ai

Multilingual (FR / EN / MSA / Tunisian Arabic, Arabic + Latin script) conversational assistant for Tunisian travel agencies.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
curl http://localhost:8000/health
pytest
```

## Database

```bash
alembic upgrade head            # apply migrations
pytest                          # integration tests need the migrated DB (skipped otherwise)
```

**Windows:** psycopg async cannot use the default Proactor loop. Run the server with
`uvicorn app.main:app --reload` (reload mode already uses the selector loop) or, without reload,
`uvicorn app.main:app --loop asyncio:SelectorEventLoop`. Linux/Docker need nothing special.

## Run the demo

```powershell
# terminal 1 - API (Windows: selector loop required by psycopg async)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --loop asyncio:SelectorEventLoop
# terminal 2 - web app (chat + agency dashboard)
.\.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
```

`LLM_PROVIDER=mock` (default) needs no key. For Claude, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`
(the account needs credit). `OPENWEATHER_API_KEY` / `EXCHANGE_RATE_API_KEY` enable live weather / rates; without them
the assistant answers with clearly-labelled demo values.

## Deploy on Railway (lean mode, no ML packages)

The free-tier API cannot hold torch / sentence-transformers / spaCy / fastText (2 GB+). On Railway the API runs with
`USE_MOCK_NLP=true`: a keyword-based `MockNLPPipeline` replaces the ML pipeline (entities, dates and flight routes still
come from the real rule-based extractor). The ML pipeline is unchanged locally. `requirements.txt` is the lean, pinned
runtime; heavy packages stay in the `nlp` extra.

> Railway's pricing changes: check whether the trial credit / entry plan fits your budget. Nothing below has been
> deployed yet, so verify each step on the first deploy.

**0. Code on GitHub** (this folder is not a git repository yet): `git init`, commit, push. `.env` and `data/models/` are gitignored.

**1. API service** (repo root): New Project → Deploy from GitHub → add a **PostgreSQL** database. `railway.json` sets the
build (Nixpacks), the start command, `/health` as health check, and runs `alembic upgrade head` before each deploy
(`preDeployCommand`; Railway has no Procfile `release:` phase). Variables:

| Variable | Value |
|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (reference; the app converts `postgresql://` to `postgresql+psycopg://` itself) |
| `USE_MOCK_NLP` | `true` |
| `LLM_PROVIDER` | `mock` (until Anthropic credits; then `anthropic` + `ANTHROPIC_API_KEY`) |
| `OPENWEATHER_API_KEY` / `EXCHANGE_RATE_API_KEY` | your keys (without them replies use clearly-labelled demo values) |
| `ANALYTICS_API_KEY` | a long random string: protects `/api/v1/analytics` (header `X-API-Key`) |

**2. Streamlit service** (same repo, second service): Settings → **Root Directory = `frontend`** and **Config file path =
`/frontend/railway.json`** (Railway does not apply the root directory to the config file, so without this it would run the API's
config). Variables: `API_BASE_URL=https://<api-service-domain>`, `DASHBOARD_PASSWORD=<password>`, and the **same** `ANALYTICS_API_KEY`.

**3. Seed the catalogue and demo data** (from your laptop, pointing at the Railway database's *public* URL):

```powershell
$env:DATABASE_URL = "<Postgres public connection URL>"; $env:USE_MOCK_NLP = "true"
python scripts/seed_destinations.py        # required: the destination tool reads this table
python scripts/seed_demo_data.py           # optional: 5 demo conversations (--reset removes them)
```

**4. Smoke test:** `https://<api>/health` → `"db":"connected"`; send a message in the Streamlit app; open the dashboard.
