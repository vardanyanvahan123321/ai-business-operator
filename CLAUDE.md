# CLAUDE.md - working notes for AI-assisted sessions

Portfolio project of Vahan Vardanyan: **AI Business Operator**.
Conveyor: Channel -> Queue -> AI brain -> Actions -> Database -> Reports.
Talk to the user in Russian, briefly, one step at a time. Quality over speed.

## Hard constraints of the user's PC
- Do NOT touch `~/deep-research`, LM Studio, its processes or supervisor.
- No `sudo`, no system upgrade. Save RAM; stop Docker containers after checks.
- GitHub CLI: `~/.local/bin/gh` (logged in as `vardanyanvahan123321`).
- Secrets only in `.env` (git-ignored). Only `.env.example` is committed.
- `docker compose` plugin is NOT installed and installing it was denied by the
  permission system. Ask the user before trying again (needed to verify `make demo`).

## How to run things
```bash
# test infrastructure (light containers, local ports only)
docker start abo-test-pg abo-test-redis     # or see `make test-infra`
uv run pytest                               # tests need Postgres :55432 and Redis :56379
uv run ruff check . && uv run ruff format --check . && uv run mypy
docker stop abo-test-pg abo-test-redis      # when done
```
Git identity is set locally (noreply address). Commit after each green stage.

## Architecture in one screen
- `src/ai_operator/config.py` - `Settings` (env) + `BusinessConfig` (`config/business.yaml`).
- `core/` - unified message format, i18n (`locales/*.json`, 3 languages), PII masking, logging.
- `channels/` - adapters (telegram, email, webform) -> `InboundMessage`; `Channel.send()`.
- `queue/ingest.py` - store in Postgres first (idempotent), then enqueue.
- `queue/tasks.py` - arq jobs: retries with backoff, dead letters, sweeper.
- `pipeline.py` - claim -> understand (AI) -> decide (code) -> apply (one transaction, outbox).
- `ai/` - providers (gemini/anthropic/openai/mock), RAG on pgvector, guard, rules for mock.
- `actions/` - Google Sheets / CSV sink, PDF invoices (WeasyPrint), audit + owner notifications.
- `pricing/` - polite scraper (robots.txt, delays) + change report with AI summary.
- `reports/morning.py`, `scheduler.py` (APScheduler only enqueues jobs).
- `web/` - FastAPI + Jinja2 + HTMX panel.

## Key decisions (keep them)
- The AI only fills validated schemas; all business rules are plain Python in `pipeline.py`.
- Price on an invoice always comes from the catalogue, never from the customer's text.
- Prompt injection: pattern check -> handoff; untrusted text in delimiters; cited sources
  must be among retrieved chunks.
- Embeddings are 768-dim for every provider; `mock`/`anthropic` use local hashing embeddings.
- Anthropic default model: `claude-opus-5-5` with server-side refusal fallback.
- Tests run against real Postgres; schema is created by Alembic migrations.

## Progress
- [x] Stage 0: scaffold (uv, ruff, mypy, pytest, alembic)
- [x] Stage 1: DB models + migration, message format, queue, pipeline, e2e tests
- [x] Stage 2: AI providers, RAG, guard, mock rules
- [x] Stage 3: channels (telegram, email, webform) - code done, unit tests pending
- [x] Stage 4: actions (sheets/csv, invoices, notifications)
- [x] Stage 5: price monitoring + morning report - code done, tests pending
- [ ] Stage 6: web panel
- [ ] Stage 7: evals, coverage >= 85%, CI
- [ ] Stage 8: demo mode (`make demo`), screenshots
- [ ] Stage 9: docs (README x3, ARCHITECTURE, CASE_STUDY, EXPLAIN_RU)
- [ ] Stage 10: publish to GitHub, verify CI

## Open questions for the user
- docker compose plugin (see above).
- Gemini API key - needed only to fill the "real LLM" column of the evals table.

## Where the last session stopped (2026-11-15, usage limit)
- Done since the progress list above: Stage 6 web panel (30 tests green), `demo.py`
  (seed + traffic), `scripts/screenshots.py`. Dashboard checked visually, looks right.
- Left: Stage 7 (evals ~100 messages, unit tests to >= 85% coverage, CI), rest of Stage 8
  (Dockerfile, compose with mailpit + `demo/competitor_site`, Makefile, `.env.example`,
  final screenshots), Stage 9 docs + MIT LICENSE, Stage 10 publish.
- Small fix pending: dashboard tiles leave the 6th tile alone on a second row.
- Local run: start `abo-test-pg`/`abo-test-redis`, export DATABASE_URL (:55432), REDIS_URL
  (:56379), AI_PROVIDER=mock, DEMO_MODE=true, ADMIN_PASSWORD=demo, then
  `python -m ai_operator.demo seed`, uvicorn `ai_operator.web.app:app_factory --factory`,
  and `arq ai_operator.queue.worker.WorkerSettings`.
