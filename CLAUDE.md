# CLAUDE.md - working notes for AI-assisted sessions

Portfolio project of Vahan Vardanyan: **AI Business Operator**.
Conveyor: Channel -> Queue -> AI brain -> Actions -> Database -> Reports.
Talk to the user in Russian, briefly, one step at a time. Quality over speed.

## Hard constraints of the user's PC
- No `sudo`, no system upgrade. Save RAM; stop Docker containers after checks.
- GitHub CLI: `~/.local/bin/gh` (logged in as `vardanyanvahan123321`).
- Secrets only in `.env` (git-ignored). Only `.env.example` is committed.
- `docker compose` plugin is installed in `~/.docker/cli-plugins`; the user ran `make demo` themself.

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
- `scheduling.py` - pure booking-calendar rules (past dates, working hours, free slots).
- `ai/` - providers (gemini/anthropic/openai/mock), RAG on pgvector, guard, rules for mock.
- `actions/` - Google Sheets / CSV sink, PDF invoices (WeasyPrint), audit + owner notifications.
- `pricing/` - polite scraper (robots.txt, delays) + change report with AI summary.
- `reports/morning.py`, `scheduler.py` (APScheduler only enqueues jobs).
- `web/` - FastAPI + Jinja2 + HTMX panel.

## Key decisions (keep them)
- The AI only fills validated schemas; all business rules are plain Python in `pipeline.py`.
- Price on an invoice always comes from the catalogue, never from the customer's text.
- Booking rules live in `scheduling.py` + `Pipeline._book`: no past dates, a date without a
  year is confirmed, slots are checked against `schedule` in `business.yaml` and existing
  leads (advisory lock in `_apply`), an uncertain service match (`service_certain=false`)
  is clarified, never priced. Regression tests: `tests/test_booking_rules.py`.
- Prompt injection: pattern check -> handoff; untrusted text in delimiters; cited sources
  must be among retrieved chunks.
- Embeddings are 768-dim for every provider; `mock`/`anthropic` use local hashing embeddings.
- Anthropic default model: `claude-opus-5-5` with server-side refusal fallback.
- Tests run against real Postgres; schema is created by Alembic migrations.

## Progress
- [x] Stages 0-9: core pipeline, AI providers, channels, actions, pricing, reports, web panel,
      evals (112 messages), 247 tests / 94% coverage, Docker demo stack, CI workflow,
      screenshots + tour GIF, docs (README x3, ARCHITECTURE, CASE_STUDY, EXPLAIN_RU), MIT.
- [ ] Stage 10: published to GitHub; watch CI (the `docker` job is the only place where
      `docker compose` itself is exercised - it is not installed locally).

## Useful facts
- Evals: `make evals` (mock). Real LLM row: `make evals PROVIDER=gemini` (key in `.env`),
  ~25 min for 112 messages (~4 generate calls each). Measured 2026-11-16 on the free tier:
  `gemini-3.5-flash-lite` (default) completed the full run; `gemini-3.6-flash` allows only
  20 requests/day; `gemini-3.8-flash` returned 504; `gemini-flash-latest` no longer exists.
- Do not run `pytest` while an eval is running: both use the test database.
- Demo without compose (how it was verified locally): build the image, create a docker
  network, run postgres/redis/mailpit/competitor, then the image with an --env-file as
  migrate / web / worker / scheduler. Note: in zsh use `${=VAR}` to word-split.
- `pkill -f "arq ai_operator"` kills its own shell; use `pkill -f "bin/[a]rq"`.
- Screenshots: `make demo` then `make screenshots` (Playwright chromium in ~/.cache).
- The dashboard's "median response time" in the demo comes from seeded history
  (timestamps are simulated); do not quote it as a measurement.

## Open questions for the user
- Armenian texts should be proofread by a native speaker.
