# CLAUDE.md - working notes for AI-assisted sessions

This project is built with an AI coding assistant (Claude Code); see "How it was built"
in the README. This file is the assistant's standing brief: what the project is, how to
run the checks, and which design decisions must not be undone.

**AI Business Operator** - conveyor: Channel -> Queue -> AI brain -> Actions -> Database -> Reports.

## Ground rules
- Quality over speed. Commit after each green stage (ruff, mypy, tests).
- Secrets only in `.env` (git-ignored). Only `.env.example` is committed.
- No personal data, real phone numbers or machine-specific paths in the repository.
- Stop the test containers after checks.

## How to run things
```bash
make test-infra      # Postgres (pgvector) on :55432 and Redis on :56379, local ports only
make test            # pytest with coverage (fails under 85%)
make lint            # ruff check, ruff format --check, mypy --strict
make evals           # AI evals on the offline baseline; PROVIDER=gemini for a real LLM
make test-infra-down
```

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
- `web/` - FastAPI + Jinja2 + HTMX panel; `web/deps.py` has auth, CSRF and the `Throttle`
  used for sign-in attempts and the public form.

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

## Useful facts
- Real-LLM evals take ~25 min for 112 messages (~4 generate calls each). Measured
  2026-11-16 on the Gemini free tier: `gemini-3.5-flash-lite` (default) completed the full
  run; `gemini-3.6-flash` allows only 20 requests/day.
- Do not run `pytest` while an eval is running: both use the test database.
- Screenshots: `make demo` then `make screenshots` (Playwright).
- The dashboard's "median response time" in the demo comes from seeded history
  (timestamps are simulated); do not quote it as a measurement.
- When numbers change (test count, coverage, eval rows), update all three READMEs.

## Open questions
- Armenian texts should be proofread by a native speaker.
