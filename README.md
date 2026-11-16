**English** · [Русский](README.ru.md) · [Հայերեն](README.hy.md)

# AI Business Operator

[![CI](https://github.com/vardanyanvahan123321/ai-business-operator/actions/workflows/ci.yml/badge.svg)](https://github.com/vardanyanvahan123321/ai-business-operator/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Coverage 94%](https://img.shields.io/badge/coverage-94%25-brightgreen)
![Type-checked: mypy strict](https://img.shields.io/badge/mypy-strict-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**An AI operator that takes over the daily routine of a small business:** it answers
customers in Telegram, email and a web form, books appointments, issues PDF invoices,
writes bookings into a spreadsheet, watches competitor prices and sends the owner a
morning report — and hands the conversation to a human whenever it is not sure.

One conveyor, the same for every channel:

**Channel → Queue → AI brain → Actions → Database → Reports**

![Product tour](docs/screenshots/tour.gif)

> The demo company, "Demo Auto Service Yerevan", is fictional. Everything that
> describes the business lives in [one YAML file](config/business.yaml), so the system
> can be pointed at a dental clinic or a repair shop without touching the code.

## Try it in one command

```bash
git clone https://github.com/vardanyanvahan123321/ai-business-operator.git
cd ai-business-operator
make demo            # or: docker compose up --build -d --wait
```

No API keys, no accounts. The demo runs on a built-in rule-based "mock" AI and comes
with two weeks of history that was produced by the real pipeline.

| What | Where |
|---|---|
| Admin panel (password `demo`) | http://localhost:8088/panel |
| Customer request form | http://localhost:8088/request |
| Local mail inbox (nothing leaves your machine) | http://localhost:8025 |
| Fictional competitor site that gets scraped | http://localhost:8090/autofix/prices |

Then run `make demo-traffic`: three customer emails (a price question, a booking, a
complaint) arrive, and you can watch each one get classified, answered or handed off.
`make demo-down` stops everything.

## The problem it solves

A small service business loses money in a boring way: messages arrive in three places,
the owner answers the same ten questions all day, a booking from Telegram never reaches
the spreadsheet, and nobody notices that a competitor dropped a price last week.

| Routine | Before | With the operator |
|---|---|---|
| "How much is…?", "Are you open on Sunday?" | Someone types the answer by hand, hours later | Answered in seconds, from the company's own documents, with the source named |
| Booking a visit | Back-and-forth, then copied to a spreadsheet | Details extracted, missing ones asked for, booking saved, invoice sent |
| A complaint | Lost among other messages | Recognised, never auto-answered, owner alerted immediately |
| Competitor prices | Checked "when there is time" | Checked daily, changes summarised in plain language |
| "What happened yesterday?" | Scrolling through chats | One morning report |

More detail and an honest estimate of the time saved: [docs/CASE_STUDY.md](docs/CASE_STUDY.md).

## What makes it trustworthy

This is the part I care about most: an assistant that talks to customers must be
predictable.

- **The AI describes, plain code decides.** The model only fills validated schemas
  (category, confidence, extracted fields, a grounded answer). Every business rule —
  when to hand off, when a booking is complete, what price goes on the invoice — is
  ordinary, unit-tested Python in [`pipeline.py`](src/ai_operator/pipeline.py).
- **It does not make things up.** Answers come only from the knowledge base and name
  their source. If the documents do not contain the answer, the customer is told so and
  a human takes over.
- **A booking is a real slot, not a guess.** A date in the past is refused; a date
  without a year ("20.10") means its next occurrence, and the customer confirms it
  before anything is booked. Working hours and service duration come from the config,
  a taken slot is never booked twice (also when two customers write at the same
  second), and the nearest free times are offered instead. A request that does not
  clearly match one service ("fix my car after an accident") gets a clarifying question
  and a suggestion to start with diagnostics — never a guessed service or its price.
  All three rules were added after I tested the live Telegram bot and it failed them;
  each has regression tests in [`tests/test_booking_rules.py`](tests/test_booking_rules.py).
- **Human in the loop.** Complaints, low confidence, unanswerable questions and
  suspicious messages go to a person, with a notification to the owner. The confidence
  threshold is adjustable in the panel.
- **Nothing is lost.** A message is stored in PostgreSQL before anything else happens.
  Jobs retry with exponential backoff; what still fails lands in a dead-letter queue
  with a "retry" button; a sweeper re-queues anything whose job went missing.
- **Nothing happens twice.** Channels redeliver, workers crash, jobs are retried —
  idempotency keys make sure a customer never gets two replies or two invoices.
- **Prompt-injection defence in three layers**: pattern detection, untrusted text kept
  inside delimiters, and output that is never trusted (citations must point at chunks
  the system itself retrieved).
- **Personal data is masked** in logs and in the audit trail.
- **The price on an invoice comes from the catalogue**, never from the customer's text.

## Features

**Channels** — Telegram bot, email (IMAP + SMTP), web form. Each is one adapter class
that converts to a single message format; adding WhatsApp means writing one more class.

**AI brain** — classification (question / booking / complaint / spam / other),
retrieval-augmented answers on PostgreSQL + pgvector, extraction of booking data into a
strict Pydantic schema, confidence scoring, multi-message bookings ("please send your
phone" → the next message completes the same booking), slot checks against working
hours and existing bookings.

**Actions** — bookings to Google Sheets (CSV fallback), PDF invoices in the customer's
language, owner notifications in Telegram.

**Competitor price monitoring** — a scheduled scraper that obeys `robots.txt` and
crawl delays, stores every reading and has the AI summarise the changes.

**Morning report** — yesterday's messages and bookings, today's visits, what needs
attention, competitor moves.

**Web panel** — inbox, bookings, hand-off queue with reply box, knowledge-base editor
with a retrieval playground, price comparison, logs with the dead-letter queue,
statistics, settings.

**Three languages** — English, Russian and Armenian everywhere: bot, customer replies
(in the customer's own language), panel, invoices.

**Switch the AI vendor with one line** — `AI_PROVIDER=gemini | anthropic | openai | mock`.
Gemini's free tier is the default; the app paces, retries and caches its calls to live
within small quotas.

| | |
|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![A booking handled end to end](docs/screenshots/message-booking.png) |
| ![Hand-off queue](docs/screenshots/handoffs.png) | ![Competitor prices](docs/screenshots/prices.png) |
| ![Knowledge base search](docs/screenshots/knowledge-search.png) | ![Customer form with a live reply](docs/screenshots/form-reply.png) |

The same panel in [Russian](docs/screenshots/dashboard-ru.png),
[Armenian](docs/screenshots/dashboard-hy.png) and a
[dark theme](docs/screenshots/dashboard-dark.png).

## Architecture

```mermaid
flowchart LR
    subgraph Channels
        TG[Telegram bot]
        EM[Email<br/>IMAP / SMTP]
        WF[Web form]
    end

    TG & EM & WF -->|unified message| ING[Ingest<br/>store first, idempotent]
    ING --> PG[(PostgreSQL<br/>+ pgvector)]
    ING --> Q[[Redis queue<br/>arq]]

    Q --> W[Worker]
    W --> BRAIN

    subgraph BRAIN[AI brain]
        direction TB
        G[Injection guard] --> A[Classify + extract]
        A --> R[Retrieve from<br/>knowledge base]
        R --> ANS[Grounded answer]
        ANS --> D{Decision rules<br/>plain Python}
    end

    D -->|confident| OUT[Outbox:<br/>reply, booking, invoice]
    D -->|complaint · unsure · no answer| H[Hand-off to a human]

    OUT --> PG
    H --> PG
    OUT --> ACT[Sheets / CSV · PDF invoice · Channel reply]
    H --> OWN[Owner notification]

    SCH[APScheduler] -->|enqueue| Q
    SCH -.-> PR[Price scraper<br/>robots.txt aware]
    SCH -.-> MR[Morning report]

    PG --> PANEL[Web panel<br/>FastAPI + HTMX]
    PANEL -->|human reply · retry| Q
```

Design decisions and their reasons are written up in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quality: tests and AI evals

| Check | Result |
|---|---|
| Automated tests | **247**, including end-to-end pipeline scenarios against real PostgreSQL and a real Redis worker |
| Coverage | **94%** (CI fails below 85%) |
| Static analysis | `ruff` and `mypy --strict`, zero errors |
| CI | lint → tests + evals → Docker build and a smoke test of the one-command demo |

### AI evaluation

[112 test messages](evals/dataset.jsonl) in three languages with expected outcomes are
run through the real pipeline by [`evals/run.py`](evals/run.py):

| Provider | Classification | Extraction (fields) | Answer accuracy | No made-up answers | Injection defence |
|---|---|---|---|---|---|
| `mock` (rules-v1) | 97% (103/106) | 100% (108/108) | 60% (24/40) | 92% (48/52) | 100% (6/6) |
| `gemini` (gemini-3.5-flash-lite, free tier, 2026-11-16) | 100% (106/106) | 98% (106/108) | 100% (40/40) | 100% (52/52) | 100% (6/6) |

How to read this honestly:

- The `mock` row is the **offline baseline**: keyword rules and lexical search, no
  neural model. It is what `make demo` and CI use. I wrote both the rules and the test
  set, so its scores on simple phrasing are flattering; its weak spot is clearly
  visible — it finds a correct answer for only 60% of answerable questions.
- When the baseline cannot answer, it mostly **hands off instead of guessing**: that is
  the 92% in "no made-up answers". The 4 failures are questions where lexical search
  matched the wrong paragraph.
- The `gemini` row is **one full run** of all 112 messages on 16 Nov 2026 with a free
  Google AI Studio key: model `gemini-3.5-flash-lite`, embeddings `gemini-embedding-001`,
  temperature 0, no provider errors. Raw output: [`evals/results/gemini.json`](evals/results/gemini.json).
  It has two misses. A message signed "…, Ани": the model took the signature as the
  customer's name and my test set expected no name — a debatable label that I left as it
  was. And "I need to get my engine looked at and repaired": the model was not sure this
  means the "Engine repair" service, so the system asked the customer instead of booking.
  That second miss is the price of the "do not guess the service" rule above, and I
  prefer it to a wrong booking with a wrong price.
- A near-perfect score says the test set is **easy for a modern LLM**, not that the system
  is never wrong: I wrote both the prompts and the 112 messages, and it is a single run.
  Real customer traffic will be messier; that is what the confidence threshold and the
  human hand-off are for.
- Reproduce with `make evals PROVIDER=gemini` (about 25 minutes with the built-in pacing).
  Free-tier quotas differ a lot between models: on the day of the run `gemini-3.6-flash`
  allowed only 20 requests per day, which is why the lighter model is the default. If the
  provider stops answering mid-run, the runner aborts instead of reporting quota errors
  as model mistakes.

"No made-up answers" counts, over all questions, the cases where the system did *not*
send an unsupported answer: a correct sourced answer and an honest hand-off both pass.

## Stack

Python 3.12 · FastAPI · Jinja2 + HTMX · SQLAlchemy 2 (async) + Alembic · PostgreSQL 17 +
pgvector · Redis + arq · APScheduler · Pydantic 2 · WeasyPrint · gspread · httpx ·
BeautifulSoup · structlog · pytest · ruff · mypy · Docker Compose · GitHub Actions.
AI: Google Gemini, Anthropic Claude, OpenAI — or none at all.

## Project structure

```
src/ai_operator/
  config.py            settings (.env) and business profile (YAML)
  core/                unified message format, i18n, PII masking, logging
  channels/            telegram, email, web form -> one message format
  queue/               ingest (store first), worker jobs, retries, sweeper
  pipeline.py          claim -> understand (AI) -> decide (code) -> apply
  ai/                  providers, RAG, injection guard, schemas, prompts
  actions/             spreadsheet sync, PDF invoices, audit trail
  pricing/             polite scraper and change reports
  reports/             morning report
  web/                 public form, admin panel, SVG charts
  locales/             en / ru / hy
config/business.yaml   the company: services, prices, competitors, schedule
data/knowledge/        the knowledge base (Markdown, three languages)
evals/                 AI evaluation dataset and runner
tests/                 unit, integration and end-to-end tests
demo/competitor_site/  a fictional competitor for the scraping demo
docs/                  architecture, case study, screenshots
```

## Use it for your own business

1. **Describe the business** in [`config/business.yaml`](config/business.yaml): name,
   hours, services with prices and the words customers use for them, competitors'
   price pages with CSS selectors.
2. **Write the knowledge base**: Markdown files in `data/knowledge/<lang>/`, or upload
   and edit documents in the panel.
3. **Copy `.env.example` to `.env`** and set what you need:
   - `AI_PROVIDER` and its key (a free [Google AI Studio](https://aistudio.google.com/apikey) key works);
   - `TELEGRAM_BOT_TOKEN` from @BotFather and `TELEGRAM_OWNER_CHAT_ID`;
     the owner's chat receives hand-off alerts, new-booking notices and the morning
     report. When you try the bot yourself, you are both the customer and the owner,
     so those owner messages arrive in the same chat as the customer replies — in real
     use they go to the owner's own account;
   - IMAP/SMTP credentials for email;
   - `GOOGLE_SHEETS_ID` and a service-account key for Sheets;
   - `ADMIN_PASSWORD`, `SESSION_SECRET`, and `DEMO_MODE=false`.
4. `docker compose up --build -d`.

Development:

```bash
uv sync                 # install dependencies
make test-infra         # Postgres + Redis for tests
make test               # tests with coverage
make lint               # ruff + mypy
make evals              # AI evals on the offline baseline
```

## Limitations

Stated plainly, so nobody finds them the hard way:

- Delivery is **at-least-once**: if a process dies between sending a reply and recording
  that it was sent, the reply can be sent twice.
- The `mock` provider is for demos and tests, not for production conversations.
- With Anthropic (no embeddings API) retrieval falls back to local lexical embeddings.
- The panel has a single administrator password; there are no user roles.
- The Google Sheets sink and the real LLM providers are covered by tests with faked
  clients; they have not been exercised against live accounts in CI.
- The Armenian texts were written with AI assistance and deserve a native proofread
  before real use.

## Roadmap

- WhatsApp Business and Instagram channels
- Sync with an external calendar (Google Calendar); slot checks today use the system's
  own bookings
- Voice messages (speech-to-text) in Telegram
- User roles in the panel and an audit log of operator actions
- Published eval results for several LLMs, and an LLM-judged answer-quality metric

## How it was built

I, **Vahan Vardanyan**, designed this system, set its requirements and quality bar,
and tested it. The code was written with an AI coding assistant (Claude Code).

I say this openly because it is the point: AI automation is what this project is
about, and it is also how I work. My part was deciding what to build and what "good"
means — the store-first conveyor, human hand-off, the rule that the model never decides
on its own, the evals — and checking that the result actually does it.

## License

[MIT](LICENSE)
