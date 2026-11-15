# Architecture

This document explains how the system is put together and, more importantly, *why*.
Every section names the decision, the alternative that was rejected, and the trade-off.

## 1. The conveyor

```
Channel -> Ingest -> Queue -> Pipeline (AI brain) -> Outbox -> Actions
                \-> PostgreSQL (system of record) <-/            |
                                   |                             v
                             Web panel, reports          customer / owner / sheet
```

A message passes through the same six steps regardless of where it came from:

1. **Channel adapter** turns a Telegram update, an email or a form submission into an
   `InboundMessage` ([`core/message.py`](../src/ai_operator/core/message.py)).
2. **Ingest** stores it in PostgreSQL and only then enqueues a job
   ([`queue/ingest.py`](../src/ai_operator/queue/ingest.py)).
3. **Worker** (arq on Redis) picks the job up.
4. **Pipeline** understands the message with the AI and decides what to do
   ([`pipeline.py`](../src/ai_operator/pipeline.py)).
5. The decision is written in **one database transaction**, including rows for
   everything that must be sent (the outbox).
6. **Delivery jobs** send the reply, notify the owner, sync the spreadsheet.

Processes: `web` (FastAPI), `worker` (arq), `scheduler` (APScheduler + Telegram
polling), plus PostgreSQL and Redis. All three app processes are the same Docker image
with a different command.

## 2. Reliability

### Store first, then queue

*Decision:* a message is committed to PostgreSQL before it is put on the queue.

*Why:* Redis is fast but it is not where data should live. If the message were only in
Redis and Redis lost it, nobody would ever know a customer wrote. With the row in
PostgreSQL, the worst case is a delay.

*The gap this creates* — committed but not enqueued (Redis down, process killed between
the two steps) — is closed by a **sweeper**: a cron job inside the worker that every
minute looks for rows stuck in `received`/`processing`/`pending` for longer than
`STUCK_AFTER_SECONDS` and enqueues them again
([`queue/tasks.py::sweep`](../src/ai_operator/queue/tasks.py)). The sweeper lives in the
worker, not the scheduler, so it runs even if the scheduler is down.

### Idempotency everywhere

At-least-once delivery means every step can run twice. Each one is safe to repeat:

| Step | What makes a repeat harmless |
|---|---|
| Ingest | unique `(channel, external_id)`; a duplicate insert is a no-op |
| Enqueue | arq job id = `job:row_id`; the same job cannot be queued twice |
| Pipeline | a message in a final status is skipped; the final write re-checks under a row lock |
| Reply | `outbound_messages.dedupe_key = msg:<id>:ai_reply` is unique |
| Invoice | one invoice per lead (unique `lead_id`) |
| Owner notification | unique `dedupe_key` such as `handoff:<message id>` |
| Spreadsheet | upsert by lead id, never append blindly |

### Transactional outbox

*Decision:* the pipeline never calls Telegram, SMTP or Google directly. It writes
"reply to send", "notification to deliver", "lead to sync" rows in the same transaction
as the decision; separate jobs deliver them.

*Why:* otherwise a crash between "sent the reply" and "saved that we replied" leaves the
database and the outside world disagreeing. With the outbox, the decision and the
intent to act are atomic, and delivery has its own retries.

*Trade-off:* delivery is **at-least-once**. If a process dies after the channel accepted
a message but before the row is marked `sent`, the retry sends it again. A duplicate
reply is a much smaller problem than a lost one; this is stated in the README.

### Retries, backoff, dead letters

Every job retries with exponential backoff and jitter (`base * 2^(attempt-1)`, capped
at five minutes). After `MAX_ATTEMPTS` the job is *buried*: a `dead_letters` row is
written, the owner is notified, and the panel shows a "Retry" button
([`ops.retry_dead_letter`](../src/ai_operator/ops.py)).

A message whose processing failed every attempt is not just marked dead: it also gets a
hand-off of reason `failed`, so a human answers the customer.

### No network calls inside a database transaction

The pipeline is three short phases: *claim* (transaction), *understand* (AI calls, no
transaction), *apply* (transaction). A slow or hanging vendor API therefore never holds
a row lock or a pooled connection.

## 3. The AI brain

### The model describes, code decides

The model is asked for exactly two things, both as validated Pydantic schemas
([`ai/schemas.py`](../src/ai_operator/ai/schemas.py)):

- `Analysis` — category, confidence, language, extracted booking fields, an injection
  flag;
- `GroundedAnswer` — `found`, the answer text, and the ids of the sources used.

Everything else is deterministic Python in `Pipeline._decide`:

```
injection suspected            -> hand off
complaint                      -> hand off (with an apology)
confidence below the threshold -> hand off
spam                           -> store, never answer
question                       -> retrieve -> answer -> validate -> reply or hand off
booking                        -> merge fields -> complete? confirm + invoice : ask for the rest
other                          -> short generic reply
```

*Why:* business rules in a prompt cannot be unit-tested, drift with every model update
and fail silently. Rules in code are reviewed, tested and identical for every vendor.
Customer-facing text other than the grounded answer comes from translation templates,
so the wording of a booking confirmation or an apology is fixed and predictable.

### Retrieval-augmented answers that refuse to guess

Documents are split into heading-aware chunks, embedded and stored in PostgreSQL with
pgvector (HNSW index, cosine distance). A question is answered in four guarded steps:

1. Retrieve the top chunks; drop those under a relevance floor. Nothing left → hand off.
2. Ask the model to answer **only** from those chunks and to return `found=false`
   otherwise.
3. Validate: `found` must be true, the text non-empty, and the cited source ids must be
   among the chunks *we* retrieved. Anything else → hand off.
4. Append "Source: <document title>" to the reply.

*Why pgvector and not a dedicated vector database:* one database to run, back up and
keep consistent; the knowledge base of a small business is a few hundred chunks.

*Embedding dimension is fixed at 768* for every provider, so switching vendors does not
need a schema migration — only a re-index, which the panel offers when it detects chunks
embedded by a different model.

### Provider abstraction

`AIProvider` exposes four business-level operations: `analyze`, `answer`,
`summarize_prices`, `embed`. `LLMProvider` implements them once on top of two vendor
hooks (`_generate`, `_embed`) and adds the cross-cutting behaviour
([`ai/providers/base.py`](../src/ai_operator/ai/providers/base.py)):

- **pacing** — a rate limiter spaces calls to stay inside free-tier quotas;
- **retries** — only for errors the vendor adapter classified as transient;
- **cache** — responses are cached in Redis by prompt hash. Relative dates are resolved
  against the time the customer wrote (not "now"), which keeps the prompt identical
  across retries, so a retried job does not pay for the same call twice;
- **repair** — an invalid reply gets one chance: the validation error is sent back.

`MockProvider` implements the same four operations with keyword rules and extractive
answers. It exists so that the demo and CI need no keys, and it doubles as the baseline
in the evals.

### Prompt-injection defence

Three independent layers ([`ai/guard.py`](../src/ai_operator/ai/guard.py)):

1. **Before the model:** patterns in three languages ("ignore previous instructions",
   role hijacking, fake system tags). A hit sends the message to a human — the AI does
   not answer it at all.
2. **In the prompt:** customer text is normalised (NFKC), stripped of control characters
   and of our own delimiter tags, length-capped, and wrapped in
   `<customer_message>` tags that the system prompt declares to be data.
3. **After the model:** output is only ever a validated schema; citations must match
   retrieved chunks; a reply that looks like leaked instructions is discarded.

Even a successful injection has little to steal or do: the model has no tools, cannot
set prices (the invoice amount comes from the catalogue) and cannot send anything except
a validated answer.

## 4. Channels

`Channel.send()` is the whole outgoing contract. How a channel *receives* is its own
business: Telegram long-polls (or uses a webhook), email is polled, the form posts HTTP.
All of them end in `ingest()`.

- **Telegram** is a ~100-line client over the Bot API rather than a bot framework: the
  bot has no logic of its own, everything goes to the shared pipeline. Errors never
  include the request URL, because it contains the token.
- **Email** reads with `BODY.PEEK` and marks a message `\Seen` only after it is stored,
  so a crash loses nothing. Auto-replies, bounces and mailing lists are never answered,
  and our own mail carries `Auto-Submitted: auto-replied` — this prevents mail loops.
- **Web form** has no inbox to push to: the reply is simply read from the outbox by a
  page that polls with HTMX.

## 5. Actions

- **Spreadsheet** — PostgreSQL is the source of truth; the sheet is a synced copy
  (Google Sheets, or CSV when no credentials are configured). Customer-supplied text is
  neutralised against spreadsheet formula injection.
- **Invoices** — the invoice is a database row with a stable number; the PDF is derived
  and rendered on demand, so there is no file state to keep consistent.
- **Notifications** — an outbox like any other; without Telegram they are shown in the
  panel.

## 6. Price monitoring

The scraper is polite by construction
([`pricing/scraper.py`](../src/ai_operator/pricing/scraper.py)): it fetches and obeys
`robots.txt`, treats an unreadable `robots.txt` (5xx, network error) as "disallowed",
spaces requests per host by the larger of its own delay and the site's `Crawl-delay`,
and identifies itself honestly. Each competitor page is described by CSS selectors in
the business config. Every reading is stored; a change triggers an AI-written summary.

## 7. Scheduling

APScheduler decides *when*; it only enqueues jobs. The worker does the work, so a report
or a scrape gets the same retries and dead-letter handling as everything else, and a
slow run never blocks the clock.

## 8. Web panel

Server-rendered Jinja2 with a little HTMX (auto-refreshing inbox, live reply on the
form, search-as-you-type in the knowledge base). No build step, no SPA.

Security: signed session cookie, constant-time password check, login throttling, CSRF
token on every state-changing request, security headers, autoescaping everywhere. With
no `ADMIN_PASSWORD` configured, login is disabled rather than open.

Charts are SVG generated in Python ([`web/charts.py`](../src/ai_operator/web/charts.py)):
no JavaScript chart library, testable output, colours from CSS variables so light and
dark themes work, a table view next to each chart.

## 9. Data model

| Table | Purpose |
|---|---|
| `messages` | every inbound message and what the AI concluded about it |
| `outbound_messages` | outbox of replies (AI and human) |
| `leads`, `invoices` | bookings and their invoices |
| `handoffs` | conversations given to a human, with the reason |
| `kb_documents`, `kb_chunks` | knowledge base and its embeddings |
| `price_snapshots`, `price_reports` | competitor price history and summaries |
| `reports`, `notifications` | morning reports and the owner's outbox |
| `events` | audit trail of pipeline decisions (PII-masked) |
| `dead_letters` | jobs that exhausted their retries |
| `settings` | runtime overrides (confidence threshold, language preferences) |

The schema is created by Alembic migrations; the test suite builds its database with
those same migrations.

## 10. Testing strategy

- **End-to-end scenarios** push a message into `ingest()` and assert on what came out:
  the reply text, the lead, the invoice PDF, the CSV row, the notification. They run
  against real PostgreSQL; the queue is drained in-process.
- **One test runs a real arq worker on real Redis** to prove the wiring.
- **Failure paths are tested as carefully as success paths**: retries, dead letters,
  the sweeper, duplicate delivery, a failing AI vendor, a failing spreadsheet.
- **Vendor adapters** are tested with faked SDK clients: request shape, error mapping.
- **Evals** measure AI quality separately from correctness, and gate CI.

## 11. What I would do next at larger scale

- Replace polling the outbox by a sweeper with `LISTEN/NOTIFY` or logical decoding.
- Move from one worker to several, with per-conversation ordering keys.
- Add a second retrieval stage (re-ranking) and hybrid lexical + vector search.
- Per-tenant configuration in the database instead of one YAML file.
