# Case study: a car workshop that stops answering the same ten questions

> **Read this first.** "Demo Auto Service Yerevan" is a fictional company and no real
> business has used this system yet. The scenario below is modelled on how small
> service businesses commonly work; all measurements come from the demo data set and
> are labelled as such. The time-saving figure is an estimate with its assumptions
> spelled out, not a measured result.

## The business

A car workshop with four mechanics and one administrator. Customers write wherever it
is convenient for them: Telegram, the email address on the business card, the form on
the website. Three languages are in daily use — Armenian, Russian and English.

## The problem

The administrator's day, before automation:

- **Repetition.** Most incoming messages are variations of a few questions: prices,
  opening hours, the address, whether one can pay by card, how long a job takes. Each
  one is answered by hand, usually with a delay because the administrator is busy with
  customers at the desk.
- **Bookings fall through.** A booking arrives as "can I come tomorrow for an oil
  change?" — without a phone number or a time. It takes two or three more messages to
  complete, and then it must be copied into the spreadsheet the mechanics use. Some
  never make it there.
- **Complaints drown.** An angry message looks like any other message in the chat list
  and is noticed when it is already too late to react well.
- **Nobody watches the market.** Competitor prices are checked occasionally, by
  opening their websites, when someone remembers.
- **No overview.** To learn how yesterday went, the owner scrolls through three inboxes.

None of this is difficult work. It is simply work that never ends, and every slow
answer is a customer who may have already written to the next workshop.

## The solution

One operator that sits behind all three channels and treats every message the same way:

| Step | What happens |
|---|---|
| 1. Receive | The message is stored first, so it cannot be lost, then queued. |
| 2. Understand | The AI classifies it and pulls out booking details. |
| 3. Decide | Plain rules decide: answer, book, ask for missing details, or hand to a human. |
| 4. Act | Reply in the customer's language, save the booking, issue the invoice, update the sheet, alert the owner. |
| 5. Report | Everything is visible in the panel; the owner gets a summary each morning. |

## What is automated, and what is deliberately not

**Automated**

- Answers to factual questions, taken only from the company's own documents and always
  naming the source.
- Bookings: extracting the service, date, time, name and phone; asking for whatever is
  missing; confirming; writing the row to the spreadsheet; sending a PDF invoice.
- Filtering spam, which is stored but never answered.
- A daily check of competitor price pages with a short written summary of what changed.
- The morning report.

**Deliberately left to people**

- **Complaints.** The customer gets an immediate apology and a promise that the manager
  will be in touch; the manager gets an alert. The AI does not argue or negotiate.
- **Anything the documents do not cover.** The assistant says it does not know and
  passes the question on rather than guessing.
- **Anything it is unsure about.** Below a configurable confidence threshold the
  conversation goes to a human.
- **Suspicious messages** that try to instruct the assistant.

This split is the core of the design: the routine goes to the machine, judgement stays
with people, and the boundary between them is explicit and adjustable.

## What the demo data shows

Measured on the built-in demo (two weeks of sample traffic processed by the real
pipeline with the offline rule-based AI):

| Measure | Value on demo data |
|---|---|
| Conversations closed without a human | about 9 in 10 of those that needed a reply |
| Bookings captured into the spreadsheet | every completed booking, automatically |
| Complaints auto-answered on substance | 0 — all handed to a human |

And from the evaluation set of 112 messages (see the README): the offline baseline
classified 97% of messages correctly and refused to give an unsupported answer in 92%
of question cases. Its weakness is recall — it found a correct answer for 60% of the
answerable questions and handed most of the rest to a human. A real LLM is expected to
answer more of them; that number will be published when it has been measured.

**Caveat.** The demo traffic is a sample I wrote, not real customer messages. Real
customers are messier, so real automation rates will be lower than on demo data.

## Estimated time saved

A way to estimate it for a real business, with every assumption visible:

```
minutes saved per day = messages per day × share handled automatically × minutes per message
```

Illustrative example — the inputs are assumptions, not measurements:

| Input | Assumed value | Comment |
|---|---|---|
| Messages per day | 40 | all channels together |
| Share handled without a human | 70% | deliberately lower than the demo's ~90% |
| Manual handling time per message | 3 minutes | read, look up, type, copy to the sheet |

`40 × 0.70 × 3 ≈ 84 minutes a day` — roughly an hour and a half of an administrator's
time, every working day. With 20 messages a day it is about 40 minutes; with 100 it is
three and a half hours.

The larger effect is harder to put a number on: customers get an answer in seconds
instead of hours, including evenings and Sundays, and bookings stop getting lost
between a chat and a spreadsheet.

## What it costs to run

- Infrastructure: one small server running five containers.
- AI: Google Gemini's free tier is the default, and the app paces, caches and retries
  its calls to stay within a small quota. At higher volumes this becomes a paid
  per-message cost that depends on the provider chosen.

## Risks, and how the design addresses them

| Risk | Mitigation |
|---|---|
| The AI states something false | Answers only from retrieved documents, cited; otherwise hand-off |
| A message is lost during an outage | Stored in PostgreSQL before processing; sweeper re-queues |
| A customer gets duplicate replies or invoices | Idempotency keys on every step |
| A customer tries to manipulate the assistant | Three-layer injection defence; the model has no tools and cannot set prices |
| The AI vendor is down or over quota | Retries with backoff; after that a human is alerted and takes over |
| Personal data leaks through logs | Phones and emails are masked in logs and the audit trail |

## Adapting it to another business

Nothing in the code is specific to cars. A dental clinic, a beauty salon or a repair
shop needs three things: its own `config/business.yaml` (services, prices, hours), its
own knowledge-base documents, and its channel credentials.
