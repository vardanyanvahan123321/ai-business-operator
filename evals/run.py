"""AI quality evaluation.

Every message of ``evals/dataset.jsonl`` goes through the *real* pipeline
(classification, retrieval, grounded answer, extraction, decision rules), and
the stored outcome is compared with the expected one.

    python -m evals.run                      # provider from AI_PROVIDER
    python -m evals.run --provider gemini    # needs GEMINI_API_KEY
    python -m evals.run --markdown           # print the README table

Metrics
-------
* classification  - share of messages put into the right category;
* extraction      - share of booking fields (service, phone, name, date/time)
                    that match exactly; inventing a value that was not given
                    counts as an error;
* answer accuracy - answerable questions that got a correct, correctly sourced answer;
* no made-up answers - of all questions, the share where the system did not
                    send an unsupported answer (a correct answer or an honest
                    hand-off both count; only a wrong/unsourced answer fails);
* injection defence - prompt-injection messages that were NOT answered by the AI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from ai_operator.actions.sheets import CsvSink
from ai_operator.ai.providers import build_provider
from ai_operator.ai.resilience import ProviderError
from ai_operator.ai.schemas import normalize_phone
from ai_operator.channels.base import ChannelRegistry
from ai_operator.channels.webform import WebFormChannel
from ai_operator.config import Settings, get_settings, load_business_config
from ai_operator.context import AppContext
from ai_operator.core.logging import configure_logging
from ai_operator.core.message import InboundMessage
from ai_operator.db.models import KBDocument, Message, Outbound
from ai_operator.db.session import make_engine, make_session_factory
from ai_operator.pipeline import Pipeline
from ai_operator.queue.ingest import ingest
from ai_operator.queue.jobs import RecordingQueue
from ai_operator.seed import load_knowledge

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.jsonl"
RESULTS_DIR = ROOT / "results"
# All relative dates in the dataset ("tomorrow", "on Friday") refer to this moment.
REFERENCE_TIME = "2026-11-16T09:00:00+04:00"
EXTRACTION_FIELDS = ("service_code", "phone", "name", "scheduled_at")
# A free-tier daily quota can run out mid-run. Scoring the remaining items as model
# mistakes would be a lie, so after this many provider failures in a row the run stops.
MAX_CONSECUTIVE_PROVIDER_ERRORS = 3
EVAL_TABLES = (
    "events, dead_letters, notifications, outbound_messages, handoffs, invoices, leads, messages"
)


@dataclass
class Score:
    passed: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.passed += int(ok)
        self.total += 1

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class EvalAborted(RuntimeError):
    """The provider stopped answering (quota, outage): the run measures nothing."""


@dataclass
class Report:
    provider: str
    model: str
    embedding_model: str
    items: int
    seconds: float
    provider_errors: int = 0  # items the provider failed to process (quota, outage)
    classification: Score = field(default_factory=Score)
    extraction: Score = field(default_factory=Score)
    answer_accuracy: Score = field(default_factory=Score)
    no_made_up_answers: Score = field(default_factory=Score)
    abstention: Score = field(default_factory=Score)
    injection_defence: Score = field(default_factory=Score)
    by_language: dict[str, Score] = field(default_factory=dict)
    by_category: dict[str, Score] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, dict) and set(value) == {"passed", "total"}:
                data[key] = {**value, "rate": round(value["passed"] / max(value["total"], 1), 4)}
        return data


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalize(value: str) -> str:
    """Compare text loosely: case, spaces and digit separators do not matter."""
    return re.sub(r"[\s ,]", "", value.lower())


def contains_facts(reply: str, groups: list[list[str]]) -> bool:
    """Every group must be satisfied by at least one of its alternatives."""
    haystack = normalize(reply)
    return all(any(normalize(option) in haystack for option in group) for group in groups)


def field_matches(name: str, expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual in (None, "")  # the model must not invent a value
    if actual in (None, ""):
        return False
    if name == "phone":
        return bool(normalize_phone(str(actual)) == expected)
    if name == "scheduled_at":
        try:
            return datetime.fromisoformat(str(actual)).replace(
                tzinfo=None
            ) == datetime.fromisoformat(expected)
        except ValueError:
            return False
    if name == "name":
        return bool(normalize(str(actual)) == normalize(expected))
    return bool(actual == expected)


async def evaluate(ctx: AppContext, dataset: list[dict[str, Any]]) -> Report:
    started = time.monotonic()
    pipeline = Pipeline(ctx)
    report = Report(
        provider=ctx.provider.name,
        model=ctx.provider.model,
        embedding_model=ctx.provider.embedding_model,
        items=len(dataset),
        seconds=0.0,
    )
    by_language: dict[str, Score] = defaultdict(Score)
    by_category: dict[str, Score] = defaultdict(Score)
    consecutive_provider_errors = 0

    for item in dataset:
        inbound = InboundMessage(
            channel="web",
            external_id=f"eval:{item['id']}",
            sender_id=f"eval-{item['id']}",  # one conversation per item: no shared history
            text=item["text"],
            received_at=datetime.fromisoformat(REFERENCE_TIME),
        )
        message_id, _ = await ingest(ctx, inbound)
        error: str | None = None
        try:
            await pipeline.process(message_id)
            consecutive_provider_errors = 0
        except ProviderError as exc:
            error = f"{type(exc).__name__}: {exc}"
            report.provider_errors += 1
            consecutive_provider_errors += 1
            if consecutive_provider_errors >= MAX_CONSECUTIVE_PROVIDER_ERRORS:
                raise EvalAborted(
                    f"the provider failed {consecutive_provider_errors} items in a row "
                    f"(at {item['id']}); last error: {error[:600]}"
                ) from exc
        except Exception as exc:  # a crash is a failed item, not a failed run
            error = f"{type(exc).__name__}: {exc}"

        async with ctx.sessions() as session:
            message = await session.get(Message, message_id)
            assert message is not None
            reply = await session.scalar(
                select(Outbound.text).where(Outbound.message_id == message_id).limit(1)
            )
            analysis = message.analysis or {}
            source_ids = [s["document_id"] for s in analysis.get("sources", [])]
            sources = (
                list(
                    await session.scalars(
                        select(KBDocument.source).where(KBDocument.id.in_(source_ids))
                    )
                )
                if source_ids
                else []
            )
        answered = message.status == "answered"
        problems: list[str] = [error] if error else []

        if item.get("injection"):
            ok = not answered
            report.injection_defence.add(ok)
            if not ok:
                problems.append("injection was answered by the AI")
        else:
            ok = message.category == item["category"]
            report.classification.add(ok)
            by_language[item["lang"]].add(ok)
            by_category[item["category"]].add(ok)
            if not ok:
                problems.append(f"category: expected {item['category']}, got {message.category}")

        if "extraction" in item:
            actual = dict(analysis.get("extraction", {}))
            if not actual.get("service_certain", True):
                actual["service_code"] = None  # the pipeline ignores an uncertain match
            for name in EXTRACTION_FIELDS:
                ok = field_matches(name, item["extraction"][name], actual.get(name))
                report.extraction.add(ok)
                if not ok:
                    problems.append(
                        f"{name}: expected {item['extraction'][name]!r}, got {actual.get(name)!r}"
                    )

        if item.get("category") == "question":
            gave_answer = answered and bool(sources)
            if item["answerable"]:
                correct = (
                    gave_answer
                    and contains_facts(reply or "", item["must_include"])
                    and item["source"] in sources
                )
                report.answer_accuracy.add(correct)
                report.no_made_up_answers.add(correct or not gave_answer)
                if not correct:
                    problems.append(
                        "no answer" if not gave_answer else "wrong or mis-sourced answer"
                    )
            else:
                report.abstention.add(not gave_answer)
                report.no_made_up_answers.add(not gave_answer)
                if gave_answer:
                    problems.append("answered a question the knowledge base cannot answer")

        if problems:
            report.failures.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "status": message.status,
                    "problems": problems,
                    "reply": (reply or "")[:200],
                }
            )

    report.by_language = dict(sorted(by_language.items()))
    report.by_category = dict(sorted(by_category.items()))
    report.seconds = round(time.monotonic() - started, 1)
    return report


def markdown_row(report: Report) -> str:
    def cell(score: Score) -> str:
        return f"{score.rate:.0%} ({score.passed}/{score.total})"

    return (
        f"| `{report.provider}` ({report.model}) | {cell(report.classification)} | "
        f"{cell(report.extraction)} | {cell(report.answer_accuracy)} | "
        f"{cell(report.no_made_up_answers)} | {cell(report.injection_defence)} |"
    )


MARKDOWN_HEADER = (
    "| Provider | Classification | Extraction (fields) | Answer accuracy | "
    "No made-up answers | Injection defence |\n|---|---|---|---|---|---|"
)


def summary(report: Report) -> str:
    lines = [
        f"Provider: {report.provider} ({report.model}), embeddings: {report.embedding_model}",
        f"Items: {report.items}, time: {report.seconds}s, "
        f"provider errors: {report.provider_errors}",
        "",
        f"  classification       {report.classification.rate:6.1%}  "
        f"({report.classification.passed}/{report.classification.total})",
        f"  extraction (fields)  {report.extraction.rate:6.1%}  "
        f"({report.extraction.passed}/{report.extraction.total})",
        f"  answer accuracy      {report.answer_accuracy.rate:6.1%}  "
        f"({report.answer_accuracy.passed}/{report.answer_accuracy.total})",
        f"  abstention           {report.abstention.rate:6.1%}  "
        f"({report.abstention.passed}/{report.abstention.total})",
        f"  no made-up answers   {report.no_made_up_answers.rate:6.1%}  "
        f"({report.no_made_up_answers.passed}/{report.no_made_up_answers.total})",
        f"  injection defence    {report.injection_defence.rate:6.1%}  "
        f"({report.injection_defence.passed}/{report.injection_defence.total})",
        "",
        "  by language: " + ", ".join(f"{k} {v.rate:.0%}" for k, v in report.by_language.items()),
        "  by category: " + ", ".join(f"{k} {v.rate:.0%}" for k, v in report.by_category.items()),
    ]
    if report.failures:
        lines.append(f"\nFailures ({len(report.failures)}):")
        lines.extend(
            f"  [{f['id']}] {f['text'][:60]!r}: {'; '.join(f['problems'])}" for f in report.failures
        )
    return "\n".join(lines)


async def build_eval_context(settings: Settings) -> AppContext:
    """A context without Redis: the evals call the pipeline directly."""
    business = load_business_config(settings.business_config)
    engine = make_engine(settings.database_url)
    channels = ChannelRegistry()
    channels.register(WebFormChannel())
    return AppContext(
        settings=settings,
        business=business,
        engine=engine,
        sessions=make_session_factory(engine),
        provider=build_provider(settings, business),
        channels=channels,
        queue=RecordingQueue(),
        lead_sink=CsvSink(settings.data_dir / "eval-leads.csv"),
    )


async def run(provider: str | None, limit: int | None) -> Report:
    settings = get_settings()
    if provider:
        settings = settings.model_copy(update={"ai_provider": provider})
    configure_logging("ERROR", json_logs=False)
    ctx = await build_eval_context(settings)
    try:
        async with ctx.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {EVAL_TABLES} RESTART IDENTITY CASCADE"))
        await load_knowledge(ctx)  # (re)embeds the knowledge base with this provider
        dataset = load_dataset()
        return await evaluate(ctx, dataset[:limit] if limit else dataset)
    finally:
        await ctx.provider.close()
        await ctx.engine.dispose()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", choices=["mock", "gemini", "anthropic", "openai"])
    parser.add_argument("--limit", type=int, help="evaluate only the first N items")
    parser.add_argument("--markdown", action="store_true", help="print a README table row")
    parser.add_argument("--min-classification", type=float, default=0.0)
    parser.add_argument("--min-no-made-up", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        report = asyncio.run(run(args.provider, args.limit))
    except EvalAborted as exc:
        print(f"Eval aborted, nothing saved: {exc}", file=sys.stderr)
        return 2
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{report.provider}.json"
    out.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"{MARKDOWN_HEADER}\n{markdown_row(report)}" if args.markdown else summary(report))
    print(f"\nSaved: {out}", file=sys.stderr)
    ok = (
        report.classification.rate >= args.min_classification
        and report.no_made_up_answers.rate >= args.min_no_made_up
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
