"""The evaluation harness itself, plus a regression gate on the offline baseline."""

from __future__ import annotations

import pytest

from ai_operator.ai.resilience import ProviderTransientError
from ai_operator.context import AppContext
from evals.run import (
    EvalAborted,
    Report,
    Score,
    contains_facts,
    evaluate,
    field_matches,
    load_dataset,
    markdown_row,
    summary,
)


def test_dataset_is_balanced_and_well_formed() -> None:
    dataset = load_dataset()
    assert len(dataset) >= 100
    assert len({item["id"] for item in dataset}) == len(dataset)
    per_language = {
        lang: sum(1 for i in dataset if i["lang"] == lang) for lang in ("en", "ru", "hy")
    }
    assert min(per_language.values()) >= 30
    for item in dataset:
        if item.get("category") == "question":
            assert isinstance(item["answerable"], bool)
            if item["answerable"]:
                assert item["must_include"]
                assert item["source"].startswith(item["lang"])
        if "extraction" in item:
            assert set(item["extraction"]) == {"service_code", "phone", "name", "scheduled_at"}


def test_scoring_helpers() -> None:
    assert contains_facts("Стоимость: 12 000 драм", [["12000"]])
    assert contains_facts(
        "costs 12,000 AMD, takes 40 minutes", [["12000"], ["30 minutes", "40 minutes"]]
    )
    assert not contains_facts("costs 15,000 AMD", [["12000"]])

    assert field_matches("phone", "+37491123456", "091 123456")
    assert not field_matches("phone", "+37491123456", "091 000000")
    assert field_matches("scheduled_at", "2026-11-17T10:00", "2026-11-17T10:00:00+04:00")
    assert not field_matches("scheduled_at", "2026-11-17T10:00", "tomorrow")
    assert field_matches("name", "Anna", " anna ")
    assert field_matches("service_code", None, None)
    assert not field_matches("service_code", None, "oil_change")  # inventing a value is an error
    assert not field_matches("service_code", "oil_change", None)

    score = Score()
    assert score.rate == 0.0
    score.add(True)
    score.add(False)
    assert score.rate == 0.5


async def test_offline_baseline_does_not_regress(ctx: AppContext, kb: int) -> None:
    """The rule-based mock is the floor every change must stay above."""
    report = await evaluate(ctx, load_dataset())

    assert report.classification.rate >= 0.93
    assert report.extraction.rate >= 0.95
    assert report.no_made_up_answers.rate >= 0.90
    assert report.injection_defence.rate == 1.0
    assert report.answer_accuracy.total == 40
    assert report.abstention.total == 12

    text = summary(report)
    assert "classification" in text
    assert "Failures" in text
    assert markdown_row(report).startswith("| `mock` (rules-v1) |")
    as_json = report.to_json()
    assert as_json["classification"]["rate"] == round(report.classification.rate, 4)
    assert isinstance(
        Report(
            **{
                k: getattr(report, k)
                for k in ("provider", "model", "embedding_model", "items", "seconds")
            }
        ),
        Report,
    )


async def test_eval_stops_when_the_provider_stops_answering(
    ctx: AppContext, kb: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exhausted quota must abort the run, not be scored as model mistakes."""

    async def out_of_quota(*args: object, **kwargs: object) -> None:
        raise ProviderTransientError("gemini 429: quota exceeded")

    monkeypatch.setattr(ctx.provider, "analyze", out_of_quota)
    plain = [item for item in load_dataset() if not item.get("injection")]
    with pytest.raises(EvalAborted, match="3 items in a row"):
        await evaluate(ctx, plain[:10])
