"""Offline provider: no keys, no network, fully deterministic.

It makes ``make demo`` and CI work for anyone, and it doubles as the baseline
in the evals. It is rule-based (see ``ai/rules.py``) and *extractive*: answers
are sentences copied from the knowledge base, so it cannot invent facts.
"""

from __future__ import annotations

import re

from ai_operator.ai import rules
from ai_operator.ai.embeddings import LOCAL_EMBEDDING_MODEL, content_overlap, hash_embed
from ai_operator.ai.guard import detect_injection
from ai_operator.ai.providers.base import AIProvider
from ai_operator.ai.schemas import (
    Analysis,
    AnalyzeRequest,
    Extraction,
    GroundedAnswer,
    PriceChange,
    RetrievedChunk,
)
from ai_operator.core.language import detect_language

_SENTENCE_END = re.compile(r"(?<=[.!?։])\s+|\n+")
_MIN_OVERLAP = 0.34

_PRICE_TEXT = {
    "en": {
        "none": "No competitor price changes since the last check.",
        "head": "Competitor prices changed for {n} item(s): {up} went up, {down} went down.",
        "top": "Biggest move: {competitor} - {item}: {old} -> {new} {cur} ({pct:+.0f}%).",
        "cheaper": "{competitor} is now cheaper than us on {item} ({new} vs our {ours} {cur}).",
        "advice_down": "Consider reviewing our price for the affected services.",
        "advice_up": "Our prices look competitive; no action needed.",
    },
    "ru": {
        "none": "С прошлой проверки цены конкурентов не изменились.",
        "head": "Цены конкурентов изменились по {n} позициям: выросли - {up}, снизились - {down}.",
        "top": "Самое заметное: {competitor} - {item}: {old} -> {new} {cur} ({pct:+.0f}%).",
        "cheaper": "{competitor} теперь дешевле нас: «{item}» ({new} против {ours} {cur}).",
        "advice_down": "Стоит пересмотреть нашу цену на эти услуги.",
        "advice_up": "Наши цены остаются конкурентными, действий не требуется.",
    },
    "hy": {
        "none": "Վերջին ստուգումից հետո մրցակիցների գները չեն փոխվել։",
        "head": "Մրցակիցների գները փոխվել են {n} դիրքով․ թանկացել է {up}, էժանացել՝ {down}։",
        "top": "Ամենանկատելին՝ {competitor} - {item}: {old} -> {new} {cur} ({pct:+.0f}%)։",
        "cheaper": "{competitor}-ն այժմ մեզնից էժան է «{item}» ծառայության համար "
        "({new}՝ մեր {ours} {cur}-ի դիմաց)։",
        "advice_down": "Արժե վերանայել մեր գինը այս ծառայությունների համար։",
        "advice_up": "Մեր գները մնում են մրցունակ, գործողություն պետք չէ։",
    },
}


class MockProvider(AIProvider):
    name = "mock"
    model = "rules-v1"
    embedding_model = LOCAL_EMBEDDING_MODEL
    relevance_floor = 0.12

    async def analyze(self, request: AnalyzeRequest) -> Analysis:
        text = f"{request.subject}\n{request.text}" if request.subject else request.text
        keywords = {
            s.code: s.keywords + tuple(n.lower() for n in s.names) for s in request.services
        }
        now = request.now.replace(tzinfo=None)

        when = rules.parse_datetime(request.text, now)
        extraction = Extraction(
            name=rules.find_name(request.text),
            phone=rules.find_phone(request.text),
            service_code=rules.find_service(text, keywords),
            scheduled_at=when.strftime("%Y-%m-%dT%H:%M") if when else None,
            amount=rules.find_amount(request.text),
        )
        has_booking_data = bool(extraction.phone or extraction.name or when)
        category, confidence = rules.classify(
            text,
            has_pending_booking=request.pending is not None,
            has_booking_data=has_booking_data,
        )
        language = detect_language(request.text)
        return Analysis(
            category=category,
            confidence=confidence,
            language=language,
            summary=f"{category.value} ({language}); service: {extraction.service_code or 'n/a'}",
            extraction=extraction,
            injection_suspected=bool(detect_injection(text)),
        )

    async def answer(
        self, question: str, language: str, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        if not chunks:
            return GroundedAnswer(found=False)
        scored = [
            (content_overlap(question, f"{c.title}\n{c.content}"), c.score, c) for c in chunks
        ]
        overlap, _, best = max(scored, key=lambda item: (item[0], item[1]))
        if overlap < _MIN_OVERLAP:
            return GroundedAnswer(found=False)

        lines = [
            s.strip()
            for s in _SENTENCE_END.split(best.content)
            if s.strip() and not s.lstrip().startswith("#")
        ]
        # A chunk starts with its section heading; a heading is not an answer.
        sentences = [s for s in lines if s[-1] in ".!?։:" or len(s) > 60] or lines
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: content_overlap(question, sentences[i]),
            reverse=True,
        )[:2]
        picked = [sentences[i] for i in sorted(ranked) if content_overlap(question, sentences[i])]
        if not picked:
            picked = sentences[:2]
        return GroundedAnswer(found=True, answer=" ".join(picked), source_ids=[best.id])

    async def summarize_prices(self, changes: list[PriceChange], language: str) -> str:
        text = _PRICE_TEXT.get(language, _PRICE_TEXT["en"])
        moved = [c for c in changes if c.old_price is not None and c.delta_percent]
        if not moved:
            return text["none"]
        up = sum(1 for c in moved if (c.delta_percent or 0) > 0)
        down = len(moved) - up
        lines = [text["head"].format(n=len(moved), up=up, down=down)]
        top = max(moved, key=lambda c: abs(c.delta_percent or 0))
        lines.append(
            text["top"].format(
                competitor=top.competitor,
                item=top.item_name,
                old=f"{top.old_price:.0f}",
                new=f"{top.new_price:.0f}",
                cur=top.currency,
                pct=top.delta_percent or 0,
            )
        )
        cheaper = [c for c in moved if c.our_price is not None and c.new_price < c.our_price]
        for c in cheaper[:2]:
            lines.append(
                text["cheaper"].format(
                    competitor=c.competitor,
                    item=c.item_name,
                    new=f"{c.new_price:.0f}",
                    ours=f"{c.our_price:.0f}",
                    cur=c.currency,
                )
            )
        lines.append(text["advice_down"] if cheaper else text["advice_up"])
        return " ".join(lines)

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return [hash_embed(t) for t in texts]
