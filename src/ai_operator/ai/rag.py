"""Knowledge base: chunking, indexing and retrieval on PostgreSQL + pgvector."""

from __future__ import annotations

import re

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_operator.ai.providers.base import AIProvider
from ai_operator.ai.schemas import RetrievedChunk
from ai_operator.core.language import detect_language
from ai_operator.db.models import KBChunk, KBDocument

MAX_CHUNK_CHARS = 700
# A small nudge towards sources in the customer's own language.
SAME_LANGUAGE_BONUS = 0.05

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def chunk_markdown(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a document into retrieval-sized chunks.

    Splits on headings and blank lines; every chunk is prefixed with the heading
    it sits under, so a chunk still makes sense when read on its own.
    """
    chunks: list[str] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(f"{heading}\n{body}".strip() if heading else body)
        buffer.clear()

    for block in re.split(r"\n\s*\n", content.strip()):
        block = block.strip()
        if not block:
            continue
        match = _HEADING.match(block.splitlines()[0])
        if match:
            flush()
            heading = match.group(1).strip()
            rest = "\n".join(block.splitlines()[1:]).strip()
            if rest:
                buffer.append(rest)
            continue
        if buffer and sum(len(b) for b in buffer) + len(block) > max_chars:
            flush()
        while len(block) > max_chars:  # a single oversized paragraph
            cut = block.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            buffer.append(block[:cut])
            flush()
            block = block[cut:].strip()
        buffer.append(block)
    flush()
    return chunks


class KnowledgeBase:
    def __init__(self, session: AsyncSession, provider: AIProvider) -> None:
        self._session = session
        self._provider = provider

    async def upsert_document(
        self, *, source: str, title: str, content: str, language: str | None = None
    ) -> KBDocument:
        """Create or replace a document and (re)build its chunks and embeddings."""
        document = await self._session.scalar(select(KBDocument).where(KBDocument.source == source))
        if document is None:
            document = KBDocument(source=source, title=title, content=content, language="en")
            self._session.add(document)
        document.title = title
        document.content = content
        document.language = language or detect_language(content)
        await self._session.flush()
        await self.index_document(document)
        return document

    async def index_document(self, document: KBDocument) -> int:
        await self._session.execute(delete(KBChunk).where(KBChunk.document_id == document.id))
        pieces = chunk_markdown(document.content)
        if not pieces:
            return 0
        # The title is embedded with the text: it often carries the key topic words.
        vectors = await self._provider.embed([f"{document.title}\n{p}" for p in pieces])
        self._session.add_all(
            KBChunk(
                document_id=document.id,
                ord=index,
                content=piece,
                embedding=vector,
                embedding_model=self._provider.embedding_model,
            )
            for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
        )
        await self._session.flush()
        return len(pieces)

    async def reindex_all(self) -> int:
        documents = (await self._session.scalars(select(KBDocument))).all()
        return sum([await self.index_document(document) for document in documents])

    async def stale_chunks(self) -> int:
        """Chunks embedded by a different model than the current one (need a reindex)."""
        count = await self._session.scalar(
            select(func.count())
            .select_from(KBChunk)
            .where(KBChunk.embedding_model != self._provider.embedding_model)
        )
        return int(count or 0)

    async def delete_document(self, document_id: int) -> None:
        document = await self._session.get(KBDocument, document_id)
        if document is not None:
            await self._session.delete(document)
            await self._session.flush()

    async def search(self, query: str, *, language: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Nearest chunks by cosine similarity, weakest matches filtered out."""
        [vector] = await self._provider.embed([query], query=True)
        distance = KBChunk.embedding.cosine_distance(vector)
        rows = await self._session.execute(
            select(KBChunk, KBDocument, distance.label("distance"))
            .join(KBDocument, KBDocument.id == KBChunk.document_id)
            .where(KBChunk.embedding_model == self._provider.embedding_model)
            .order_by(distance)
            .limit(top_k * 3)
        )
        found = []
        for chunk, document, dist in rows:
            score = 1.0 - float(dist)
            if document.language == language:
                score += SAME_LANGUAGE_BONUS
            if score >= self._provider.relevance_floor:
                found.append(
                    RetrievedChunk(
                        id=chunk.id,
                        document_id=document.id,
                        title=document.title,
                        language=document.language,
                        content=chunk.content,
                        score=round(score, 4),
                    )
                )
        found.sort(key=lambda c: c.score, reverse=True)
        return found[:top_k]
