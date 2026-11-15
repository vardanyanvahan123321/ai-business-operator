"""Loading the knowledge base from Markdown files (``data/knowledge/<lang>/*.md``)."""

from __future__ import annotations

from pathlib import Path

from ai_operator.ai.rag import KnowledgeBase
from ai_operator.config import LANGS
from ai_operator.context import AppContext
from ai_operator.db.session import session_scope

KNOWLEDGE_DIR = Path("data/knowledge")


def document_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def read_knowledge_files(directory: Path) -> list[tuple[str, str, str, str]]:
    """``(source, title, content, language)`` for every Markdown file in the directory."""
    found: list[tuple[str, str, str, str]] = []
    for path in sorted(directory.glob("*/*.md")):
        language = path.parent.name
        if language in LANGS:
            content = path.read_text(encoding="utf-8")
            source = f"{language}/{path.name}"
            found.append((source, document_title(content, path.stem), content, language))
    return found


async def load_knowledge(ctx: AppContext, directory: Path = KNOWLEDGE_DIR) -> int:
    """Create or refresh one KB document per file. Returns the number of documents."""
    files = read_knowledge_files(directory)
    async with session_scope(ctx.sessions) as session:
        kb = KnowledgeBase(session, ctx.provider)
        for source, title, content, language in files:
            await kb.upsert_document(source=source, title=title, content=content, language=language)
    return len(files)
