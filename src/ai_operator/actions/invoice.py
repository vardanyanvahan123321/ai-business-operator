"""PDF invoices (WeasyPrint), in the customer's language.

An invoice is a database row with a stable number. The PDF is *derived* from it
and rendered on demand, so there is no file state to keep consistent: lose the
file and it is simply rendered again, byte-for-byte equivalent.
"""

from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache, partial
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_operator.config import BusinessConfig
from ai_operator.core.formatting import format_datetime, format_money
from ai_operator.core.i18n import t
from ai_operator.db.models import Invoice, Lead

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


@lru_cache
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),  # customer names are untrusted input
    )


async def ensure_invoice(session: AsyncSession, lead: Lead) -> Invoice | None:
    """Create the invoice for a lead exactly once. No amount - no invoice."""
    if lead.amount is None:
        return None
    existing = await session.scalar(select(Invoice).where(Invoice.lead_id == lead.id))
    if existing is not None:
        return existing
    invoice = Invoice(
        number=f"tmp-{uuid.uuid4().hex[:20]}",
        lead_id=lead.id,
        amount=lead.amount,
        currency=lead.currency,
        language=lead.language,
    )
    session.add(invoice)
    await session.flush()
    await session.refresh(invoice)
    # The number comes from the primary key: unique and gap-free enough for a demo.
    invoice.number = f"INV-{invoice.issued_at:%Y}-{invoice.id:05d}"
    await session.flush()
    return invoice


def render_invoice_html(
    invoice: Invoice, lead: Lead, business: BusinessConfig, *, demo: bool = False
) -> str:
    lang = invoice.language
    service = business.service(lead.service_code)
    return (
        _environment()
        .get_template("invoice.html")
        .render(
            lang=lang,
            t=partial(t, lang),
            company=business.name.get(lang),
            seller=business.seller,
            number=invoice.number,
            issued=f"{invoice.issued_at.astimezone(business.tz):%d.%m.%Y}",
            customer_name=lead.name or t(lang, "common.not_set"),
            customer_phone=lead.phone,
            appointment=format_datetime(lead.scheduled_at, lang, business.tz)
            if lead.scheduled_at
            else None,
            service_name=service.name.get(lang) if service else (lead.service_text or "-"),
            amount=format_money(invoice.amount),
            currency=invoice.currency,
            estimate=bool(service and service.price_from),
            demo=demo,
        )
    )


def _write_pdf(html: str, path: Path) -> None:
    from weasyprint import HTML  # imported lazily: it loads native libraries

    path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(str(path))


async def ensure_invoice_pdf(
    invoice: Invoice, lead: Lead, business: BusinessConfig, out_dir: Path, *, demo: bool = False
) -> Path:
    """Return the PDF path, rendering the file if it is not on disk yet."""
    path = out_dir / f"{invoice.number}.pdf"
    if not path.exists():
        html = render_invoice_html(invoice, lead, business, demo=demo)
        await asyncio.to_thread(_write_pdf, html, path)
    return path
