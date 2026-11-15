"""Writing bookings to a spreadsheet the owner already uses.

The database stays the source of truth; the sheet is a synced copy. Two sinks
share one contract: Google Sheets (production) and a local CSV file (demo, or
simply when no Google credentials are configured).
"""

from __future__ import annotations

import asyncio
import csv
import os
from pathlib import Path
from typing import Any, Protocol

from ai_operator.config import BusinessConfig, Settings
from ai_operator.core.logging import get_logger
from ai_operator.db.models import Lead

log = get_logger(__name__)

COLUMNS = [
    "id",
    "created_at",
    "status",
    "channel",
    "name",
    "phone",
    "email",
    "service",
    "scheduled_at",
    "amount",
    "currency",
    "language",
]


def _safe_cell(value: str) -> str:
    """Neutralise spreadsheet formulas in customer-supplied text (CSV injection)."""
    return f"'{value}" if value[:1] in ("=", "+", "-", "@") else value


def lead_to_row(lead: Lead, business: BusinessConfig) -> list[str]:
    service = business.service(lead.service_code)
    scheduled = lead.scheduled_at.astimezone(business.tz) if lead.scheduled_at else None
    return [
        str(lead.id),
        f"{lead.created_at.astimezone(business.tz):%Y-%m-%d %H:%M}",
        lead.status,
        lead.channel,
        _safe_cell(lead.name or ""),
        lead.phone or "",
        _safe_cell(lead.email or ""),
        service.name.en if service else _safe_cell(lead.service_text or ""),
        f"{scheduled:%Y-%m-%d %H:%M}" if scheduled else "",
        f"{lead.amount:.0f}" if lead.amount is not None else "",
        lead.currency,
        lead.language,
    ]


class LeadSink(Protocol):
    name: str

    async def upsert(self, row: list[str]) -> None:
        """Insert the row, or replace the existing row with the same id."""


class CsvSink:
    name = "csv"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    def _write(self, row: list[str]) -> None:
        rows: list[list[str]] = []
        if self._path.exists():
            with self._path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))[1:]
        for index, existing in enumerate(rows):
            if existing and existing[0] == row[0]:
                rows[index] = row
                break
        else:
            rows.append(row)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        os.replace(tmp, self._path)  # atomic: readers never see a half-written file

    async def upsert(self, row: list[str]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write, row)


class GoogleSheetsSink:
    name = "google_sheets"

    def __init__(self, spreadsheet_id: str, service_account_file: str) -> None:
        self._spreadsheet_id = spreadsheet_id
        self._service_account_file = service_account_file
        self._worksheet: Any = None

    def _sheet(self) -> Any:
        if self._worksheet is None:
            import gspread

            client = gspread.service_account(filename=self._service_account_file)
            self._worksheet = client.open_by_key(self._spreadsheet_id).sheet1
        return self._worksheet

    def _write(self, row: list[str]) -> None:
        sheet = self._sheet()
        if sheet.row_values(1) != COLUMNS:
            sheet.update(range_name="A1", values=[COLUMNS])
        cell = sheet.find(row[0], in_column=1)
        if cell is not None:
            sheet.update(range_name=f"A{cell.row}", values=[row], value_input_option="RAW")
        else:
            # RAW: values are stored as text, never evaluated as formulas.
            sheet.append_row(row, value_input_option="RAW")

    async def upsert(self, row: list[str]) -> None:
        await asyncio.to_thread(self._write, row)


def build_lead_sink(settings: Settings) -> LeadSink:
    if settings.sheets_enabled:
        return GoogleSheetsSink(settings.google_sheets_id, settings.google_service_account_file)
    log.info("sheets.csv_fallback", path=str(settings.leads_csv))
    return CsvSink(settings.leads_csv)
