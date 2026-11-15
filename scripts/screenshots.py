"""Take README screenshots of a running instance with a headless browser.

    uv run --group screenshots python scripts/screenshots.py [base_url] [out_dir]

Expects the demo to be running (``make demo``) with the demo password.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8088"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots")
PASSWORD = "demo"

PANEL_PAGES = [
    ("dashboard", "/panel"),
    ("inbox", "/panel/inbox"),
    ("handoffs", "/panel/handoffs?show=all"),
    ("leads", "/panel/leads"),
    ("knowledge-base", "/panel/kb"),
    ("prices", "/panel/prices"),
    ("reports", "/panel/reports"),
    ("logs", "/panel/logs"),
    ("settings", "/panel/settings"),
]


def login(page: Page, lang: str) -> None:
    page.goto(f"{BASE}/panel/login?lang={lang}")
    page.fill("#password", PASSWORD)
    page.click("button[type=submit]")
    page.wait_for_url(f"{BASE}/panel")


def shoot(page: Page, name: str, full_page: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page.wait_for_load_state("networkidle")
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full_page)
    print(f"saved {OUT / name}.png")


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1360, "height": 860}, device_scale_factor=2, color_scheme="light"
        )
        page = context.new_page()

        login(page, "en")
        for name, path in PANEL_PAGES:
            page.goto(f"{BASE}{path}")
            shoot(page, name, full_page=name not in {"inbox", "logs", "leads"})

        # A message detail page with a booking, and one handed to a human.
        page.goto(f"{BASE}/panel/inbox?category=order&status=answered")
        page.click("table.data tbody tr a")
        shoot(page, "message-booking")
        page.goto(f"{BASE}/panel/inbox?status=handoff")
        page.click("table.data tbody tr a")
        shoot(page, "message-handoff")

        # The knowledge-base retrieval playground.
        page.goto(f"{BASE}/panel/kb")
        page.fill("input[name=q]", "Do you give a warranty?")
        page.wait_for_selector("#kb-results li")
        shoot(page, "knowledge-search", full_page=False)

        # Same panel in Russian and Armenian.
        for lang in ("ru", "hy"):
            page.goto(f"{BASE}/panel/lang/{lang}")
            page.goto(f"{BASE}/panel")
            shoot(page, f"dashboard-{lang}")
        page.goto(f"{BASE}/panel/lang/en")

        # Public form: ask a question and wait for the live reply.
        page.goto(f"{BASE}/request?lang=en")
        shoot(page, "form", full_page=False)
        page.fill("#name", "Alex")
        page.fill("#contact", "+374 91 000111")
        page.fill("#text", "Hi! Please book me for an oil change tomorrow at 10:00.")
        page.click("button[type=submit]")
        page.wait_for_selector(".bubble.out:not(.pending)", timeout=30000)
        shoot(page, "form-reply", full_page=False)

        # Dark theme.
        dark = browser.new_context(
            viewport={"width": 1360, "height": 860}, device_scale_factor=2, color_scheme="dark"
        )
        dark_page = dark.new_page()
        login(dark_page, "en")
        shoot(dark_page, "dashboard-dark")
        browser.close()


if __name__ == "__main__":
    main()
