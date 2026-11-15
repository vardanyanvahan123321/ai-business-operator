"""A tiny fake "competitor" website for the price-monitoring demo.

Two fictional workshops with differently structured price pages (a table and
cards), a robots.txt with a crawl delay and a disallowed area. Prices drift a
little from day to day, so the monitor has real changes to detect.

Standard library only: ``python app.py`` serves on port 8090.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AUTOFIX = {
    "Oil change": 11500,
    "Computer diagnostics": 8500,
    "Brake pads replacement": 16000,
    "Tire fitting": 9000,
    "Wheel alignment": 15000,
    "A/C recharge": 19000,
}
GARAGE_PRO = {
    "Engine oil replacement": 11000,
    "Full diagnostics": 7500,
    "Brake pad change": 15500,
    "Tyre fitting": 11000,
    "Wheel alignment": 14500,
}
ROBOTS = "User-agent: *\nCrawl-delay: 1\nDisallow: /admin/\nDisallow: /garagepro/internal/\n"


def drift(name: str, base: int, today: date) -> int:
    """Deterministic daily price movement: most days nothing, sometimes +-500."""
    digest = hashlib.sha256(f"{name}:{today.isoformat()}".encode()).digest()
    step = (-500, 0, 0, 0, 0, 500)[digest[0] % 6]
    return base + step


def autofix_page(today: date) -> str:
    rows = "".join(
        f'<tr class="item"><td class="name">{name}</td>'
        f'<td class="price">{drift(name, price, today):,} AMD</td></tr>'.replace(",", " ")
        for name, price in AUTOFIX.items()
    )
    return (
        "<!doctype html><html><head><title>AutoFix Center - prices</title></head><body>"
        "<h1>AutoFix Center (fictional demo site)</h1>"
        f'<table class="prices"><tr><th>Service</th><th>Price</th></tr>{rows}</table>'
        "</body></html>"
    )


def garagepro_page(today: date) -> str:
    cards = "".join(
        f'<div class="service-card"><h3>{name}</h3>'
        f'<p>from <span class="amount">{drift(name, price, today)}</span> dram</p></div>'
        for name, price in GARAGE_PRO.items()
    )
    return (
        "<!doctype html><html><head><title>Garage Pro - services</title></head><body>"
        f"<h1>Garage Pro (fictional demo site)</h1>{cards}</body></html>"
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        today = date.today()
        path = self.path.split("?")[0]
        if path == "/robots.txt":
            self._send(200, ROBOTS, "text/plain; charset=utf-8")
        elif path == "/autofix/prices":
            self._send(200, autofix_page(today))
        elif path == "/garagepro/services":
            self._send(200, garagepro_page(today))
        elif path == "/health":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8090"))
    print(f"competitor demo site on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
