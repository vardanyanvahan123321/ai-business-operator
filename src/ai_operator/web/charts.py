"""Server-rendered SVG charts.

Plain SVG generated in Python: no chart library to ship, works without
JavaScript, prints well, and is unit-testable. Colours come from CSS custom
properties (``--series-N``), so the same markup follows light and dark themes.
Every mark carries a native ``<title>`` tooltip and every chart is paired with a
table view in the template, so no value is readable by colour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

WIDTH = 720
HEIGHT = 240
PAD_LEFT = 44
PAD_RIGHT = 12
PAD_TOP = 12
PAD_BOTTOM = 28
GAP = 2  # surface gap between adjacent fills


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    values: Sequence[float]
    slot: int  # index of the categorical colour (1-based), fixed per entity


def nice_max(value: float) -> float:
    """Round an axis maximum up to 1, 2 or 5 times a power of ten."""
    if value <= 0:
        return 1.0
    magnitude: float = 10 ** (len(str(int(value))) - 1)
    for step in (1, 2, 5, 10):
        if value <= step * magnitude:
            return float(step * magnitude)
    return float(10 * magnitude)


def compact(value: float) -> str:
    """Compact axis labels: 12500 -> 12.5k, 3000000 -> 3M."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _frame(top: float, inner_height: float, bottom: float = 0.0) -> list[str]:
    """Recessive horizontal grid lines with value labels."""
    parts = []
    for fraction in (0.0, 0.5, 1.0):
        y = PAD_TOP + inner_height * (1 - fraction)
        parts.append(
            f'<line class="grid" x1="{PAD_LEFT}" x2="{WIDTH - PAD_RIGHT}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text class="axis" x="{PAD_LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">'
            f"{compact(bottom + (top - bottom) * fraction)}</text>"
        )
    return parts


def _x_labels(labels: Sequence[str], step: float) -> list[str]:
    """At most ~7 evenly spaced x labels, so they never collide."""
    every = max(1, len(labels) // 7)
    return [
        f'<text class="axis" x="{PAD_LEFT + step * (i + 0.5):.1f}" y="{HEIGHT - 8}" '
        f'text-anchor="middle">{escape(label)}</text>'
        for i, label in enumerate(labels)
        if i % every == 0
    ]


def stacked_bars(labels: Sequence[str], series: Sequence[Series], title: str) -> str:
    """Stacked vertical bars: parts of a whole over time."""
    inner_w = WIDTH - PAD_LEFT - PAD_RIGHT
    inner_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    totals = [sum(s.values[i] for s in series) for i in range(len(labels))]
    top = nice_max(max(totals, default=0))
    step = inner_w / max(len(labels), 1)
    bar_w = min(28.0, step * 0.62)
    parts = _frame(top, inner_h)
    for i, label in enumerate(labels):
        x = PAD_LEFT + step * i + (step - bar_w) / 2
        y: float = PAD_TOP + inner_h
        visible = [s for s in series if s.values[i] > 0]
        for index, s in enumerate(visible):
            height = max(inner_h * s.values[i] / top - GAP, 1.0)
            y -= height + GAP
            # Only the top segment gets a rounded data-end.
            radius = 3 if index == len(visible) - 1 else 0
            parts.append(
                f'<rect x="{x:.1f}" y="{y + GAP:.1f}" width="{bar_w:.1f}" height="{height:.1f}" '
                f'rx="{radius}" fill="var(--series-{s.slot})">'
                f"<title>{escape(label)} - {escape(s.label)}: {s.values[i]:g}</title></rect>"
            )
    parts.extend(_x_labels(labels, step))
    return _svg(parts, title)


def line_chart(labels: Sequence[str], series: Sequence[Series], title: str) -> str:
    """Lines for change over time; the last point of each series is labelled directly.

    Unlike bars, lines do not need a zero baseline: the axis is zoomed to the
    data range, otherwise prices that differ by a few percent would overlap.
    """
    inner_w = WIDTH - PAD_LEFT - PAD_RIGHT - 96  # room for end labels
    inner_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    values = [v for s in series for v in s.values]
    low, high = min(values, default=0.0), max(values, default=1.0)
    margin = (high - low) * 0.35 or max(high * 0.1, 1.0)
    bottom, top = max(0.0, low - margin), high + margin
    step = inner_w / max(len(labels) - 1, 1)
    parts = _frame(top, inner_h, bottom)
    ends: list[tuple[float, float, str]] = []
    for s in series:
        points = [
            (PAD_LEFT + step * i, PAD_TOP + inner_h * (1 - (value - bottom) / (top - bottom)))
            for i, value in enumerate(s.values)
        ]
        if not points:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline class="line" points="{path}" stroke="var(--series-{s.slot})"/>')
        for (x, y), label, value in zip(points, labels, s.values, strict=True):
            parts.append(
                f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                f'fill="var(--series-{s.slot})">'
                f"<title>{escape(label)} - {escape(s.label)}: {value:g}</title></circle>"
            )
        ends.append((points[-1][1], points[-1][0], s.label))
    # Keep end labels at least one line apart so they never overlap.
    previous = -100.0
    for y, x, label in sorted(ends):
        y = max(y, previous + 14)
        previous = y
        parts.append(
            f'<text class="end-label" x="{x + 10:.1f}" y="{y + 4:.1f}">{escape(label)}</text>'
        )
    every = max(1, len(labels) // 7)
    parts.extend(
        f'<text class="axis" x="{PAD_LEFT + step * i:.1f}" y="{HEIGHT - 8}" '
        f'text-anchor="middle">{escape(label)}</text>'
        for i, label in enumerate(labels)
        if i % every == 0
    )
    return _svg(parts, title)


def horizontal_bars(rows: Sequence[tuple[str, float]], title: str) -> str:
    """Magnitude across categories: one hue, values labelled on the bars."""
    row_h = 30
    label_w = 150
    height = row_h * max(len(rows), 1) + 8
    top = max((value for _, value in rows), default=0) or 1
    inner_w = WIDTH - label_w - 70
    parts = []
    for i, (label, value) in enumerate(rows):
        y = 4 + row_h * i
        width = max(inner_w * value / top, 2)
        parts.append(
            f'<text class="row-label" x="{label_w - 10}" y="{y + 18}" text-anchor="end">'
            f"{escape(label)}</text>"
            f'<rect x="{label_w}" y="{y + 5}" width="{width:.1f}" height="{row_h - 12}" rx="3" '
            f'fill="var(--series-1)"><title>{escape(label)}: {value:g}</title></rect>'
            f'<text class="value" x="{label_w + width + 8:.1f}" y="{y + 18}">{value:g}</text>'
        )
    return _svg(parts, title, height=height)


def _svg(parts: Sequence[str], title: str, height: int = HEIGHT) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-label="{escape(title)}" preserveAspectRatio="xMidYMid meet">'
        f"<title>{escape(title)}</title>{''.join(parts)}</svg>"
    )
