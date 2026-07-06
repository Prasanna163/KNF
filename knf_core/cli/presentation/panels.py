from __future__ import annotations

from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ...engine.constants import CLI_NAME, CLI_SUBTITLE, CLI_VERSION
from .formatting import (
    ACCENT,
    ACCENT_BOLD,
    BRAND_STYLE,
    FAIL_STYLE,
    INFO_BORDER,
    MUTED,
    OK_STYLE,
    WARN_STYLE,
    can_render_unicode,
    glyph,
    panel_box,
    status_label,
    table_box,
)


# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------
def brand_banner() -> Panel:
    """Full-width brand header shown once at the top of a run."""
    spaced = " ".join(CLI_NAME.upper())
    title = Text()
    title.append(spaced, style=BRAND_STYLE)
    title.append(f"   {CLI_VERSION}", style=f"bold {MUTED}")
    subtitle = Text(CLI_SUBTITLE, style=ACCENT)

    body = Table.grid(padding=(0, 0))
    body.add_column(justify="center")
    body.add_row(title)
    body.add_row(subtitle)

    return Panel(
        Align.center(body),
        border_style=ACCENT,
        box=panel_box(),
        padding=(1, 4),
    )


# retained name for existing call sites / tests
def brand_panel() -> Panel:
    return brand_banner()


# ---------------------------------------------------------------------------
# Key/value + specs panels
# ---------------------------------------------------------------------------
def _kv_grid(rows: list[tuple[str, str]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, justify="right", no_wrap=True)
    table.add_column(style="bold")
    for key, value in rows:
        table.add_row(str(key), str(value))
    return table


def key_value_panel(
    rows: list[tuple[str, str]],
    *,
    title: str,
    border_style: str = ACCENT,
) -> Panel:
    return Panel(
        _kv_grid(rows),
        title=f"[{ACCENT_BOLD}]{title}[/]",
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        box=panel_box(),
    )


def specs_columns(
    config_rows: list[tuple[str, str]],
    system_rows: list[tuple[str, str]],
) -> RenderableType:
    """Run configuration and host system, side by side.

    Uses a two-column grid so the panels reliably sit next to each other and
    split the terminal width evenly (``Columns`` would wrap them onto separate
    rows on anything but a very wide terminal).
    """
    gear = glyph("⚙", "*")
    chip = glyph("▣", "#")
    config = Panel(
        _kv_grid(config_rows),
        title=f"[{ACCENT_BOLD}]{gear} Run Configuration[/]",
        title_align="left",
        border_style=ACCENT,
        padding=(1, 2),
        box=panel_box(),
    )
    system = Panel(
        _kv_grid(system_rows),
        title=f"[bold {INFO_BORDER}]{chip} System[/]",
        title_align="left",
        border_style=INFO_BORDER,
        padding=(1, 2),
        box=panel_box(),
    )
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(config, system)
    return grid


# ---------------------------------------------------------------------------
# Live stats
# ---------------------------------------------------------------------------
def stat_tiles(metrics: list[tuple[str, str, str]], *, title: str = "Live Stats") -> Panel:
    """A row of ``(label, value, style)`` tiles: bold value over a dim label."""
    grid = Table.grid(expand=True)
    for _ in metrics:
        grid.add_column(justify="center", ratio=1)
    values = [Text(str(value), style=style, justify="center") for _, value, style in metrics]
    labels = [Text(str(label), style=MUTED, justify="center") for label, _, _ in metrics]
    grid.add_row(*values)
    grid.add_row(*labels)
    pulse = glyph("◆", "*")
    return Panel(
        grid,
        title=f"[{ACCENT_BOLD}]{pulse} {title}[/]",
        title_align="left",
        border_style=MUTED,
        padding=(1, 2),
        box=panel_box(),
    )


def progress_panel(progress: RenderableType) -> Panel:
    arrow = glyph("➤", ">")
    return Panel(
        progress,
        title=f"[{ACCENT_BOLD}]{arrow} Progress[/]",
        title_align="left",
        border_style=ACCENT,
        padding=(1, 2),
        box=panel_box(),
    )


# ---------------------------------------------------------------------------
# Job tables
# ---------------------------------------------------------------------------
def _jobs_table(rows: list[tuple[str, str, str]], *, limit: int, empty_hint: bool) -> Table:
    table = Table(
        expand=True,
        border_style=MUTED,
        header_style=ACCENT_BOLD,
        box=table_box(),
        pad_edge=False,
    )
    table.add_column("File", overflow="fold", ratio=3)
    table.add_column("Time", justify="right", ratio=1)
    table.add_column("Status", justify="center", ratio=1)
    for file_name, elapsed, status in rows[-limit:]:
        table.add_row(str(file_name), str(elapsed), str(status))
    if not rows and empty_hint:
        table.add_row("[dim]waiting for first result…[/]", "[dim]—[/]", status_label("running"))
    return table


def recent_jobs_panel(rows: list[tuple[str, str, str]], *, limit: int = 10) -> Panel:
    book = glyph("▤", "=")
    return Panel(
        _jobs_table(rows, limit=limit, empty_hint=True),
        title=f"[{ACCENT_BOLD}]{book} Recent Completions[/]",
        title_align="left",
        border_style=ACCENT,
        padding=(0, 1),
        box=panel_box(),
    )


def calculation_results_panel(
    rows: list[tuple[str, str, str]],
    *,
    title: str = "CALCULATION RESULTS",
) -> Panel:
    return Panel(
        _jobs_table(rows, limit=12, empty_hint=False),
        title=f"[{ACCENT_BOLD}]{title}[/]",
        title_align="left",
        border_style=ACCENT,
        padding=(0, 1),
        box=panel_box(),
    )


def failed_files_table(failures: list[tuple[str, str]]) -> Panel:
    cross = glyph("✗", "x")
    table = Table(
        expand=True,
        border_style="red",
        header_style="bold red",
        box=table_box(),
        pad_edge=False,
    )
    table.add_column("File", overflow="fold", ratio=1)
    table.add_column("Error", overflow="fold", ratio=2)
    for file_path, error in failures:
        table.add_row(str(file_path), str(error))
    return Panel(
        table,
        title=f"[bold red]{cross} Failed Files[/]",
        title_align="left",
        border_style="red",
        padding=(0, 1),
        box=panel_box(),
    )


# ---------------------------------------------------------------------------
# Summary panels
# ---------------------------------------------------------------------------
def summary_panel(rows: list[tuple[str, str]], *, title: str, tone: str = "ok") -> Panel:
    palette = {
        "ok": ("green", OK_STYLE, glyph("✓", "*")),
        "warn": ("yellow", WARN_STYLE, glyph("▲", "!")),
        "fail": ("red", FAIL_STYLE, glyph("✗", "x")),
    }
    border, accent, mark = palette.get(tone, palette["ok"])
    return Panel(
        _kv_grid(rows),
        title=f"[{accent}]{mark} {title}[/]",
        title_align="left",
        border_style=border,
        padding=(1, 2),
        box=panel_box(),
    )


def divider(label: str = "") -> Rule:
    return Rule(f"[{MUTED}]{label}[/]" if label else "", style=MUTED)


__all__ = [
    "brand_banner",
    "brand_panel",
    "key_value_panel",
    "specs_columns",
    "stat_tiles",
    "progress_panel",
    "recent_jobs_panel",
    "calculation_results_panel",
    "failed_files_table",
    "summary_panel",
    "status_label",
    "divider",
    "Group",
    "can_render_unicode",
]
