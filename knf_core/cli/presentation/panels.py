from __future__ import annotations

import sys

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...engine.constants import CLI_NAME, CLI_SUBTITLE, CLI_VERSION
from .formatting import supports_unicode_terminal


def brand_panel() -> Panel:
    title = Text(f"{CLI_NAME} {CLI_VERSION}", style="bold cyan")
    subtitle = Text(CLI_SUBTITLE, style="dim")
    table = Table.grid(padding=(0, 1))
    table.add_row(title)
    table.add_row(subtitle)
    return Panel(
        table,
        border_style="cyan",
        box=(box.ROUNDED if supports_unicode_terminal(sys.stdout) else box.ASCII),
        padding=(0, 1),
    )


def key_value_panel(
    rows: list[tuple[str, str]],
    *,
    title: str,
    border_style: str = "cyan",
) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    for key, value in rows:
        table.add_row(str(key), str(value))
    return Panel(
        table,
        title=title,
        title_align="left",
        border_style=border_style,
        padding=(0, 1),
        box=(box.ROUNDED if supports_unicode_terminal(sys.stdout) else box.ASCII),
    )


def calculation_results_panel(
    rows: list[tuple[str, str, str]],
    *,
    title: str = "CALCULATION RESULTS",
) -> Panel:
    table = Table(expand=True, box=(box.SIMPLE if supports_unicode_terminal(sys.stdout) else box.ASCII))
    table.add_column("File", overflow="fold")
    table.add_column("Time", justify="right")
    table.add_column("Status", justify="center")
    for file_name, elapsed, status in rows[-12:]:
        table.add_row(str(file_name), str(elapsed), str(status))
    return Panel(
        table,
        title=title,
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
        box=(box.ROUNDED if supports_unicode_terminal(sys.stdout) else box.ASCII),
    )


def failed_files_table(failures: list[tuple[str, str]]) -> Table:
    table = Table(
        title="FAILED FILES",
        expand=True,
        box=(box.SIMPLE_HEAVY if supports_unicode_terminal(sys.stdout) else box.ASCII),
    )
    table.add_column("File")
    table.add_column("Error")
    for file_path, error in failures:
        table.add_row(str(file_path), str(error))
    return table
