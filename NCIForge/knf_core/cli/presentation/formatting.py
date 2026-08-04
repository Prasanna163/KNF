from __future__ import annotations

import os
import sys

from rich import box as _box
from rich.console import Console

DISPLAY_NAME_LIMIT = 40
OUTPUT_PATH_DISPLAY_LIMIT = 72

# ---------------------------------------------------------------------------
# Shared visual language
# ---------------------------------------------------------------------------
# One accent palette so every panel, table and rule reads as the same product.
ACCENT = "cyan"
ACCENT_BOLD = "bold cyan"
BRAND_STYLE = "bold bright_cyan"
INFO_BORDER = "blue"
OK_STYLE = "bold green"
FAIL_STYLE = "bold red"
WARN_STYLE = "bold yellow"
MUTED = "dim"


def fmt_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def truncate_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def display_name(file_path: str) -> str:
    return truncate_middle(os.path.basename(str(file_path)), DISPLAY_NAME_LIMIT)


def display_path(path: str, max_len: int = OUTPUT_PATH_DISPLAY_LIMIT) -> str:
    if not path:
        return ""
    return truncate_middle(str(path), max_len)


_TERMINAL_CONFIGURED = False
_VT_ENABLED = False
# Whether box-drawing glyphs can be written to the console. Resolved ONCE from
# the real stdout in configure_terminal() and cached, because rich.live.Live
# swaps sys.stdout for a FileProxy whose ``.encoding`` is None mid-run — reading
# it per panel would make every frame drawn during Live fall back to ASCII.
_UNICODE_OK = True


def _enable_windows_vt_and_utf8() -> bool:
    """Switch the Windows console to UTF-8 and turn on ANSI/VT processing.

    Without this the console reports a legacy code page (e.g. cp1252), so Rich
    falls back to ASCII box characters (``+---+``) and, when virtual-terminal
    processing was never enabled, prints raw escape codes such as ``\x1b[34m``
    literally instead of applying color. Both are silently ignored off Windows
    or when stdout is redirected (``GetConsoleMode`` fails on a non-console).

    Returns True when virtual-terminal processing is confirmed active on the
    stdout handle, so callers can force Rich's modern (non-legacy) renderer.
    """
    if os.name != "nt":
        return True  # *nix terminals speak ANSI natively.
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False
    try:
        kernel32 = ctypes.windll.kernel32
    except Exception:
        return False

    # UTF-8 code page so Unicode box-drawing characters render (not mojibake).
    for setter in ("SetConsoleOutputCP", "SetConsoleCP"):
        try:
            getattr(kernel32, setter)(65001)
        except Exception:
            pass

    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    try:
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    except Exception:
        pass
    vt_ok = False
    for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
        try:
            handle = kernel32.GetStdHandle(handle_id)
            mode = wintypes.DWORD()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue  # redirected / not a real console: nothing to enable
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
            if handle_id == -11:  # confirm VT on the stdout handle specifically
                confirm = wintypes.DWORD()
                if kernel32.GetConsoleMode(handle, ctypes.byref(confirm)):
                    vt_ok = bool(confirm.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        except Exception:
            pass
    return vt_ok


def configure_terminal() -> None:
    """Prepare the terminal so Rich renders clean Unicode boxes and real colors.

    Idempotent: safe (and cheap) to call at every dashboard entry point. This
    must run *before* any :class:`rich.console.Console` is created so Rich
    detects the VT-capable console and emits ANSI instead of latching the
    legacy-Windows fallback.
    """
    global _TERMINAL_CONFIGURED, _VT_ENABLED, _UNICODE_OK
    if _TERMINAL_CONFIGURED:
        return
    _TERMINAL_CONFIGURED = True

    _VT_ENABLED = _enable_windows_vt_and_utf8()
    # Make Python encode Unicode box characters as UTF-8 bytes rather than
    # raising UnicodeEncodeError on a legacy code page.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Resolve Unicode capability now, from the real stdout, before Rich's Live
    # can replace it with an encoding-less proxy.
    _UNICODE_OK = _stream_can_encode(sys.stdout)


def supports_unicode_terminal(stream=None) -> bool:
    stream = stream or sys.stdout
    encoding = (getattr(stream, "encoding", None) or "").lower()
    if "utf" in encoding:
        return True
    return os.name != "nt"


def _stream_can_encode(stream) -> bool:
    """Whether ``stream``'s codec can represent box-drawing glyphs."""
    encoding = (getattr(stream, "encoding", None) or "").lower()
    if "utf" in encoding:
        return True
    try:
        "─│╭╮╰╯".encode(encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def can_render_unicode(stream=None) -> bool:
    """Return True when box-drawing glyphs can be written to the console.

    With no explicit ``stream`` this returns the value cached by
    :func:`configure_terminal` (resolved from the real stdout). That indirection
    is essential: :class:`rich.live.Live` swaps ``sys.stdout`` for a proxy whose
    ``.encoding`` is ``None`` while a dashboard is live, so re-probing the current
    stdout per panel would make everything drawn during Live degrade to ASCII.
    ``configure_terminal`` forces UTF-8, so this is almost always True; it is
    False only when we are stranded on a legacy code page that cannot represent
    ``─``/``╭``, where plain ASCII avoids mojibake.
    """
    if stream is None:
        return _UNICODE_OK
    return _stream_can_encode(stream)


def panel_box(stream=None):
    """Continuous rounded frame for panels, ASCII only as a last resort."""
    return _box.ROUNDED if can_render_unicode(stream) else _box.ASCII


def table_box(stream=None):
    """Clean heavy-ruled frame for tabular data."""
    return _box.SIMPLE_HEAVY if can_render_unicode(stream) else _box.ASCII


def status_label(kind: str, stream=None) -> str:
    """Rich-markup status chip used in the completed-jobs tables."""
    unicode_ok = can_render_unicode(stream)
    kind = (kind or "").strip().lower()
    if kind in {"success", "ok", "succeeded"}:
        return f"[{OK_STYLE}]{'✓ OK' if unicode_ok else 'OK'}[/]"
    if kind in {"failed", "fail", "error"}:
        return f"[{FAIL_STYLE}]{'✗ FAIL' if unicode_ok else 'FAIL'}[/]"
    if kind in {"stopped", "stop"}:
        return f"[{WARN_STYLE}]{'■ STOP' if unicode_ok else 'STOP'}[/]"
    return f"[{MUTED}]{'· running' if unicode_ok else 'running'}[/]"


def glyph(unicode_char: str, ascii_char: str, stream=None) -> str:
    """Pick ``unicode_char`` when the stream can render it, else ``ascii_char``."""
    return unicode_char if can_render_unicode(stream) else ascii_char


def force_modern_terminal() -> bool:
    """Whether to force Rich onto its modern VT renderer (not legacy Windows).

    Rich's Windows auto-detection frequently mislabels a ConPTY-backed console
    (Windows Terminal, VS Code, modern conhost) as *legacy*. In legacy mode Rich
    drives the cursor through the Win32 console API, which ConPTY does not honour
    for a :class:`rich.live.Live` region — so the dashboard is re-emitted on every
    refresh (the "repeating panels") and partial escape sequences leak (a stray
    ``m``). We sidestep that by forcing the modern path whenever we have a real,
    VT-capable console. ``NCIFORGE_LEGACY_TERM=1`` opts back into auto-detection.
    """
    if os.environ.get("NCIFORGE_LEGACY_TERM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return False  # redirected to a file/pipe: keep Rich's plain rendering
    if os.name != "nt":
        return True
    if _VT_ENABLED:  # configure_terminal confirmed ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return True
    return bool(
        os.environ.get("WT_SESSION")
        or os.environ.get("TERM_PROGRAM")
        or os.environ.get("ANSICON")
        or os.environ.get("ConEmuANSI", "").strip().lower() == "on"
    )


def build_console(**kwargs) -> Console:
    """Configure the terminal, then build a Console that renders true rounded boxes.

    On a real terminal we force Rich's modern renderer (``legacy_windows=False``,
    ``force_terminal=True``) so that:

    * frames stay continuous and rounded (``safe_box=False`` blocks the
      square/ASCII legacy substitution),
    * :class:`rich.live.Live` repaints in place instead of stacking copies, and
    * true colour is available.

    Runs the terminal setup regardless of which entry point launched the CLI.
    """
    configure_terminal()
    kwargs.setdefault("safe_box", False)
    if force_modern_terminal():
        kwargs.setdefault("force_terminal", True)
        kwargs.setdefault("legacy_windows", False)
        kwargs.setdefault("color_system", "truecolor")
    return Console(**kwargs)
