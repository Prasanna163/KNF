from __future__ import annotations

import os
import sys

DISPLAY_NAME_LIMIT = 40
OUTPUT_PATH_DISPLAY_LIMIT = 72


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


def supports_unicode_terminal(stream=None) -> bool:
    stream = stream or sys.stdout
    encoding = (getattr(stream, "encoding", None) or "").lower()
    if "utf" in encoding:
        return True
    return os.name != "nt"
