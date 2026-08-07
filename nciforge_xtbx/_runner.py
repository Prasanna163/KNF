"""Script-file runner used by subprocess calls from arbitrary working dirs."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nciforge_xtbx.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
