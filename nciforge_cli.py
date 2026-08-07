"""Console-script bootstrap for the local NCIForge checkout.

The user environment can have multiple editable projects that expose a
top-level ``knf_core`` package.  Console scripts start outside the repository,
so Python may import another project's ``knf_core`` before this distribution's
editable finder gets a chance.  Route through this unique module first, place
this checkout at the front of ``sys.path``, then import the real entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent


def _prefer_local_checkout() -> None:
    repo_root = str(_REPO_ROOT)
    if sys.path[:1] != [repo_root]:
        try:
            sys.path.remove(repo_root)
        except ValueError:
            pass
        sys.path.insert(0, repo_root)

    loaded = sys.modules.get("knf_core")
    loaded_file = getattr(loaded, "__file__", "") if loaded is not None else ""
    local_loaded = False
    if loaded_file:
        try:
            Path(loaded_file).resolve().relative_to(_REPO_ROOT)
            local_loaded = True
        except ValueError:
            local_loaded = False
    if loaded_file and not local_loaded:
        for name in list(sys.modules):
            if name == "knf_core" or name.startswith("knf_core."):
                del sys.modules[name]


def main() -> int | None:
    _prefer_local_checkout()
    from knf_core.main import main as run_main

    return run_main()


def api_main() -> int | None:
    _prefer_local_checkout()
    from knf_core.api import main as run_api

    return run_api()


def geoinit_main() -> int | None:
    _prefer_local_checkout()
    from geoinit.cli.main import main as run_geoinit

    return run_geoinit()
