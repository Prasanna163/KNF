from __future__ import annotations


def normalize_argv(argv: list[str]) -> list[str]:
    """Normalize legacy argparse conveniences before Typer parses argv.

    The public CLI supports two argparse-era conveniences that Click/Typer do
    not model directly: `knf full <path>` and a bare `--batches` flag whose
    implicit value is auto mode (`0`). Keep both rewrites framework-neutral so
    they are easy to test and reuse.
    """
    args = list(argv)
    if args and args[0] == "full":
        args.pop(0)

    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        item = args[idx]
        normalized.append(item)
        if item == "--batches":
            next_item = args[idx + 1] if idx + 1 < len(args) else None
            if next_item is None or next_item.startswith("-"):
                normalized.append("0")
        idx += 1
    return normalized
