"""Generate Windows/Electron branding assets from the approved NCIForge logos."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def _contain(source: Image.Image, size: tuple[int, int], margin: int = 0) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    available = (max(1, size[0] - margin * 2), max(1, size[1] - margin * 2))
    fitted = ImageOps.contain(source.convert("RGB"), available, Image.Resampling.LANCZOS)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def generate(source_dir: Path, public_dir: Path) -> None:
    app_source = source_dir / "App.png"
    horizontal_source = source_dir / "horizontal logo.png"
    for source in (app_source, horizontal_source):
        if not source.exists():
            raise FileNotFoundError(source)

    public_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(app_source) as app_image, Image.open(horizontal_source) as horizontal:
        icon = _contain(app_image, (512, 512))
        icon.save(public_dir / "icon.png", optimize=True)
        icon.save(
            public_dir / "icon.ico",
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        icon.save(
            public_dir / "favicon.ico",
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )

        header = _contain(horizontal, (150, 57), margin=3)
        header.save(public_dir / "installer-header.bmp", format="BMP")

        sidebar = _contain(app_image, (164, 314), margin=8)
        sidebar.save(public_dir / "installer-sidebar.bmp", format="BMP")

    print(f"Generated branded assets in {public_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source_dir.resolve(), args.public_dir.resolve())


if __name__ == "__main__":
    main()
