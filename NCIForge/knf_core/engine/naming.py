import os


def _final_output_name(filename: str, water: bool) -> str:
    if not water:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_water{ext}"
