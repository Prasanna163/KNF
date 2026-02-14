import io
import json
import zipfile
from pathlib import Path


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def zip_directory(path: str) -> bytes:
    buf = io.BytesIO()
    target = Path(path)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if target.exists():
            for file in target.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(target))
    buf.seek(0)
    return buf.read()

