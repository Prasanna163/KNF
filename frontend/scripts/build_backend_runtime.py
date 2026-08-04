"""Build the private Python runtime embedded in the Windows app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path


PYTHON_VERSION = "3.11.9"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
TORCH_VERSION = "2.11.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"[bundle-backend] Reusing cached {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[bundle-backend] Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(destination)


def _configure_embedded_python(runtime_dir: Path) -> Path:
    pth_files = list(runtime_dir.glob("python*._pth"))
    if len(pth_files) != 1:
        raise RuntimeError(f"Expected one embedded-Python _pth file, found {pth_files}")

    pth_files[0].write_text(
        "python311.zip\n"
        ".\n"
        "Lib\\site-packages\n"
        "import site\n",
        encoding="utf-8",
    )

    site_packages = runtime_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "sitecustomize.py").write_text(
        '"""NCIForge managed runtime path selection."""\n'
        "import os\n"
        "import sys\n"
        "\n"
        'cuda_site = os.environ.get("NCIFORGE_CUDA_SITE_PACKAGES", "").strip()\n'
        "if cuda_site and os.path.isdir(cuda_site):\n"
        "    normalized = os.path.normcase(os.path.abspath(cuda_site))\n"
        "    existing = {os.path.normcase(os.path.abspath(p or '.')) for p in sys.path}\n"
        "    if normalized not in existing:\n"
        "        sys.path.insert(0, cuda_site)\n",
        encoding="utf-8",
    )
    return site_packages


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(backend_dir: Path) -> str:
    digest = hashlib.sha256()
    roots = [
        backend_dir / "knf_core",
        backend_dir / "nciforge_xtbx",
        backend_dir / "nciforge_cli.py",
        backend_dir / "setup.py",
        backend_dir / "pyproject.toml",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                item
                for item in root.rglob("*")
                if item.is_file()
                and item.suffix.lower() in {".py", ".json", ".txt", ".conf", ".sh", ".exe", ".dll"}
                and "__pycache__" not in item.parts
            )
    for item in sorted(files, key=lambda p: p.as_posix().lower()):
        relative = item.relative_to(backend_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build(repo_root: Path, output_dir: Path) -> None:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    expected_parent = (repo_root / "frontend" / "resources").resolve()
    if expected_parent not in output_dir.parents:
        raise RuntimeError(f"Refusing to build outside {expected_parent}: {output_dir}")

    backend_dir = repo_root / "NCIForge"
    helper_source = repo_root / "frontend" / "scripts" / "install_cuda_torch.py"
    cache_dir = repo_root / "frontend" / ".bundle-cache"
    embed_zip = cache_dir / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    source_fingerprint = _source_fingerprint(backend_dir)

    existing_manifest_path = output_dir / "runtime-manifest.json"
    existing_python = output_dir / "runtime" / "python.exe"
    existing_helper = output_dir / "install_cuda_torch.py"
    if existing_manifest_path.exists() and existing_python.exists() and existing_helper.exists():
        try:
            existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if (
            existing.get("python_version") == PYTHON_VERSION
            and existing.get("torch_version") == TORCH_VERSION
            and existing.get("torch_variant") == "cpu"
            and existing.get("source_fingerprint") == source_fingerprint
        ):
            print("[bundle-backend] Reusing verified runtime; source fingerprint is unchanged.")
            return

    _download(PYTHON_EMBED_URL, embed_zip)

    staging = output_dir.parent / f".backend-staging-{uuid.uuid4().hex}"
    runtime_dir = staging / "runtime"
    try:
        runtime_dir.mkdir(parents=True)
        with zipfile.ZipFile(embed_zip) as archive:
            archive.extractall(runtime_dir)

        site_packages = _configure_embedded_python(runtime_dir)
        common_pip = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--ignore-installed",
            "--no-compile",
            "--target",
            str(site_packages),
        ]

        _run([*common_pip, "pip"], cwd=backend_dir)
        _run(
            [
                *common_pip,
                "--prefer-binary",
                ".",
                "fastapi",
                "uvicorn[standard]",
                "python-multipart",
                "matplotlib",
                "pandas",
                f"torch=={TORCH_VERSION}+cpu",
                "--extra-index-url",
                TORCH_CPU_INDEX,
            ],
            cwd=backend_dir,
        )

        shutil.copy2(helper_source, staging / "install_cuda_torch.py")

        verify_code = (
            "import json, torch, fastapi, uvicorn, knf_core.api\n"
            "print(json.dumps({"
            "'torch': torch.__version__, "
            "'cuda_build': torch.version.cuda, "
            "'cuda_available': torch.cuda.is_available(), "
            "'api': knf_core.api.app.title"
            "}))\n"
        )
        verify_env = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
        proc = subprocess.run(
            [str(runtime_dir / "python.exe"), "-c", verify_code],
            cwd=runtime_dir,
            env=verify_env,
            capture_output=True,
            text=True,
            check=True,
        )
        verification = json.loads(proc.stdout.strip().splitlines()[-1])
        if verification.get("cuda_build") is not None:
            raise RuntimeError(f"CPU runtime unexpectedly contains CUDA Torch: {verification}")

        manifest = {
            "schema": 1,
            "python_version": PYTHON_VERSION,
            "python_embed_url": PYTHON_EMBED_URL,
            "python_embed_sha256": _sha256(embed_zip),
            "torch_version": TORCH_VERSION,
            "torch_variant": "cpu",
            "torch_index_url": TORCH_CPU_INDEX,
            "source_fingerprint": source_fingerprint,
            "runtime_bytes": _directory_size(runtime_dir),
            "verification": verification,
        }
        (staging / "runtime-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        # os.replace/Path.replace can return WinError 5 for a populated
        # directory even on the same volume (antivirus/indexer handles are
        # enough to trigger it). copytree is slower but reliable for release
        # packaging and keeps the destination complete.
        shutil.copytree(staging, output_dir)
        print(
            "[bundle-backend] Runtime complete: "
            f"{manifest['runtime_bytes'] / 1024 / 1024:.0f} MiB"
        )
        print(json.dumps(verification, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root, args.output)


if __name__ == "__main__":
    main()
