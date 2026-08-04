# KNF Studio / NCIForge F-F

KNF Studio is the Electron + React desktop interface for the NCIForge
non-covalent-interaction descriptor engine in this repository.

This is the active integrated checkout:

```text
NCI-FORGE-F-F/
├── frontend/       Electron + React application
├── NCIForge/       Python engine and FastAPI backend
├── run.ps1         Canonical complete-app launcher
└── run-backend.ps1 Canonical API-only launcher
```

The sibling `MAGGIE\NCIForge` folder is a separate repository with a different
frontend/backend contract and port. Do not combine files or launch commands
from the two projects.

## Quick Start

From `E:\Prasanna\NICForge-Ui\MAGGIE`:

```powershell
.\NCI-FORGE-F-F\run.ps1
```

This starts the Vite frontend and Electron. Electron starts the F-F API on
`http://127.0.0.1:8000`, or reuses it when the expected NCIForge API is already
healthy there.

To run only the API:

```powershell
.\NCI-FORGE-F-F\run-backend.ps1
```

Do not run the global command below from the `MAGGIE` directory:

```powershell
uvicorn knf_core.api:app --host 127.0.0.1 --port 8000
```

This machine has other editable projects that expose a `knf_core` package.
Global Python can therefore combine modules from BioForge or the sibling
NCIForge checkout. The supplied launchers always use:

```text
NCI-FORGE-F-F\NCIForge\.venv-nciforge\Scripts\python.exe
```

## One-Time Setup

Backend:

```powershell
cd E:\Prasanna\NICForge-Ui\MAGGIE\NCI-FORGE-F-F\NCIForge
python -m venv .venv-nciforge
.\.venv-nciforge\Scripts\python.exe -m pip install -e ".[api]"
.\.venv-nciforge\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Frontend:

```powershell
cd E:\Prasanna\NICForge-Ui\MAGGIE\NCI-FORGE-F-F\frontend
npm install
```

Full computations also require `xtb` and `obabel` on `PATH`. Multiwfn is only
required when selecting the Multiwfn NCI backend.

## Windows Installer

Build the complete 64-bit Windows installer from `frontend`:

```powershell
cd E:\Prasanna\NICForge-Ui\MAGGIE\NCI-FORGE-F-F\frontend
npm run electron:dist
```

The distributable is written to:

```text
frontend\dist-installer\NCIForge-Setup-1.0.0.exe
```

The ready-to-share release archive is:

```text
frontend\dist-installer\NCIForge-1.0.0-Windows-x64.zip
```

It contains the installer, blockmap, SHA-256 checksums, installation guide,
and license.

The setup executable installs one main `NCIForge.exe`, Start Menu and desktop
shortcuts, an uninstaller, and a private Python 3.11 backend. Users do not need
Node.js, Python, or PyTorch installed globally.

Every installation includes CPU-only PyTorch. On first interactive install,
the installer checks `nvidia-smi`. When a supported NVIDIA GPU is present, it
offers CUDA 12.8 PyTorch as an optional per-user download. Declining the
download, installing without an NVIDIA GPU, running a silent install, or a
CUDA setup failure leaves the bundled CPU runtime active.

CUDA packages are stored under:

```text
%LOCALAPPDATA%\NCIForge\runtime\cuda-site-packages
```

The uninstaller removes this optional CUDA layer. CUDA installation requires
an internet connection and several gigabytes of free disk space.

## Runtime Contract

The frontend and backend use the job-oriented API:

- `GET /health`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/input`
- `GET /jobs/{job_id}/download/{artifact_name}`
- `POST /jobs/path`
- `POST /jobs/upload`
- `DELETE /jobs/{job_id}`

Default development addresses:

| Service | Address |
|---|---|
| Vite frontend | `http://127.0.0.1:5173` |
| NCIForge API | `http://127.0.0.1:8000` |
| API documentation | `http://127.0.0.1:8000/docs` |

## Repairs Applied on 2026-07-29/30

- Installed and verified CUDA-enabled PyTorch `2.11.0+cu128` in the F-F venv.
- Verified CUDA execution on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Prevented API worker threads from opening interactive CUDA-install prompts.
  A missing CUDA runtime now becomes a clear API error instead of
  `EOF when reading a line`.
- Removed Uvicorn auto-reload from Electron's backend process.
- Made Electron reuse a healthy F-F API on port 8000 instead of killing it.
- Corrected `frontend/scripts/run-backend.mjs` to use the F-F backend,
  `knf_core.api:app`, and port 8000.
- Added root `run.ps1` and `run-backend.ps1` launchers that explicitly use the
  isolated F-F environment.
- Made stale job IDs terminal failures in the UI instead of polling 404s
  indefinitely after a backend restart.
- Distinguished real HTTP 404 responses from temporary network failures so
  transient outages remain retryable.
- Added regression coverage for non-interactive CUDA preflight and stale jobs.
- Replaced the broken frozen-backend packaging path with an embedded private
  Python runtime containing verified CPU PyTorch.
- Added the NVIDIA-gated optional CUDA PyTorch installer flow and branded NSIS
  setup assets from `E:\Prasanna\NICForge-Ui\Logos`.

## Verified Result

The molecule that previously failed immediately with `EOF when reading a line`
was rerun using the same CUDA configuration:

```text
Input:   Benzoic_Acid--Ethyl_Acetate.xyz
Job ID:  72e82ad4b09445e081e483b1c310b8cc
Status:  succeeded
Runtime: 56.39 seconds
NCI:     Torch CUDA, 2.23 seconds
Result:  E:\Prasanna\New folder\Benzoic_Acid--Ethyl_Acetate\knf.json
```

Validation completed:

- Focused API tests: `4 passed`
- Frontend regression tests: `2 passed`
- Frontend production build: passed
- Embedded CPU runtime API smoke test: passed
- Packaged Electron application startup: passed
- Optional CUDA layer selection and CUDA tensor test: passed
- Branded NSIS installer build: passed
- Full backend suite: `81 passed`, `2 skipped`
- Four additional backend tests currently fail because this checkout lacks
  their expected root-level `molden.input` fixture; the live end-to-end
  calculation is not affected.

## Current Limitations

- API jobs are stored in memory. Restarting the backend invalidates old job
  IDs, while the frontend's run grouping remains in browser `localStorage`.
- The API runs one queued job at a time by default
  (`NCIFORGE_API_WORKERS=1`).
- Active job cancellation and live WebSocket log streaming are not available.
- The generated installer is not digitally signed. Windows SmartScreen can
  show an "Unknown publisher" warning until a code-signing certificate is
  configured.

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for expanded setup instructions and
[COMPATIBILITY_CHANGES.md](COMPATIBILITY_CHANGES.md) for the frontend/backend
adaptation history.
