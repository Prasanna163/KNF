# NCI-FORGE-FRONTEND

Electron + React desktop UI ("KNF Studio") for the NCIForge backend included
in the parent `NCI-FORGE-F-F` repository.

The frontend talks to the job-oriented `knf_core.api:app` service at
`http://127.0.0.1:8000`. In development, Electron starts that backend from
`../NCIForge/.venv-nciforge`, or reuses an already healthy F-F API on port
8000. Do not launch a global `uvicorn` command from the `MAGGIE` directory;
other editable `knf_core` installations on this machine can be imported.

The authoritative complete-app instructions are in the parent
[`README.md`](../README.md). See
[`COMPATIBILITY_CHANGES.md`](../COMPATIBILITY_CHANGES.md) for the API adaptation
history.

## Development

From the parent repository, the preferred command is:

```powershell
.\run.ps1
```

Or from this directory:

```powershell
npm install
npm run dev
```

Electron will start or reuse the F-F backend automatically.

## Build

```powershell
npm run build
```

To build the complete 64-bit Windows installer:

```powershell
npm run electron:dist
```

This command:

1. Builds the React and Electron application.
2. Creates or reuses a source-fingerprinted private Python 3.11 runtime.
3. Bundles CPU-only PyTorch as the universal fallback.
4. Adds an NVIDIA-gated installer prompt for the optional CUDA 12.8 PyTorch
   layer.
5. Produces `dist-installer\NCIForge-Setup-1.0.0.exe`.

The release ZIP `dist-installer\NCIForge-1.0.0-Windows-x64.zip` contains the
installer, blockmap, integrity checksums, installation guide, and license.

The generated backend runtime and installer artifacts are intentionally
ignored by Git. They can be reproduced from the checked-in build scripts.

The installer is currently unsigned; production distribution should add a
Windows code-signing certificate to avoid an Unknown Publisher warning.
