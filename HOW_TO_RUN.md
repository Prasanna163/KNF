# KNF Studio — How to Run on Localhost

> **What is this?**  
> KNF Studio is a desktop app (Electron + React) backed by a Python FastAPI server.  
> You need to run **both** the backend and the frontend together.

---

## Recommended: Install the Windows App

For normal use, run:

```text
frontend\dist-installer\NCIForge-Setup-1.0.0.exe
```

The installer includes the frontend, backend, private Python runtime, and CPU
PyTorch. No separate Python or Node.js setup is needed.

If an NVIDIA GPU is detected, setup offers an optional CUDA 12.8 PyTorch
download. Choose **Yes** for GPU jobs or **No** to keep CPU Torch. CUDA setup
requires internet access and several gigabytes of free space.

The remaining instructions below are for development from source.

---

## Prerequisites — Install These First (One Time Only)

| Tool | Where to get it | Check with |
|------|----------------|------------|
| Python 3.10 or 3.11 | https://python.org → Download, tick "Add to PATH" | `python --version` |
| Node.js 18+ | https://nodejs.org → LTS version | `node --version` |
| Git | https://git-scm.com | `git --version` |

> **Important for Python:** When installing, **tick the checkbox** that says  
> ✅ "Add Python to PATH" on the first page of the installer.

---

## Step 1 — Clone the Repo

Open **PowerShell** or **Command Prompt** and run:

```powershell
git clone https://github.com/Uttkarsh779/NCI-FORGE-F-F.git
cd NCI-FORGE-F-F
```

---

## Step 2 — Set Up the Python Backend

Open a **new terminal window** (keep this one open — this runs the server).

```powershell
# Go into the backend folder
cd NCIForge

# Create a Python virtual environment
python -m venv .venv-nciforge

# Activate the virtual environment
.venv-nciforge\Scripts\activate

# Install the backend + API dependencies
pip install -e ".[api]"

# Install the CUDA-enabled PyTorch runtime used by GPU jobs
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> 💡 The `.[api]` installs FastAPI, Uvicorn, and multipart upload support.
> PyTorch is installed separately from the official CUDA 12.8 wheel index so
> the app's GPU mode uses the NVIDIA GPU instead of failing at run time.
> This takes 2–5 minutes on first run.

---

## Step 3 — Backend Launching

The Electron desktop app starts the backend automatically on port `8000`.
Do not start a second backend for normal desktop development.

If a healthy NCIForge API is already running on port `8000`, Electron reuses
it instead of killing and replacing it.

---

## Step 4 — Set Up the Frontend

Open a terminal window.

```powershell
# Go into the frontend folder
cd NCI-FORGE-F-F\frontend

# Install Node.js dependencies
npm install
```

> This takes 1–3 minutes on first run (downloads ~500 MB of packages).

---

## Step 5 — Launch the Desktop App

Still in the `frontend/` folder:

```powershell
npm run dev
```

This will:
1. Start the Vite development server (React UI)
2. Start or reuse the NCIForge API on port 8000
3. Wait for both services to be ready
4. Launch the Electron desktop window automatically

✅ A native desktop window titled **"KNF Studio"** will appear.

---

## Quick Reference — All Commands in Order

From the `MAGGIE` folder, launch the complete F-F app with:

```powershell
.\NCI-FORGE-F-F\run.ps1
```

For the API alone, use:

```powershell
.\NCI-FORGE-F-F\run-backend.ps1
```

These launchers use the F-F virtual environment explicitly. Do not run the
global `uvicorn knf_core.api:app` command from the `MAGGIE` folder because
other editable `knf_core` installations on the machine can be imported.

---

## Troubleshooting

### "python is not recognized"
→ Reinstall Python from https://python.org and tick **"Add Python to PATH"**

### "npm is not recognized"
→ Install Node.js from https://nodejs.org (LTS version)

### "Module not found: knf_core"
→ Make sure you ran `pip install -e ".[api]"` inside `NCIForge/` with the venv activated

### Backend shows error about `xtb` or `obabel`
→ These are optional external chemistry tools. The basic app runs without them.  
→ For full functionality, see the NCIForge README.md in the NCIForge/ folder.

### Port 8000 already in use
```powershell
# Find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### The Electron window doesn't open
→ Make sure `npm install` completed successfully first  
→ Check that the backend (Window 1) is running and showing "Uvicorn running on..."

---

## Folder Structure

```
NCI-FORGE-F-F/
├── NCIForge/          ← Python backend (FastAPI + KNF engine)
│   ├── knf_core/      ← Core computation library
│   ├── scripts/       ← Install helpers
│   └── setup.py       ← Python package definition
├── frontend/          ← Electron + React desktop UI
│   ├── src/           ← React source code
│   ├── electron/      ← Electron main process
│   └── package.json   ← Node dependencies
└── HOW_TO_RUN.md      ← This file
```
