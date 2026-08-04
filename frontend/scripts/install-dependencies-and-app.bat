@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo KNF Studio - Dependency & Application Installer
echo ===================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3 was not found on your system.
    echo Attempting to install Python 3.11 via winget...
    winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Failed to install Python via winget.
        echo Please install Python 3.11 manually from https://python.org
        echo and ensure you check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    echo Python installed successfully!
)

:: 2. Create directory for virtual environment
set "VENV_DIR=%USERPROFILE%\.knf-studio-venv"
echo Setting up Python virtual environment at: %VENV_DIR%
if not exist "%VENV_DIR%" (
    mkdir "%VENV_DIR%"
)

python -m venv "%VENV_DIR%"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to create Python virtual environment.
    pause
    exit /b 1
)

:: 3. Install packages inside virtual environment
echo Installing molecular descriptor engine (nciforge + PyTorch)...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
"%VENV_DIR%\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"%VENV_DIR%\Scripts\python.exe" -m pip install nciforge
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install Python packages.
    pause
    exit /b 1
)

:: 4. Register environment variable persistently so KNF Studio can find it
echo Registering environment variables...
setx KNF_STUDIO_VENV "%VENV_DIR%" >nul

:: 5. Run the generated Electron Installer
if exist "KNFStudio-Setup-1.0.0.exe" (
    echo Starting KNF Studio Installer...
    start "" "KNFStudio-Setup-1.0.0.exe"
) else (
    echo [WARNING] KNFStudio-Setup-1.0.0.exe was not found in this folder.
    echo Please run the installer manually after this setup.
)

echo.
echo === Dependencies setup successfully! ===
echo.
pause
