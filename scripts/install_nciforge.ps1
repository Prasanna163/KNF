param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
& $PythonExe ".\scripts\install_nciforge_cli.py"
