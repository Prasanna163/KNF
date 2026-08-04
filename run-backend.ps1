$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$backendDir = Join-Path $repoRoot "NCIForge"
$python = Join-Path $backendDir ".venv-nciforge\Scripts\python.exe"
$healthUrl = "http://127.0.0.1:8000/health"

if (-not (Test-Path -LiteralPath $python)) {
    throw "F-F backend environment not found: $python"
}

try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    if ($health.status -eq "ok" -and $health.service -eq "NCIForge API") {
        Write-Host "NCIForge API is already healthy at http://127.0.0.1:8000"
        exit 0
    }
}
catch {
    # Nothing healthy is responding; continue to the explicit F-F launch.
}

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port 8000 is occupied by another service. Stop it before launching the F-F backend."
}

Push-Location $backendDir
try {
    & $python -m uvicorn knf_core.api:app --host 127.0.0.1 --port 8000
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
