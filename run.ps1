$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$frontendDir = Join-Path $repoRoot "frontend"
$backendPython = Join-Path $repoRoot "NCIForge\.venv-nciforge\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw "F-F backend environment not found: $backendPython"
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "node_modules"))) {
    throw "Frontend dependencies are missing. Run npm install inside: $frontendDir"
}

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
}
if (-not $npm) {
    throw "npm was not found. Install Node.js or add npm to PATH."
}

Push-Location $frontendDir
try {
    & $npm.Source run dev
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
