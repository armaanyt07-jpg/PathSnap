param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($env:PYTHON) { $python = $env:PYTHON }
elseif (Test-Path (Join-Path $root '.venv\Scripts\python.exe')) { $python = Join-Path $root '.venv\Scripts\python.exe' }
else { $python = 'python' }

if (-not $env:PATHSNAP_DATA_DIR) { $env:PATHSNAP_DATA_DIR = $root }
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }
$hostAddress = if ($env:PATHSNAP_API_HOST) { $env:PATHSNAP_API_HOST } else { '0.0.0.0' }
$port = if ($env:PATHSNAP_API_PORT) { $env:PATHSNAP_API_PORT } else { '8000' }

Write-Host "Starting PathSnap API at http://${hostAddress}:${port}"
Write-Host "Data root: $env:PATHSNAP_DATA_DIR"
& $python -m uvicorn api.main:app --host $hostAddress --port $port
exit $LASTEXITCODE
