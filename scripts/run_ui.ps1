param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($env:PYTHON) { $python = $env:PYTHON }
elseif (Test-Path (Join-Path $root '.venv\Scripts\python.exe')) { $python = Join-Path $root '.venv\Scripts\python.exe' }
else { $python = 'python' }

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }
if (-not $env:PATHSNAP_API_URL) { $env:PATHSNAP_API_URL = 'http://localhost:8000' }
$hostAddress = if ($env:PATHSNAP_UI_HOST) { $env:PATHSNAP_UI_HOST } else { '0.0.0.0' }
$port = if ($env:PATHSNAP_UI_PORT) { $env:PATHSNAP_UI_PORT } else { '8501' }

Write-Host "Starting PathSnap UI on ${hostAddress}:${port}; API: $env:PATHSNAP_API_URL"
& $python -m streamlit run src/ui/app.py --server.address $hostAddress --server.port $port
exit $LASTEXITCODE
