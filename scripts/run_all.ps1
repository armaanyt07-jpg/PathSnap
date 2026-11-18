param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($env:PYTHON) { $python = $env:PYTHON }
elseif (Test-Path (Join-Path $root '.venv\Scripts\python.exe')) { $python = Join-Path $root '.venv\Scripts\python.exe' }
else { $python = 'python' }

if (-not $env:PATHSNAP_DATA_DIR) { $env:PATHSNAP_DATA_DIR = $root }
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }
$apiPort = if ($env:PATHSNAP_API_PORT) { $env:PATHSNAP_API_PORT } else { '8000' }
$uiPort = if ($env:PATHSNAP_UI_PORT) { $env:PATHSNAP_UI_PORT } else { '8501' }
$env:PATHSNAP_API_URL = if ($env:PATHSNAP_API_URL) { $env:PATHSNAP_API_URL } else { "http://127.0.0.1:$apiPort" }
$uiHost = if ($env:PATHSNAP_UI_HOST) { $env:PATHSNAP_UI_HOST } else { '127.0.0.1' }
$apiLog = Join-Path $env:TEMP 'pathsnap-api.log'
$apiErr = Join-Path $env:TEMP 'pathsnap-api-error.log'
$apiProcess = $null

try {
    $apiProcess = Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','api.main:app','--host','127.0.0.1','--port',$apiPort) -WorkingDirectory $root -PassThru -WindowStyle Hidden -RedirectStandardOutput $apiLog -RedirectStandardError $apiErr
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        if ($apiProcess.HasExited) { throw "API exited early. See $apiErr" }
        try {
            $health = Invoke-RestMethod -Uri "$($env:PATHSNAP_API_URL)/health" -TimeoutSec 2
            if ($health.status -eq 'ok') { $ready = $true; break }
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw "FastAPI did not become ready at $($env:PATHSNAP_API_URL)/health. See $apiErr" }
    Write-Host "API is ready. Starting Streamlit; press Ctrl+C to stop both services."
    & $python -m streamlit run src/ui/app.py --server.address $uiHost --server.port $uiPort
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    if ($apiProcess -and -not $apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue }
}
