[CmdletBinding()]
param(
    [string]$DataDir,
    [ValidateRange(0, 65535)][int]$ApiPort = 0,
    [ValidateRange(0, 65535)][int]$UiPort = 0
)

$ErrorActionPreference = 'Stop'
$CallerDirectory = (Get-Location).Path
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if ($ApiPort -eq 0) {
    $ApiPort = if ($env:PATHSNAP_API_PORT) { [int]$env:PATHSNAP_API_PORT } else { 8000 }
}
if ($UiPort -eq 0) {
    $UiPort = if ($env:PATHSNAP_UI_PORT) { [int]$env:PATHSNAP_UI_PORT } else { 8501 }
}
if ($ApiPort -lt 1 -or $ApiPort -gt 65535 -or $UiPort -lt 1 -or $UiPort -gt 65535) {
    throw 'PATHSNAP_API_PORT and PATHSNAP_UI_PORT must be between 1 and 65535.'
}

if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = $env:PATHSNAP_DATA_DIR
}
if ([string]::IsNullOrWhiteSpace($DataDir) -and
    (Test-Path (Join-Path $RepoRoot 'train')) -and
    (Test-Path (Join-Path $RepoRoot 'student_test'))) {
    $DataDir = $RepoRoot
}
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    throw 'Dataset root not found. Set PATHSNAP_DATA_DIR or pass -DataDir to a folder containing train/ and student_test/.'
}
if (-not [System.IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $CallerDirectory $DataDir
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "Dataset directory not found: $DataDir"
}
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
$TestDir = Join-Path $DataDir 'student_test'
if (-not (Test-Path (Join-Path $DataDir 'train')) -or -not (Test-Path $TestDir)) {
    throw "Dataset root must contain train/ and student_test/: $DataDir"
}
$TestTrips = @(Get-ChildItem -LiteralPath $TestDir -Directory)
if ($TestTrips.Count -ne 10) {
    throw "Expected 10 reserved student_test trips; found $($TestTrips.Count) under $TestDir"
}

$VenvPython = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw 'Virtual environment is missing. Run scripts/setup.ps1 first.'
}
$env:PATHSNAP_DATA_DIR = $DataDir
$env:PATHSNAP_CORS = if ($env:PATHSNAP_CORS) { $env:PATHSNAP_CORS } else { 'http://localhost:8501' }
$env:PATHSNAP_API_URL = "http://127.0.0.1:$ApiPort"
$SourcePath = Join-Path $RepoRoot 'src'
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$SourcePath;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $SourcePath
}
if (-not [string]::IsNullOrWhiteSpace($env:PATHSNAP_MODEL)) {
    if (-not [System.IO.Path]::IsPathRooted($env:PATHSNAP_MODEL)) {
        $env:PATHSNAP_MODEL = Join-Path $CallerDirectory $env:PATHSNAP_MODEL
    }
    if (-not (Test-Path -LiteralPath $env:PATHSNAP_MODEL -PathType Leaf)) {
        throw "PATHSNAP_MODEL does not exist: $env:PATHSNAP_MODEL"
    }
    $env:PATHSNAP_MODEL = (Resolve-Path -LiteralPath $env:PATHSNAP_MODEL).Path
}
& $VenvPython (Join-Path $RepoRoot 'scripts/validate_dataset.py') --data-dir $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed with exit code $LASTEXITCODE"
}

$ApiProcess = $null
try {
    $ApiProcess = Start-Process -FilePath $VenvPython -ArgumentList @(
        '-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', "$ApiPort"
    ) -WorkingDirectory $RepoRoot -PassThru -NoNewWindow

    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        if ($ApiProcess.HasExited) {
            throw "FastAPI exited during startup with code $($ApiProcess.ExitCode)"
        }
        try {
            $Health = Invoke-RestMethod -Uri "$($env:PATHSNAP_API_URL)/health" -TimeoutSec 2
            $Ready = $true
            if (-not $Health.model_loaded) {
                Write-Warning 'API is reachable, but no model bundle is loaded; predictions will return HTTP 503.'
            }
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $Ready) {
        throw "FastAPI did not become reachable at $($env:PATHSNAP_API_URL)/health"
    }

    Write-Host "PathSnap API ready at $($env:PATHSNAP_API_URL)"
    Write-Host "Starting Streamlit at http://localhost:$UiPort; press Ctrl+C to stop both services."
    & $VenvPython -m streamlit run src/ui/app.py --server.address 127.0.0.1 --server.port $UiPort
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE"
    }
} finally {
    if ($ApiProcess -and -not $ApiProcess.HasExited) {
        Stop-Process -Id $ApiProcess.Id -Force -ErrorAction SilentlyContinue
        try {
            Wait-Process -Id $ApiProcess.Id -Timeout 10 -ErrorAction SilentlyContinue
        } catch {
            Write-Warning "Timed out waiting for FastAPI process $($ApiProcess.Id) to stop."
        }
    }
}
