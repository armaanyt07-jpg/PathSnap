[CmdletBinding()]
param(
    [string]$DataDir,
    [string]$HostAddress,
    [ValidateRange(0, 65535)][int]$Port = 0,
    [string]$ModelPath,
    [string]$CorsOrigins
)

$ErrorActionPreference = 'Stop'
$CallerDirectory = (Get-Location).Path
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if ([string]::IsNullOrWhiteSpace($HostAddress)) {
    $HostAddress = if ($env:PATHSNAP_API_HOST) { $env:PATHSNAP_API_HOST } else { '0.0.0.0' }
}
if ($Port -eq 0) {
    $Port = if ($env:PATHSNAP_API_PORT) { [int]$env:PATHSNAP_API_PORT } else { 8000 }
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "PATHSNAP_API_PORT must be between 1 and 65535; got $Port"
}
if ([string]::IsNullOrWhiteSpace($CorsOrigins)) {
    $CorsOrigins = if ($env:PATHSNAP_CORS) { $env:PATHSNAP_CORS } else { 'http://localhost:8501' }
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
$TrainDir = Join-Path $DataDir 'train'
$TestDir = Join-Path $DataDir 'student_test'
if (-not (Test-Path $TrainDir -PathType Container) -or -not (Test-Path $TestDir -PathType Container)) {
    throw "Dataset root must contain train/ and student_test/: $DataDir"
}
$TestTrips = @(Get-ChildItem -LiteralPath $TestDir -Directory)
if ($TestTrips.Count -ne 10) {
    throw "Expected 10 reserved student_test trips; found $($TestTrips.Count) under $TestDir"
}
foreach ($Trip in $TestTrips) {
    $Base = Join-Path $Trip.FullName $Trip.Name
    foreach ($Extension in @('.track', '.nodes', '.arcs', '.route')) {
        if (-not (Test-Path -LiteralPath "$Base$Extension" -PathType Leaf)) {
            throw "Missing required trip file: $Base$Extension"
        }
    }
}

$VenvPython = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Virtual environment is missing. Run scripts/setup.ps1 first. Expected $VenvPython"
}
$env:PATHSNAP_DATA_DIR = $DataDir
$env:PATHSNAP_CORS = $CorsOrigins
$SourcePath = Join-Path $RepoRoot 'src'
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$SourcePath;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $SourcePath
}
& $VenvPython (Join-Path $RepoRoot 'scripts/validate_dataset.py') --data-dir $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed with exit code $LASTEXITCODE"
}
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    if (-not [System.IO.Path]::IsPathRooted($ModelPath)) {
        $ModelPath = Join-Path $CallerDirectory $ModelPath
    }
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw "Requested model bundle not found: $ModelPath"
    }
    $env:PATHSNAP_MODEL = (Resolve-Path -LiteralPath $ModelPath).Path
} elseif (-not [string]::IsNullOrWhiteSpace($env:PATHSNAP_MODEL)) {
    if (-not [System.IO.Path]::IsPathRooted($env:PATHSNAP_MODEL)) {
        $env:PATHSNAP_MODEL = Join-Path $CallerDirectory $env:PATHSNAP_MODEL
    }
    if (-not (Test-Path -LiteralPath $env:PATHSNAP_MODEL -PathType Leaf)) {
        throw "PATHSNAP_MODEL does not exist: $env:PATHSNAP_MODEL"
    }
    $env:PATHSNAP_MODEL = (Resolve-Path -LiteralPath $env:PATHSNAP_MODEL).Path
}

$DefaultModel = Join-Path $RepoRoot 'artifacts/model_bundle.joblib'
if (-not $env:PATHSNAP_MODEL -and -not (Test-Path -LiteralPath $DefaultModel -PathType Leaf)) {
    Write-Warning 'No model bundle found. The API will list trips, report model_loaded=false, and return HTTP 503 from /predict.'
}
Write-Host "Starting PathSnap API at http://$HostAddress`:$Port"
Write-Host "Dataset root: $DataDir (10 reserved trips validated)"
if ($HostAddress -eq '0.0.0.0') {
    Write-Warning "The API listens on all interfaces. Keep port $Port limited to a trusted private LAN; do not expose it publicly."
}
& $VenvPython -m uvicorn api.main:app --host $HostAddress --port $Port
if ($LASTEXITCODE -ne 0) {
    throw "FastAPI exited with code $LASTEXITCODE"
}
