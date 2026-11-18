[CmdletBinding()]
param(
    [string]$DataDir,
    [switch]$SkipInstall,
    [switch]$ForTests
)

$ErrorActionPreference = 'Stop'
$CallerDirectory = (Get-Location).Path
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = $env:PATHSNAP_DATA_DIR
}
if ([string]::IsNullOrWhiteSpace($DataDir) -and
    (Test-Path (Join-Path $RepoRoot 'train')) -and
    (Test-Path (Join-Path $RepoRoot 'student_test'))) {
    $DataDir = $RepoRoot
}
if (-not [string]::IsNullOrWhiteSpace($DataDir)) {
    if (-not [System.IO.Path]::IsPathRooted($DataDir)) {
        $DataDir = Join-Path $CallerDirectory $DataDir
    }
    if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
        throw "Dataset directory not found: $DataDir"
    }
    $DataDir = (Resolve-Path -LiteralPath $DataDir).Path
    if (-not (Test-Path (Join-Path $DataDir 'train')) -or
        -not (Test-Path (Join-Path $DataDir 'student_test'))) {
        throw "Dataset root must contain train/ and student_test/: $DataDir"
    }
    $testTrips = @(Get-ChildItem -LiteralPath (Join-Path $DataDir 'student_test') -Directory)
    if ($testTrips.Count -ne 10) {
        throw "Expected 10 reserved student_test trips; found $($testTrips.Count) under $DataDir"
    }
    $env:PATHSNAP_DATA_DIR = $DataDir
}

$VenvDir = Join-Path $RepoRoot '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $PythonArgs = @()
    if (-not $PythonCommand) {
        $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
        $PythonArgs = @('-3')
    }
    if (-not $PythonCommand) {
        throw 'Python 3 was not found. Install Python 3.10+ and add python.exe or py.exe to PATH.'
    }
    $VenvArguments = @($PythonArgs) + @('-m', 'venv', $VenvDir)
    & $PythonCommand.Source @VenvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipInstall) {
    $RequirementsFile = if ($ForTests) { 'requirements-test.txt' } else { 'requirements.txt' }
    & $VenvPython -m pip install -r (Join-Path $RepoRoot $RequirementsFile)
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed with exit code $LASTEXITCODE"
    }
}
if ($DataDir) {
    & $VenvPython (Join-Path $RepoRoot 'scripts/validate_dataset.py') --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Repository: $RepoRoot"
Write-Host "Virtual environment: $VenvDir"
if ($DataDir) {
    Write-Host "Dataset root: $DataDir (10 reserved trips found)"
} else {
    Write-Host 'Dataset root not set. Backend startup requires PATHSNAP_DATA_DIR or repository train/ and student_test/ folders.'
}
if (Test-Path (Join-Path $RepoRoot 'artifacts/model_bundle.joblib')) {
    Write-Host 'Model bundle found.'
} else {
    Write-Warning 'No model bundle found; the API will report model_loaded=false and /predict will return HTTP 503.'
}
