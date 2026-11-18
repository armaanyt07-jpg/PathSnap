[CmdletBinding()]
param(
    [string]$ApiUrl,
    [string]$HostAddress,
    [ValidateRange(0, 65535)][int]$Port = 0
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if ([string]::IsNullOrWhiteSpace($ApiUrl)) {
    $ApiUrl = $env:PATHSNAP_API_URL
}
if ([string]::IsNullOrWhiteSpace($ApiUrl)) {
    $ApiUrl = 'http://localhost:8000'
}
if ([string]::IsNullOrWhiteSpace($HostAddress)) {
    $HostAddress = if ($env:PATHSNAP_UI_HOST) { $env:PATHSNAP_UI_HOST } else { '127.0.0.1' }
}
if ($Port -eq 0) {
    $Port = if ($env:PATHSNAP_UI_PORT) { [int]$env:PATHSNAP_UI_PORT } else { 8501 }
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "PATHSNAP_UI_PORT must be between 1 and 65535; got $Port"
}
$ApiUrl = $ApiUrl.TrimEnd('/')
$VenvPython = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Virtual environment is missing. Run scripts/setup.ps1 first. Expected $VenvPython"
}
$env:PATHSNAP_API_URL = $ApiUrl
$SourcePath = Join-Path $RepoRoot 'src'
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$SourcePath;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $SourcePath
}

Write-Host "Starting PathSnap UI at http://$HostAddress`:$Port"
Write-Host "FastAPI URL: $ApiUrl"
& $VenvPython -m streamlit run src/ui/app.py --server.address $HostAddress --server.port $Port
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit exited with code $LASTEXITCODE"
}
