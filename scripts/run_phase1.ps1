param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($env:PYTHON) { $python = $env:PYTHON }
elseif (Test-Path (Join-Path $root '.venv\Scripts\python.exe')) { $python = Join-Path $root '.venv\Scripts\python.exe' }
else { $python = 'python' }
if (-not $env:PATHSNAP_DATA_DIR) { $env:PATHSNAP_DATA_DIR = $root }

$limit = if ($env:PATHSNAP_LIMIT) { $env:PATHSNAP_LIMIT } else { '0' }
$negFrac = if ($env:NEG_FRAC) { $env:NEG_FRAC } else { '1.0' }
$mapTrip = if ($env:MAP_TRIP) { $env:MAP_TRIP } else { '00000056' }
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }

Write-Host "Python: $python"
Write-Host "Data:   $env:PATHSNAP_DATA_DIR"
& $python scripts/inventory.py; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python scripts/explore_data.py; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python scripts/build_features.py --splits train student_test --limit $limit; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python scripts/train_baseline.py --neg-frac $negFrac; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python scripts/make_map.py --split student_test --trip $mapTrip; if ($LASTEXITCODE) { exit $LASTEXITCODE }
Write-Host "Phase 1 complete. Start services with scripts/run_all.ps1 or scripts/run_api.ps1 and scripts/run_ui.ps1."
