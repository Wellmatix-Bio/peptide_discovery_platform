<#
.SYNOPSIS
  Start the job API on this computer so the food dashboard's "Run" button runs the pipeline locally (no Vertex, no bucket, no image).

  It links the model weights the food pipeline needs into model_store/model_weights/, starts the API with DEV_MODE=true on
  http://localhost:8000, and removes the links when you stop it (Ctrl+C). Run it from anywhere.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy/run-food-local.ps1
  powershell -ExecutionPolicy Bypass -File deploy/run-food-local.ps1 -WeightsDir "D:\AI Peptide Discovery Platform\Models\model_weights" -Port 8000
#>
param(
    [string]$WeightsDir = 'D:\AI Peptide Discovery Platform\Models\model_weights',
    [int]$Port = 8000,
    [string]$Python = 'python'  # must be the environment with requirements.txt installed
)

$ErrorActionPreference = 'Stop'
$Repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$Target = Join-Path $Repo 'model_store/model_weights'
# The folders the food pipeline reads (see FOOD_REQUIRED_WEIGHT_FILES in src/schemas/food_run_config.py).
$Needed = @('esm2_t30_150M', 'allergenicity_predictor_v1', 'cytotoxicity_predictor_v1', 'solubility_predictor_v1', 'aggregation_predictor_v1', 'synthesis_feasibility_predictor_v1', 'bitterness_predictor_v1')

# The API and the worker it starts run in this Python: stop before linking anything if it lacks what they import.
$probe = "import importlib.util as u; m = [x for x in ('fastapi','uvicorn','pydantic','yaml','google.cloud.storage','torch','transformers','xgboost','reportlab') if u.find_spec(x.split('.')[0]) is None]; print(' '.join(m))"
$missing = (& $Python -c $probe 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: could not run '$Python' (pass -Python with the full path of the python.exe that has the project installed)" -ForegroundColor Red; exit 1 }
if ($missing) {
    Write-Host "ERROR: '$Python' is missing: $missing" -ForegroundColor Red
    Write-Host "Install the project there (python -m pip install -r requirements.txt) or pass -Python with the full path of a python.exe that has it." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $WeightsDir)) { Write-Host "ERROR: weights folder not found: $WeightsDir (pass -WeightsDir)" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force $Target | Out-Null

$linked = @()
foreach ($name in $Needed) {
    $dest = Join-Path $Target $name
    if (Test-Path $dest) { continue }  # already there (for example the bitterness weights copied into the repo)
    $src = Join-Path $WeightsDir $name
    if (-not (Test-Path $src)) { Write-Host "ERROR: missing weights: $src" -ForegroundColor Red; exit 1 }
    cmd /c mklink /J "$dest" "$src" | Out-Null
    $linked += $dest
}
Write-Host "Linked $($linked.Count) weight folder(s); starting the API on http://localhost:$Port (Ctrl+C to stop)" -ForegroundColor Green

$env:DEV_MODE = 'true'
$env:PYTHONPATH = "$Repo\src;$Repo"
try {
    & $Python -m uvicorn backend.api_e2e.api:app --host 127.0.0.1 --port $Port
}
finally {
    # rmdir on a junction removes the link only, never the weights.
    foreach ($dest in $linked) { cmd /c rmdir "$dest" | Out-Null }
    Write-Host "Removed $($linked.Count) weight link(s)" -ForegroundColor Green
}
