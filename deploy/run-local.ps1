<#
.SYNOPSIS
  Run the UI, accounts service and job API locally in Docker; pipeline jobs execute remotely on Vertex AI.

  Standalone: needs only this script and the three images already in Docker. No repository files.
  Settings are kept in a .env file next to this script; anything missing is asked for and saved.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File run-local.ps1
  powershell -ExecutionPolicy Bypass -File run-local.ps1 -Port 8090
  powershell -ExecutionPolicy Bypass -File run-local.ps1 -Stop
  powershell -ExecutionPolicy Bypass -File run-local.ps1 -ApiImage myrepo/api:1.0 -AccountsImage myrepo/accounts:1.0 -WebImage myrepo/web:1.0
#>
param(
    [string]$EnvFile,
    [int]$Port = 8088,
    [string]$ApiImage = 'peptide-discovery-api',
    [string]$AccountsImage = 'peptide-discovery-accounts',
    [string]$WebImage = 'peptide-discovery-web',
    [switch]$Stop,
    [switch]$NoBrowser
)

# Native tools write progress to stderr; failures are detected by exit code instead.
$ErrorActionPreference = 'Continue'

if (-not $EnvFile) { $EnvFile = Join-Path $PSScriptRoot '../.env' }

$Network = 'peptide-discovery'
$Volume = 'peptide-accounts-data'
$Containers = @('peptide-web', 'peptide-accounts', 'peptide-api')

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    !!  $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

function Remove-Containers {
    foreach ($c in $Containers) { $null = & docker rm -f $c 2>$null }
}

# ---------- Docker ----------
Step 'Checking Docker'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail 'Docker is not installed. Install Docker Desktop and run this again.' }
$null = & docker info 2>$null
if ($LASTEXITCODE -ne 0) { Fail 'Docker is not running. Start Docker Desktop, wait until it says "Running", and run this again.' }
Ok 'Docker is running'

if ($Stop) {
    Step 'Stopping'
    Remove-Containers
    Ok 'Containers removed (accounts data is kept in the volume)'
    exit 0
}

# ---------- Settings file ----------
Step "Reading settings from $EnvFile"
if (-not (Test-Path $EnvFile)) {
    [IO.File]::WriteAllText($EnvFile, "# Settings for run-local.ps1. Never share this file: it holds the sign-in signing key.`r`n", (New-Object Text.UTF8Encoding $false))
    Warn "No settings file yet; created $EnvFile and will ask for each setting"
}

function Read-EnvFile($path) {
    $vars = [ordered]@{}
    foreach ($line in [IO.File]::ReadAllLines($path)) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith('#') -or -not $t.Contains('=')) { continue }
        $i = $t.IndexOf('=')
        $k = $t.Substring(0, $i).Trim()
        $v = $t.Substring($i + 1).Trim().Trim('"').Trim("'")
        # Template placeholders such as <your-bucket> count as unset.
        if ($v -match '<[^>]+>') { $v = '' }
        $vars[$k] = $v
    }
    return $vars
}

function Set-EnvValue($path, $key, $value) {
    $lines = [Collections.Generic.List[string]]([IO.File]::ReadAllLines($path))
    $idx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match "^\s*$key\s*=") { $idx = $i; break } }
    if ($idx -ge 0) { $lines[$idx] = "$key=$value" } else { $lines.Add("$key=$value") }
    [IO.File]::WriteAllLines($path, $lines, (New-Object Text.UTF8Encoding $false))
}

$envVars = Read-EnvFile $EnvFile

# Generated once and saved, so sign-ins survive restarts.
if (-not $envVars['ACCOUNTS_SECRET_KEY'] -or $envVars['ACCOUNTS_SECRET_KEY'].Length -lt 32) {
    $bytes = New-Object byte[] 48
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
    Set-EnvValue $EnvFile 'ACCOUNTS_SECRET_KEY' $key
    $envVars['ACCOUNTS_SECRET_KEY'] = $key
    Ok 'Generated ACCOUNTS_SECRET_KEY and saved it to the settings file'
}

$defaults = @{
    'VERTEX_LOCATION'   = 'us-central1'
    'MACHINE_TYPE'      = 'g2-standard-4'
    'ACCELERATOR_TYPE'  = 'NVIDIA_L4'
    'ACCELERATOR_COUNT' = '1'
}
$prompts = [ordered]@{
    'VERTEX_CLOUD_PROJECT'   = 'GCP project id (e.g. my-project-123)'
    'VERTEX_LOCATION'        = 'Vertex region'
    'VERTEX_ARTIFACTS_DIR'   = 'Artifacts folder in Cloud Storage (gs://<bucket>/artifacts)'
    'VERTEX_MODEL_STORE'     = 'Model weights folder in Cloud Storage (gs://<bucket>/model_weights)'
    'SEED_CANDIDATES_FILE'   = 'Seed FASTA in Cloud Storage (gs://<bucket>/curated_peptides.fasta)'
    'WORKER_IMAGE_URI'       = 'Wound-healing worker image (<region>-docker.pkg.dev/<project>/<repo>/worker-e2e:<tag>)'
    'WORKER_SERVICE_ACCOUNT' = 'Worker service account (worker@<project>.iam.gserviceaccount.com)'
    'MACHINE_TYPE'           = 'Vertex machine type'
    'ACCELERATOR_TYPE'       = 'GPU type'
    'ACCELERATOR_COUNT'      = 'GPU count'
}
foreach ($k in $prompts.Keys) {
    if ($envVars[$k]) { continue }
    if ($defaults.ContainsKey($k)) {
        $answer = Read-Host "$($prompts[$k]) [$($defaults[$k])]"
        if (-not $answer) { $answer = $defaults[$k] }
    } else {
        do { $answer = (Read-Host $prompts[$k]).Trim() } while (-not $answer)
    }
    Set-EnvValue $EnvFile $k $answer
    $envVars[$k] = $answer
}
foreach ($k in 'VERTEX_ARTIFACTS_DIR', 'VERTEX_MODEL_STORE', 'SEED_CANDIDATES_FILE') {
    if (-not $envVars[$k].StartsWith('gs://')) { Fail "$k must start with gs:// (it is '$($envVars[$k])'). Fix it in $EnvFile." }
}
if ($envVars['VERTEX_MODEL_STORE'] -like '*TODO-bucket*') { Fail "VERTEX_MODEL_STORE is still the placeholder. Fix it in $EnvFile." }
Ok "Project $($envVars['VERTEX_CLOUD_PROJECT']), region $($envVars['VERTEX_LOCATION'])"

# ---------- Google credentials ----------
Step 'Checking Google credentials'
$keyFile = $envVars['GOOGLE_CREDENTIALS_FILE']
if ($keyFile -and (Test-Path $keyFile)) {
    # A service-account key was given; it does not expire, so no login is needed.
    $credFile = (Resolve-Path $keyFile).Path
    $savedGac = $env:GOOGLE_APPLICATION_CREDENTIALS
    $env:GOOGLE_APPLICATION_CREDENTIALS = $credFile
    Ok "Using service-account key $credFile"
} else {
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Fail 'The Google Cloud CLI (gcloud) is not installed. Install it from https://cloud.google.com/sdk/docs/install and run this again.'
    }
    $gcloudDir = $env:CLOUDSDK_CONFIG
    if (-not $gcloudDir) { $gcloudDir = Join-Path $env:APPDATA 'gcloud' }
    $credFile = Join-Path $gcloudDir 'application_default_credentials.json'

    $null = & gcloud auth application-default print-access-token 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $credFile)) {
        Warn 'Not signed in (or the sign-in expired). A browser window will open: sign in with your Google account.'
        & gcloud auth application-default login
        if ($LASTEXITCODE -ne 0) { Fail 'Google sign-in did not complete.' }
    }
    # Bills API quota to the project; harmless if it is already set.
    $null = & gcloud auth application-default set-quota-project $envVars['VERTEX_CLOUD_PROJECT'] 2>$null
    Ok "Signed in; credentials at $credFile"
}

$token = $null
if (Get-Command gcloud -ErrorAction SilentlyContinue) {
    $token = (& gcloud auth application-default print-access-token 2>$null | Select-Object -Last 1)
}
# Restore the caller's environment after borrowing it for the key file.
if ($keyFile -and (Test-Path $keyFile)) {
    if ($null -eq $savedGac) { Remove-Item Env:GOOGLE_APPLICATION_CREDENTIALS -ErrorAction SilentlyContinue }
    else { $env:GOOGLE_APPLICATION_CREDENTIALS = $savedGac }
}

# ---------- Cloud Storage ----------
Step 'Checking Cloud Storage paths'
function Test-Gcs($gsPath) {
    $rest = $gsPath.Substring(5)
    $slash = $rest.IndexOf('/')
    if ($slash -lt 0) { $bucket = $rest; $prefix = '' } else { $bucket = $rest.Substring(0, $slash); $prefix = $rest.Substring($slash + 1) }
    $uri = "https://storage.googleapis.com/storage/v1/b/$bucket/o?maxResults=1&prefix=$([uri]::EscapeDataString($prefix))"
    try {
        $r = Invoke-RestMethod -Uri $uri -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 20
        return [bool]$r.items
    } catch {
        return $null
    }
}
if (-not $token) {
    Warn 'Could not get an access token to check storage; skipping the checks'
} else {
    $checks = [ordered]@{
        'VERTEX_ARTIFACTS_DIR' = $false
        'VERTEX_MODEL_STORE'   = $true
        'SEED_CANDIDATES_FILE' = $true
    }
    foreach ($k in 'SEED_CANDIDATES_FILE_METABOLIC', 'SEED_CANDIDATES_FILE_COSMETIC') {
        if ($envVars[$k]) { $checks[$k] = $true }
    }
    foreach ($k in $checks.Keys) {
        $path = $envVars[$k]
        $found = Test-Gcs $path
        if ($null -eq $found) { Warn "$k ($path): no access. The signed-in account needs read/write on that bucket." }
        elseif (-not $found -and $checks[$k]) { Warn "$k ($path): nothing there. Jobs will fail until it is uploaded." }
        else { Ok "$k  $path" }
    }
}

# ---------- Images ----------
Step 'Checking images'
$missingImages = @()
foreach ($name in $ApiImage, $AccountsImage, $WebImage) {
    $null = & docker image inspect $name 2>$null
    if ($LASTEXITCODE -eq 0) { Ok $name } else { $missingImages += $name }
}
if ($missingImages) {
    Write-Host "`n    Images on this machine:"
    & docker images --format '      {{.Repository}}:{{.Tag}}'
    Fail ("Not found: $($missingImages -join ', '). Pass the right names, e.g.`n" +
        "  -ApiImage <name> -AccountsImage <name> -WebImage <name>")
}

# ---------- Network, volume, port ----------
Step 'Preparing network and storage'
Remove-Containers
$null = & docker network inspect $Network 2>$null
if ($LASTEXITCODE -ne 0) { $null = & docker network create $Network }
$null = & docker volume inspect $Volume 2>$null
if ($LASTEXITCODE -ne 0) { $null = & docker volume create $Volume }
Ok "network '$Network', volume '$Volume'"

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Fail "Port $Port is already in use. Run again with a free one, e.g.  -Port 8090"
}

# ---------- Containers ----------
$apiKeys = @(
    'VERTEX_CLOUD_PROJECT', 'VERTEX_LOCATION', 'VERTEX_ARTIFACTS_DIR', 'VERTEX_MODEL_STORE',
    'SEED_CANDIDATES_FILE', 'WORKER_IMAGE_URI', 'WORKER_SERVICE_ACCOUNT',
    'MACHINE_TYPE', 'ACCELERATOR_TYPE', 'ACCELERATOR_COUNT',
    'WORKER_METABOLIC_IMAGE_URI', 'SEED_CANDIDATES_FILE_METABOLIC',
    'WORKER_COSMETIC_IMAGE_URI', 'SEED_CANDIDATES_FILE_COSMETIC',
    'WORKER_IMMUNO_IMAGE_URI'
)

function Start-Api([switch]$AsRoot) {
    $runArgs = @('run', '-d', '--name', 'peptide-api', '--network', $Network, '--network-alias', 'api', '--restart', 'unless-stopped')
    if ($AsRoot) { $runArgs += @('-u', '0') }
    foreach ($k in $apiKeys) { if ($envVars[$k]) { $runArgs += @('-e', "$k=$($envVars[$k])") } }
    $runArgs += @(
        '-e', 'DEV_MODE=false',
        '-e', 'GOOGLE_APPLICATION_CREDENTIALS=/secrets/credentials.json',
        '-e', "GOOGLE_CLOUD_PROJECT=$($envVars['VERTEX_CLOUD_PROJECT'])",
        '-v', "${credFile}:/secrets/credentials.json:ro",
        $ApiImage
    )
    $null = & docker @runArgs
    if ($LASTEXITCODE -ne 0) { Fail 'The API container did not start.' }
}

Step 'Starting the job API'
Start-Api
$null = & docker exec peptide-api python -c "open('/secrets/credentials.json').read()" 2>$null
if ($LASTEXITCODE -ne 0) {
    # Docker Desktop can mount the file with permissions the non-root user cannot read; fine to run as root locally.
    Warn 'The API could not read the credentials file as its normal user; restarting it as root (local use only)'
    $null = & docker rm -f peptide-api 2>$null
    Start-Api -AsRoot
}
$apiHealth = $null
for ($i = 0; $i -lt 30 -and -not $apiHealth; $i++) {
    Start-Sleep -Seconds 1
    $out = & docker exec peptide-api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read().decode())" 2>$null
    if ($LASTEXITCODE -eq 0) { $apiHealth = $out | ConvertFrom-Json }
}
if (-not $apiHealth) { & docker logs --tail 40 peptide-api; Fail 'The API did not become healthy; its log is above.' }
Ok 'API is up'

Step 'Starting the accounts service'
$null = & docker run -d --name peptide-accounts --network $Network --network-alias accounts --restart unless-stopped `
    -e "ACCOUNTS_SECRET_KEY=$($envVars['ACCOUNTS_SECRET_KEY'])" `
    -e 'PEPTIDE_UPSTREAM=http://api:8080' `
    -e 'WEB_CONCURRENCY=2' `
    -e 'ACCOUNTS_CLIENT_IP_HEADER=x-forwarded-for' `
    -v "${Volume}:/data" `
    $AccountsImage
if ($LASTEXITCODE -ne 0) { Fail 'The accounts container did not start.' }
Ok 'accounts started'

# nginx resolves "accounts" at startup, so the web container must start after it.
Step 'Starting the web UI'
$null = & docker run -d --name peptide-web --network $Network --restart unless-stopped -p "127.0.0.1:${Port}:80" $WebImage
if ($LASTEXITCODE -ne 0) { Fail 'The web container did not start.' }

$url = "http://127.0.0.1:$Port"
$up = $false
for ($i = 0; $i -lt 45 -and -not $up; $i++) {
    Start-Sleep -Seconds 1
    try { $up = (Invoke-WebRequest -Uri "$url/healthz" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { }
}
if (-not $up) {
    & docker logs --tail 20 peptide-accounts
    & docker logs --tail 20 peptide-web
    Fail 'The app did not come up; the logs are above.'
}
Ok "app is up at $url"

# ---------- Summary ----------
Step 'Pipelines'
foreach ($p in $apiHealth.pipelines.PSObject.Properties) {
    if ($p.Value.ready) { Ok $p.Name }
    else { Warn "$($p.Name): not available (missing $($p.Value.settings_missing -join ', '))" }
}
if ($apiHealth.settings_missing.Count -gt 0) { Warn "Missing shared settings: $($apiHealth.settings_missing -join ', ')" }

Write-Host ''
Write-Host "Open $url, register an account, and sign in. Jobs run on Vertex AI in $($envVars['VERTEX_CLOUD_PROJECT'])." -ForegroundColor Green
Write-Host 'Logs:  docker logs -f peptide-api     Stop:  powershell -ExecutionPolicy Bypass -File run-local.ps1 -Stop'
if (-not $NoBrowser) { Start-Process $url }
