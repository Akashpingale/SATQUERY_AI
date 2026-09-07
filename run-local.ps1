# SatQuery AI - one-command local run.
#   cd D:\SatQuery-AI ; .\run-local.ps1
# Starts the existing backend (:8000) and frontend (:5173) if they are not already
# running, waits until both are ready, then opens the app in the browser.
# Reuses the project's existing startup commands/ports. Changes no app logic.

$ErrorActionPreference = 'Stop'
$root = 'D:\SatQuery-AI'
Set-Location $root

$backendHealth = 'http://localhost:8000/api/v1/health'
$frontendUrl   = 'http://localhost:5173'

function Test-Up([string]$url) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}

# --- Backend (reuses the existing uvicorn command + offline model env) ---
if (Test-Up $backendHealth) {
    Write-Host "Backend already running on :8000" -ForegroundColor Green
} else {
    Write-Host "Starting backend on :8000 ..." -ForegroundColor Cyan
    $backendCmd = "cd '$root'; " +
        "`$env:HF_HUB_OFFLINE='1'; `$env:TRANSFORMERS_OFFLINE='1'; " +
        "`$env:HF_HOME='$root\models\hf-cache'; `$env:DEVICE='cpu'; " +
        ".\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000"
    Start-Process -FilePath 'powershell' -ArgumentList '-NoExit', '-Command', $backendCmd -WindowStyle Minimized
}

# --- Frontend (reuses the existing Vite dev server) ---
if (Test-Up $frontendUrl) {
    Write-Host "Frontend already running on :5173" -ForegroundColor Green
} else {
    Write-Host "Starting frontend on :5173 ..." -ForegroundColor Cyan
    $frontendCmd = "cd '$root\frontend'; npm run dev"
    Start-Process -FilePath 'powershell' -ArgumentList '-NoExit', '-Command', $frontendCmd -WindowStyle Minimized
}

# --- Wait until both are ready (up to ~120s) ---
Write-Host "Waiting for servers to be ready ..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ((Test-Up $backendHealth) -and (Test-Up $frontendUrl)) {
        Write-Host "Both servers are ready." -ForegroundColor Green
        Start-Process $frontendUrl
        Write-Host "Opened $frontendUrl" -ForegroundColor Green
        return
    }
    Start-Sleep -Seconds 2
}

Write-Host "Timed out waiting for servers. Backend up: $(Test-Up $backendHealth), Frontend up: $(Test-Up $frontendUrl)" -ForegroundColor Yellow
