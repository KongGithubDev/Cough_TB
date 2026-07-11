<#
.SYNOPSIS
    CoughTB — Windows VPS Setup & Run Script
.DESCRIPTION
    Install dependencies, clone repo, and start FastAPI server.
    Run as Administrator for best results.
.EXAMPLE
    .\run_windows_vps.ps1 -Port 8000
#>

param(
    [int]$Port = 80,
    [string]$RepoUrl = "https://github.com/KongGithubDev/Cough_TB.git",
    [switch]$InstallPython,
    [switch]$InstallFfmpeg
)

$ErrorActionPreference = "Stop"
$AppDir = "$env:USERPROFILE\coughtb"

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   CoughTB — Windows VPS Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Install ffmpeg ----
if ($InstallFfmpeg) {
    Write-Host "[1/5] Installing ffmpeg via winget..." -ForegroundColor Yellow
    try {
        winget install ffmpeg --accept-package-agreements --silent 2>$null
        Write-Host "  ✓ ffmpeg installed" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ winget failed. Install ffmpeg manually from https://ffmpeg.org/download.html" -ForegroundColor Red
        Write-Host "    Add ffmpeg.exe to PATH after installing."
    }
}

# ---- Step 2: Install Python ----
if ($InstallPython) {
    Write-Host "[2/5] Installing Python 3.11..." -ForegroundColor Yellow
    $pyUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $pyInstaller = "$env:TEMP\python-3.11.9-amd64.exe"
    Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller
    Start-Process -Wait -FilePath $pyInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1"
    Write-Host "  ✓ Python 3.11 installed" -ForegroundColor Green
}

# ---- Step 3: Clone / Update repo ----
Write-Host "[3/5] Cloning CoughTB repo..." -ForegroundColor Yellow
if (Test-Path "$AppDir\.git") {
    Write-Host "  Repo exists — pulling latest..." -ForegroundColor Gray
    Set-Location $AppDir
    git pull
} else {
    if (Test-Path $AppDir) {
        Remove-Item -Recurse -Force $AppDir
    }
    git clone $RepoUrl $AppDir
    Set-Location $AppDir
}
Write-Host "  ✓ Repo ready at $AppDir" -ForegroundColor Green

# ---- Step 4: Python venv + dependencies ----
Write-Host "[4/5] Setting up Python virtual environment..." -ForegroundColor Yellow

# Find Python
$python = "python"
try {
    $ver = & $python --version 2>&1
    Write-Host "  Using: $ver" -ForegroundColor Gray
} catch {
    $python = "$env:ProgramFiles\Python311\python.exe"
    Write-Host "  Using: $python" -ForegroundColor Gray
}

# Create venv
$venvDir = "$AppDir\venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    & $python -m venv $venvDir
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Activate and install
$pip = "$venvDir\Scripts\pip.exe"
& $pip install --upgrade pip --quiet
Write-Host "  Installing dependencies..." -ForegroundColor Gray
& $pip install -r "$AppDir\web\requirements.txt" --quiet
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "  ✓ All dependencies installed" -ForegroundColor Green

# ---- Step 5: Firewall rule ----
Write-Host "[5/5] Opening firewall port $Port..." -ForegroundColor Yellow
$ruleName = "CoughTB Port $Port"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if (-not $existing) {
    New-NetFirewallRule -DisplayName $ruleName `
        -Direction Inbound -Protocol TCP -LocalPort $Port `
        -Action Allow -Profile Any | Out-Null
    Write-Host "  ✓ Firewall rule added for port $Port" -ForegroundColor Green
} else {
    Write-Host "  ✓ Firewall rule already exists" -ForegroundColor Green
}

# ---- Start the app ----
Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   Starting CoughTB Server..." -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Local:    http://localhost:$Port" -ForegroundColor White
Write-Host "  Network:  http://$((Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.PrefixOrigin -eq 'Dhcp' -or $_.PrefixOrigin -eq 'Manual'}).IPAddress[0]):$Port" -ForegroundColor White
Write-Host "  Health:   http://localhost:$Port/health" -ForegroundColor White
Write-Host ""

# Activate venv in this process
$env:PATH = "$venvDir\Scripts;$env:PATH"
$env:PORT = $Port

# Start server
Set-Location "$AppDir\web"
& "$venvDir\Scripts\uvicorn.exe" app:app --host 0.0.0.0 --port $Port --workers 1
