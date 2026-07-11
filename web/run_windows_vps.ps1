<#
.SYNOPSIS
    CoughTB — Windows VPS Setup & Run Script
.DESCRIPTION
    Install dependencies, clone repo, and start FastAPI server.
    For simple foreground mode (no service), run without InstallService.
.PARAMETER Port
    HTTP port to listen on (default: 3003)
.PARAMETER InstallDeps
    Install Python 3.11 + ffmpeg automatically
.PARAMETER InstallService
    Install as a Windows service via NSSM (auto-start on boot)
.PARAMETER UninstallService
    Remove the Windows service
.EXAMPLE
    # Quick run in terminal (press Ctrl+C to stop)
    .\run_windows_vps.ps1
    
    # With auto-install deps + run as service
    .\run_windows_vps.ps1 -InstallDeps -InstallService
#>

param(
    [int]$Port = 3003,
    [switch]$InstallDeps,
    [switch]$InstallService,
    [switch]$UninstallService
)

$ErrorActionPreference = "Stop"
$AppDir = "$env:USERPROFILE\coughtb"
$ServiceName = "CoughTB"
$step = 0

# ---- Check Administrator ----
$isAdmin = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = $isAdmin.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin -and ($InstallService -or $Port -eq 80)) {
    Write-Host "⚠ Service install or port 80 require Administrator." -ForegroundColor Yellow
    Write-Host "  Right-click PowerShell → Run as Administrator" -ForegroundColor Yellow
    exit 1
}

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   CoughTB — Windows VPS Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Install dependencies ----
$totalSteps = 3
if ($InstallDeps) { $totalSteps = 5 }

if ($InstallDeps) {
    $step++
    Write-Host "[$step/$totalSteps] Installing ffmpeg..." -ForegroundColor Yellow
    try {
        winget install "FFmpeg (Essentials Build)" --accept-package-agreements --silent 2>$null
        Write-Host "  ✓ ffmpeg installed (restart session to refresh PATH)" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ winget failed. Install ffmpeg from:" -ForegroundColor Red
        Write-Host "    https://ffmpeg.org/download.html#build-windows"
    }

    $step++
    Write-Host "[$step/$totalSteps] Installing Python 3.11..." -ForegroundColor Yellow
    $pythonFound = $false
    foreach ($p in @("$env:ProgramFiles\Python311\python.exe","$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
        if (Test-Path $p) { $pythonFound = $true; break }
    }
    try { & python --version 2>$null; $pythonFound = $true } catch {}

    if (-not $pythonFound) {
        $pyUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        $pyInstaller = "$env:TEMP\python-3.11.9-amd64.exe"
        Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller
        Start-Process -Wait -FilePath $pyInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1"
        Write-Host "  ✓ Python 3.11 installed" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Python 3.11 already installed" -ForegroundColor Green
    }
}

# ---- Clone / Update repo ----
$step++
Write-Host "[$step/$totalSteps] Cloning CoughTB repo..." -ForegroundColor Yellow
if (Test-Path "$AppDir\.git") {
    Write-Host "  Repo exists — pulling latest..." -ForegroundColor Gray
    Set-Location $AppDir
    git pull
} else {
    if (Test-Path $AppDir) { Remove-Item -Recurse -Force $AppDir }
    git clone "https://github.com/KongGithubDev/Cough_TB.git" $AppDir
    Set-Location $AppDir
}
Write-Host "  ✓ Repo ready at $AppDir" -ForegroundColor Green

# ---- Python venv + dependencies ----
$step++
Write-Host "[$step/$totalSteps] Setting up Python virtual environment..." -ForegroundColor Yellow

$pythonExe = $null
foreach ($p in @(
    "$env:ProgramFiles\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue).Source,
    (Get-Command py -ErrorAction SilentlyContinue).Source
)) {
    if ($p -and (Test-Path $p)) { $pythonExe = $p; break }
}
if (-not $pythonExe) {
    Write-Host "✗ Python not found. Install Python 3.11+ first." -ForegroundColor Red
    exit 1
}
Write-Host "  Using Python: $pythonExe" -ForegroundColor Gray

$venvDir = "$AppDir\venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    & $pythonExe -m venv $venvDir
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

$pip = "$venvDir\Scripts\pip.exe"
& $pip install --upgrade pip --quiet
Write-Host "  Installing Python dependencies..." -ForegroundColor Gray
& $pip install -r "$AppDir\web\requirements.txt" --quiet
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "  ✓ All dependencies installed" -ForegroundColor Green

# ---- Install as Windows service (via NSSM) ----
if ($InstallService) {
    $step++
    Write-Host "[$step/$totalSteps] Installing as Windows service '$ServiceName'..." -ForegroundColor Yellow

    $nssmExe = "$AppDir\nssm.exe"
    if (-not (Test-Path $nssmExe)) {
        Write-Host "  Downloading NSSM..." -ForegroundColor Gray
        $nssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
        $nssmZip = "$env:TEMP\nssm.zip"
        Invoke-WebRequest -Uri $nssmUrl -OutFile $nssmZip
        Expand-Archive -Path $nssmZip -DestinationPath "$env:TEMP\nssm" -Force
        Copy-Item "$env:TEMP\nssm\nssm-2.24\win64\nssm.exe" $nssmExe
        Remove-Item -Recurse -Force "$env:TEMP\nssm", $nssmZip
    }

    & $nssmExe stop $ServiceName 2>$null
    & $nssmExe remove $ServiceName confirm 2>$null

    $uvicornExe = "$venvDir\Scripts\uvicorn.exe"
    & $nssmExe install $ServiceName $uvicornExe "app:app --host 0.0.0.0 --port $Port --workers 1"
    & $nssmExe set $ServiceName AppDirectory "$AppDir\web"
    & $nssmExe set $ServiceName AppEnvironmentExtra "PORT=$Port"
    & $nssmExe set $ServiceName AppEnvironmentExtra "OMP_NUM_THREADS=4"
    & $nssmExe set $ServiceName DisplayName "CoughTB — TB Cough Detection"
    & $nssmExe set $ServiceName Description "AI-powered TB screening from cough sounds"
    & $nssmExe set $ServiceName Start SERVICE_AUTO_START
    & $nssmExe set $ServiceName AppStdout "$AppDir\coughtb.log"
    & $nssmExe set $ServiceName AppStderr "$AppDir\coughtb-error.log"

    & $nssmExe start $ServiceName
    Write-Host "  ✓ Service '$ServiceName' installed and started" -ForegroundColor Green
    Write-Host "  Logs: $AppDir\coughtb.log" -ForegroundColor Gray
}

# ---- Uninstall service ----
if ($UninstallService) {
    Write-Host "Removing Windows service '$ServiceName'..." -ForegroundColor Yellow
    $nssmExe = "$AppDir\nssm.exe"
    if (Test-Path $nssmExe) {
        & $nssmExe stop $ServiceName 2>$null
        & $nssmExe remove $ServiceName confirm
    }
    sc.exe delete $ServiceName 2>$null
    Write-Host "  ✓ Service removed" -ForegroundColor Green
    exit 0
}

# ---- Start ----
if ($InstallService) {
    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host "   CoughTB is running as a Windows service!" -ForegroundColor Cyan
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host ""
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq 'Dhcp' -or $_.PrefixOrigin -eq 'Manual' }).IPAddress[0]
    Write-Host "  URL:    http://$($ip):$Port" -ForegroundColor White
    Write-Host "  Health: http://$($ip):$Port/health" -ForegroundColor White
    Write-Host "  Logs:   $AppDir\coughtb.log" -ForegroundColor White
    Write-Host "  Status: nssm status $ServiceName" -ForegroundColor White
    Write-Host "  Stop:   nssm stop $ServiceName" -ForegroundColor White
} else {
    $step++
    Write-Host "[$step/$totalSteps] Starting server on port $Port..." -ForegroundColor Yellow
    Write-Host "  Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq 'Dhcp' -or $_.PrefixOrigin -eq 'Manual' }).IPAddress[0]
    Write-Host "  Local:   http://localhost:$Port" -ForegroundColor White
    Write-Host "  Network: http://$($ip):$Port" -ForegroundColor White
    Write-Host "  Health:  http://localhost:$Port/health" -ForegroundColor White
    Write-Host ""

    $env:PATH = "$venvDir\Scripts;$env:PATH"
    $env:PORT = $Port

    Set-Location "$AppDir\web"
    & "$venvDir\Scripts\uvicorn.exe" app:app --host 0.0.0.0 --port $Port --workers 1
}
