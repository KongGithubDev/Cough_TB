<#
.SYNOPSIS
    CoughTB — Windows VPS Setup & Run Script
.DESCRIPTION
    Install dependencies, clone repo, and start FastAPI server as a Windows service.
    Run as Administrator for best results.
.PARAMETER Port
    HTTP port to listen on (default: 80, requires admin)
.PARAMETER InstallDeps
    Install Python 3.11 + ffmpeg automatically
.PARAMETER InstallService
    Install as a Windows service via NSSM (auto-start on boot)
.PARAMETER UninstallService
    Remove the Windows service
.EXAMPLE
    # Quick run in terminal (press Ctrl+C to stop)
    .\run_windows_vps.ps1 -Port 8080
    
    # Full setup + install as service (run once as Admin)
    .\run_windows_vps.ps1 -Port 80 -InstallDeps -InstallService
#>

param(
    [int]$Port = 80,
    [switch]$InstallDeps,
    [switch]$InstallService,
    [switch]$UninstallService
)

$ErrorActionPreference = "Stop"
$AppDir = "$env:USERPROFILE\coughtb"
$ServiceName = "CoughTB"

# ---- Check Administrator ----
$isAdmin = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = $isAdmin.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin -and ($InstallService -or $Port -eq 80)) {
    Write-Host "⚠ Port 80 and service install require Administrator." -ForegroundColor Yellow
    Write-Host "  Right-click PowerShell → Run as Administrator" -ForegroundColor Yellow
    Write-Host "  (You can use -Port 8080 without admin for testing)" -ForegroundColor Yellow
    exit 1
}

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   CoughTB — Windows VPS Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Install dependencies ----
if ($InstallDeps) {
    # ---- ffmpeg ----
    Write-Host "[1/3] Installing ffmpeg..." -ForegroundColor Yellow
    try {
        winget install "FFmpeg (Essentials Build)" --accept-package-agreements --silent 2>$null
        Write-Host "  ✓ ffmpeg installed (restart session to refresh PATH)" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ winget failed. Install ffmpeg from:" -ForegroundColor Red
        Write-Host "    https://ffmpeg.org/download.html#build-windows"
        Write-Host "    Then add ffmpeg.exe folder to PATH manually."
    }

    # ---- Python ----
    Write-Host "[2/3] Installing Python 3.11..." -ForegroundColor Yellow
    $pythonFound = $false
    $pythonPaths = @(
        "$env:ProgramFiles\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    )
    foreach ($p in $pythonPaths) {
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

# ---- Step 2: Clone / Update repo ----
Write-Host "[3/3] Cloning CoughTB repo..." -ForegroundColor Yellow
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

# ---- Step 3: Python venv + dependencies ----
Write-Host "[3/3] Setting up Python virtual environment..." -ForegroundColor Yellow

# Find Python binary
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

# Create venv
$venvDir = "$AppDir\venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    & $pythonExe -m venv $venvDir
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Install deps
$pip = "$venvDir\Scripts\pip.exe"
& $pip install --upgrade pip --quiet
Write-Host "  Installing Python dependencies..." -ForegroundColor Gray
& $pip install -r "$AppDir\web\requirements.txt" --quiet
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "  ✓ All dependencies installed" -ForegroundColor Green

# ---- Step 4: Firewall rule ----
Write-Host "[4/4] Opening firewall port $Port..." -ForegroundColor Yellow
try {
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
} catch {
    Write-Host "  ⚠ Could not add firewall rule (run as Admin)" -ForegroundColor Yellow
}

# ---- Step 5: Install as Windows service (via NSSM) ----
if ($InstallService) {
    Write-Host "[5/5] Installing as Windows service '$ServiceName'..." -ForegroundColor Yellow
    
    # Download NSSM if not present
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
    
    # Stop existing service if running
    & $nssmExe stop $ServiceName 2>$null
    & $nssmExe remove $ServiceName confirm 2>$null

    # Install (4 threads for inference — adjust if your VPS has fewer cores)
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
    
    # Start service
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

# ---- Start the app (foreground, unless service was installed) ----
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
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host "   Starting CoughTB Server (foreground)..." -ForegroundColor Cyan
    Write-Host "   Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host ""
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq 'Dhcp' -or $_.PrefixOrigin -eq 'Manual' }).IPAddress[0]
    Write-Host "  Local:   http://localhost:$Port" -ForegroundColor White
    Write-Host "  Network: http://$($ip):$Port" -ForegroundColor White
    Write-Host "  Health:  http://localhost:$Port/health" -ForegroundColor White
    Write-Host ""

    # Activate venv
    $env:PATH = "$venvDir\Scripts;$env:PATH"
    $env:PORT = $Port

    Set-Location "$AppDir\web"
    & "$venvDir\Scripts\uvicorn.exe" app:app --host 0.0.0.0 --port $Port --workers 1
}
