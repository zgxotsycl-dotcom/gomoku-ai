Param(
  [switch]$Verbose
)

$ErrorActionPreference = "Stop"
Write-Host "[Preflight] Starting GPU environment checks..." -ForegroundColor Cyan

function Check-Cmd($name, $args) {
  try {
    $p = Start-Process -FilePath $name -ArgumentList $args -NoNewWindow -PassThru -Wait -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

# 1) NVIDIA driver present
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Host "[OK] nvidia-smi found:" -ForegroundColor Green
  nvidia-smi | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
} else {
  Write-Warning "[WARN] nvidia-smi not found. Install NVIDIA drivers."
}

# 2) Docker present
if (Check-Cmd docker "--version") {
  Write-Host "[OK] Docker detected" -ForegroundColor Green
} else {
  Write-Error "[ERROR] Docker is not installed or not in PATH."
  exit 1
}

# 3) Docker Compose v2
if (Check-Cmd docker "compose version") {
  Write-Host "[OK] Docker Compose v2 detected" -ForegroundColor Green
} else {
  Write-Warning "[WARN] Docker Compose v2 not found. Update Docker Desktop."
}

Write-Host "[Preflight] Done. For GPU run: npm run docker:gpu" -ForegroundColor Cyan

