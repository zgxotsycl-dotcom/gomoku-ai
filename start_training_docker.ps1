# Stop on any error
$ErrorActionPreference = "Stop"

# Move to project root
Set-Location -Path $PSScriptRoot

$composeFiles = @('docker-compose.yml')
if (Test-Path 'docker-compose.gpu.yml') {
  $composeFiles += 'docker-compose.gpu.yml'
}
$composeArgs = @()
foreach ($file in $composeFiles) {
  $composeArgs += '-f'
  $composeArgs += $file
}

Write-Host "Building training pipeline images..."
docker compose @composeArgs build

Write-Host "Starting training pipeline (detached)..."
docker compose @composeArgs up -d

Write-Host "Tailing logs (Ctrl+C to stop viewing; container keeps running)..."
docker compose @composeArgs logs -f
