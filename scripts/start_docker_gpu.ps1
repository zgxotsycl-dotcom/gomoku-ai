$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot/..

param(
  [switch]$Quick
)

Write-Host "[GPU] Building and starting trainer with CUDA support..."
if ($Quick) {
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.quick.yml up -d --build
} else {
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
}

Write-Host "[GPU] Containers started. Checking logs..."
docker compose logs -f trainer
