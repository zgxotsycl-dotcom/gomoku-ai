$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot/..
Write-Host "Building and starting with Docker Compose..."
docker compose up -d --build
Write-Host "Trainer is running. Tail logs with: docker compose logs -f"

