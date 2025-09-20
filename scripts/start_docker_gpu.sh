#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[GPU] Building and starting trainer with CUDA support..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

echo "[GPU] Containers started. Follow logs with:" 
echo "    docker compose logs -f trainer"

