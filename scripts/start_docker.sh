#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Building and starting with Docker Compose..."
docker compose up -d --build
echo "Trainer is running. Tail logs with: docker compose logs -f"

