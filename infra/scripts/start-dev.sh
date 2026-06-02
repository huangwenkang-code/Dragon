#!/usr/bin/env bash
# Start dragon-engine in dev mode (all services via docker-compose)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$ROOT_DIR"

echo "=== Starting infra (Redis + Postgres + ChromaDB) ==="
docker-compose up -d redis postgres chromadb

echo "=== Waiting for infra ==="
sleep 3

echo "=== Starting graph-service ==="
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR:$ROOT_DIR/../TradingAgents-CN-main" \
  uvicorn services.graph-service.main:app \
  --host 0.0.0.0 --port 8002 --reload
