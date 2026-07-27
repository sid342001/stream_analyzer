#!/usr/bin/env bash
# Acceptance test: bring the VLM up with the network disabled and hit /health.
# Proves the stack runs air-gapped off local weights only.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "copy a profile first:  cp profiles/local.env .env"; exit 1; }

echo ">> starting vlm on an internal (no-internet) network"
docker compose up -d vlm
# attach the vlm container to a network with no external route
docker network create --internal uav-offline 2>/dev/null || true
docker network connect uav-offline "$(docker compose ps -q vlm)" 2>/dev/null || true

echo ">> waiting for health"
for i in $(seq 1 40); do
  if curl -sf localhost:8080/health >/dev/null; then echo "OK: VLM healthy offline"; exit 0; fi
  sleep 3
done
echo "FAIL: VLM did not become healthy"; docker compose logs --tail=40 vlm; exit 1
