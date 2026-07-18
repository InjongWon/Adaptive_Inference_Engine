#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/../.venv/bin/python" -m uvicorn app.api:app \
  --host 0.0.0.0 \
  --port "${GATEWAY_PORT:-8080}" \
  --reload