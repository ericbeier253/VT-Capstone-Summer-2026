#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===================================="
echo "Starting PostgreSQL..."
echo "===================================="

# Start database (your existing script)
bash "$BASE_DIR/setup-postgres.sh"

echo ""
echo "===================================="
echo "Starting FastAPI server..."
echo "===================================="

python -m uvicorn app:app --reload
#uvicorn app:app --reload