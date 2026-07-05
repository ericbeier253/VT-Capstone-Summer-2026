#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting FastAPI server..."

python -m uvicorn app:app --reload
#uvicorn app:app --reload