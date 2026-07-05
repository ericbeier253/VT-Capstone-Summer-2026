# This portable gitbash script was generated using ChatGPT
# Just ensure Docker Desktop is running
# Watch out for CRLF line endings from Windows editors

#!/usr/bin/env bash

set -euo pipefail

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop."
    exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    open -a Docker
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.ini"

CONTAINER_NAME=$(awk -F= '
BEGIN{section=""}
{
    gsub(/\r/, "")
    if ($0 ~ /^\[postgres\]/) section="postgres"
    else if ($0 ~ /^\[/) section=""
    else if (section=="postgres" && $1 ~ /container_name/) {
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        print $2
        exit
    }
}' "$CONFIG_FILE")

if [[ -z "$CONTAINER_NAME" ]]; then
    echo "ERROR: container_name not found."
    exit 1
fi

echo "Restarting container: $CONTAINER_NAME"

docker restart "$CONTAINER_NAME"

docker ps --filter "name=${CONTAINER_NAME}"

echo "Database restarted."