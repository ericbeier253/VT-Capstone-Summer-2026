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

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.ini not found."
    exit 1
fi

CONTAINER_NAME=""

while IFS='=' read -r key value || [[ -n "$key" ]]; do
    key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ "$key" =~ ^\[(.*)\]$ ]]; then
        section="${BASH_REMATCH[1]}"
        continue
    fi

    [[ "${section:-}" != "postgres" ]] && continue

    if [[ "$key" == "container_name" ]]; then
        CONTAINER_NAME="$value"
        break
    fi
done < "$CONFIG_FILE"

if [[ -z "$CONTAINER_NAME" ]]; then
    echo "ERROR: container_name not found in config.ini"
    exit 1
fi

echo "Starting container: $CONTAINER_NAME"
docker start "$CONTAINER_NAME"

echo "Database started."

docker ps --filter "name=${CONTAINER_NAME}"