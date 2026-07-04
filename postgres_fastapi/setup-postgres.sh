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
    echo "ERROR: config.ini not found in ${SCRIPT_DIR}"
    exit 1
fi

# Default values
CONTAINER_NAME=""
DATABASE=""
USERNAME=""
PASSWORD=""
PORT="5432"
POSTGRES_VERSION="latest"

# Parse [postgres] section from config.ini
current_section=""

while IFS='=' read -r key value || [[ -n "$key" ]]; do
    key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    [[ -z "$key" ]] && continue
    [[ "$key" == \;* || "$key" == \#* ]] && continue

    if [[ "$key" =~ ^\[(.*)\]$ ]]; then
        current_section="${BASH_REMATCH[1]}"
        continue
    fi

    [[ "$current_section" != "postgres" ]] && continue

    case "$key" in
        container_name) CONTAINER_NAME="$value" ;;
        database) DATABASE="$value" ;;
        username) USERNAME="$value" ;;
        password) PASSWORD="$value" ;;
        port) PORT="$value" ;;
        postgres_version) POSTGRES_VERSION="$value" ;;
    esac
done < "$CONFIG_FILE"

# Validate required settings
for var in CONTAINER_NAME DATABASE USERNAME PASSWORD; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: Missing required config value: ${var}"
        exit 1
    fi
done

echo "Checking Docker installation..."

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed or not in PATH."
    exit 1
fi

echo "Checking Docker daemon..."

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is installed but not running."
    echo "Please start Docker and try again."
    exit 1
fi

echo "Pulling postgres:${POSTGRES_VERSION}..."
docker pull "postgres:${POSTGRES_VERSION}"

VOLUME_NAME="${CONTAINER_NAME}-data"

echo "Ensuring volume exists: ${VOLUME_NAME}"

if ! docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
    docker volume create "$VOLUME_NAME" >/dev/null
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Creating PostgreSQL container..."

docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${PORT}:5432" \
    -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
    -e POSTGRES_DB="$DATABASE" \
    -e POSTGRES_USER="$USERNAME" \
    -e POSTGRES_PASSWORD="$PASSWORD" \
    "postgres:${POSTGRES_VERSION}" >/dev/null

echo
echo "========================================="
echo "PostgreSQL container started successfully"
echo "========================================="
echo "Container : $CONTAINER_NAME"
echo "Database  : $DATABASE"
echo "User      : $USERNAME"
echo "Host      : localhost"
echo "Port      : $PORT"
echo "Version   : $POSTGRES_VERSION"
echo "Volume    : $VOLUME_NAME"
echo
echo "Connection string:"
echo "postgresql://${USERNAME}:${PASSWORD}@localhost:${PORT}/${DATABASE}"
echo

docker ps --filter "name=${CONTAINER_NAME}"