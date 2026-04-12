#!/bin/bash
# Enter the running Algebraic-Reasoning-Distiller container.
# Usage: ./conectar.sh [usuario]
set -euo pipefail

USUARIO=${1:-$USER}
CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep "^${USUARIO}-" | head -1)

if [ -z "$CONTAINER_NAME" ]; then
    echo "ERROR: No running container found for user '${USUARIO}'."
    echo "Start it first with: docker compose up -d"
    exit 1
fi

echo "Connecting to container: $CONTAINER_NAME"
docker exec -it "${CONTAINER_NAME}" bash
