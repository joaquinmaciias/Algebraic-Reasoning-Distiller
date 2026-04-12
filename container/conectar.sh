#!/bin/bash
# Enter the running Algebraic-Reasoning-Distiller container.
# Usage: ./conectar.sh [usuario]
set -euo pipefail

USUARIO=${1:-$USER}
CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep "${USUARIO}" | grep -i "algebraic\|distiller\|ard" | head -1)

# Fallback: search by image name
if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME=$(docker ps --format '{{.Names}}\t{{.Image}}' \
        | grep "algebraic-reasoning-distiller" | awk '{print $1}' | head -1)
fi

if [ -z "$CONTAINER_NAME" ]; then
    echo "ERROR: No running container found for user '${USUARIO}'."
    echo "Running containers:"
    docker ps --format '  {{.Names}}  ({{.Image}})'
    echo "Start it first with: docker compose up -d"
    exit 1
fi

echo "Connecting to container: $CONTAINER_NAME"
docker exec -it "${CONTAINER_NAME}" bash
