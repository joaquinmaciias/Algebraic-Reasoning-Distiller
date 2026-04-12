#!/usr/bin/env bash
# set -euo pipefail
source variables.sh
git config core.hooksPath .githooks

# Ensure required directories exist
mkdir -p "${HOST_VOLUME_PATH}/${DATA_SUBDIR}"
mkdir -p "${HOST_VOLUME_PATH}/${SSH_SUBDIR}"
mkdir -p "${HOST_VOLUME_PATH}/${UV_CACHE_SUBDIR}"
mkdir -p "${HOST_VOLUME_PATH}/${CACHE_SUBDIR}"

# (Optional but recommended) sane permissions for SSH dir
chmod 700 "${HOST_VOLUME_PATH}/${SSH_SUBDIR}" || true

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

if docker compose -p "${COMPOSE_PROJECT_NAME}" ps -q >/dev/null 2>&1 \
   && [ -n "$(docker compose -p "${COMPOSE_PROJECT_NAME}" ps -q)" ]; then
    docker compose -p "${COMPOSE_PROJECT_NAME}" down
fi

docker compose -p "${COMPOSE_PROJECT_NAME}" up -d --build
# docker compose -p "${COMPOSE_PROJECT_NAME}" build --no-cache --progress=plain
docker compose -p "${COMPOSE_PROJECT_NAME}" ps
