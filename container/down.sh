#!/usr/bin/env bash
source variables.sh

UV_CACHE_PATH="${HOST_VOLUME_PATH}${UV_CACHE_SUBDIR}"

docker compose -p "${COMPOSE_PROJECT_NAME}" down

if [[ "${PERSISTENT_UV_CACHE}" != "true" ]]; then
  if [[ -d "${UV_CACHE_PATH}" ]]; then
    echo "[down] Removing uv cache at ${UV_CACHE_PATH}"
    rm -rf "${UV_CACHE_PATH}"
  fi
else
  echo "[down] Preserving uv cache at ${UV_CACHE_PATH}"
fi