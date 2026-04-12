#!/bin/bash
# Start (or rebuild) the Algebraic-Reasoning-Distiller container.
set -euo pipefail

docker compose up -d --build
docker ps --filter "name=202110004-AlgebraicDistiller"
