#!/usr/bin/env bash

source variables.sh

# echo "[attach] Waiting for docker compose service '${COMPOSE_PROJECT_NAME}' to become ready..."
# # wait until container is up and state file exists (max 5s)
# if ! timeout 5 bash -c '
#   until docker compose -p "'"${COMPOSE_PROJECT_NAME}"'" exec -T ubuntu \
#     test -f "'"${COMPOSE_STATE_DIR}"'" >/dev/null 2>&1; do
#     sleep 0.5
#   done
# '; then
#   echo "[attach] No docker compose project found with name '${COMPOSE_PROJECT_NAME}' after 5 seconds. Aborting."
#   return 1
# fi

# echo "[attach] Waiting for docker compose service '${COMPOSE_PROJECT_NAME}' to become ready..."
# # wait until container is up and state file exists (max 5s)
# if ! timeout 5 bash -c '
#   until docker compose -p "'"${COMPOSE_PROJECT_NAME}"'" exec -T ubuntu \
#     test -f "'"${COMPOSE_STATE_DIR}"'" >/dev/null 2>&1; do
#     sleep 0.5
#   done
# '; then
#   # if it became ready right after the timeout, do NOT abort / print abort message
#   if ! docker compose -p "${COMPOSE_PROJECT_NAME}" exec -T ubuntu \
#     test -f "${COMPOSE_STATE_DIR}" >/dev/null 2>&1; then
#     echo "[attach] No docker compose project found with name '${COMPOSE_PROJECT_NAME}' after 5 seconds. Aborting."
#     return 1 2>/dev/null || exit 1
#   fi
# fi

echo "[attach] Waiting for docker compose service '${COMPOSE_PROJECT_NAME}' to become ready..."
# fast path: already ready -> don't wait
if ! docker compose -p "${COMPOSE_PROJECT_NAME}" exec -T ubuntu \
  test -f "${COMPOSE_STATE_DIR}" >/dev/null 2>&1; then
  # wait until container is up and state file exists (max 5s)
  if ! timeout 5 bash -c '
    until docker compose -p "'"${COMPOSE_PROJECT_NAME}"'" exec -T ubuntu \
      test -f "'"${COMPOSE_STATE_DIR}"'" >/dev/null 2>&1; do
      sleep 0.5
    done
  '; then
    # only abort if it still isn't ready right now
    if ! docker compose -p "${COMPOSE_PROJECT_NAME}" exec -T ubuntu \
      test -f "${COMPOSE_STATE_DIR}" >/dev/null 2>&1; then
      echo "[attach] No docker compose project found with name '${COMPOSE_PROJECT_NAME}' after 5 seconds. Aborting."
      return 1 2>/dev/null || exit 1
    fi
  fi
fi

echo "[attach] Container is up. Waiting for initialization to finish..."
# wait until state becomes "false"
while true; do
  value="$(
    docker compose -p "${COMPOSE_PROJECT_NAME}" exec -T ubuntu \
      cat "${COMPOSE_STATE_DIR}" 2>/dev/null || true
  )"
  [[ "${value}" == "false" ]] && break
  sleep 0.5
done
echo "[attach] Initialization complete. Attaching interactive shell."


# attach interactive shell
docker compose -p "${COMPOSE_PROJECT_NAME}" exec -it \
  -u "${CONTAINER_USER}" \
  -w "/home/${CONTAINER_USER}" \
  ubuntu \
  bash -i
