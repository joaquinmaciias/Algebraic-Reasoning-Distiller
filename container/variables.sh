# DOCKER
COMPOSE_PROJECT_NAME="TO BE DEFINED IN __update_compose_project_name"
CONTAINER_USER="guest"
INSTALL_DOCKER=true

# CONTROL
COMPOSE_STATE_DIR="/run/control/var/compose-running"
CLEANUP_TIMER_FILE="/run/control/var/cache-cleanup-time"
CACHE_CLEANUP_TIME=86400
# 0 -> None, 1 -> Pruning files/dirs unused for CACHE_CLEANUP_TIME, 2 -> If container inactive (no runing processes/commands) for CACHE_CLEANUP_TIME, then clean EVERYTHING in cache
CACHE_CLEANUP_STRATEGY=1

# UV
PERSISTENT_UV_CACHE=true

# HOST PATHS
HOST_VOLUME_PATH="/home/${USER}/workdata/"
DATA_SUBDIR="data/"
SSH_SUBDIR="ssh/"
UV_CACHE_SUBDIR="uv_cache/"
CACHE_SUBDIR="cache/"


# DEFINE DYNAMIC (USER-UNIQUE & DIR-UNIQUE) COMPOSE_PROJECT_NAME
__update_compose_project_name() {
  local user="$USER"
  local dir
  dir="$(basename "$PWD")"

  COMPOSE_PROJECT_NAME="$(
    printf '%s-%s' "$user" "$dir" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's|[^a-z0-9_-]|-|g'
  )"
  export COMPOSE_PROJECT_NAME
}
__update_compose_project_name
if [[ ";$PROMPT_COMMAND;" == *";__update_compose_project_name;"* ]]; then
  :
elif [[ -z "$PROMPT_COMMAND" ]]; then
  PROMPT_COMMAND="__update_compose_project_name"
else
  PROMPT_COMMAND="__update_compose_project_name; $PROMPT_COMMAND"
fi

# EXPORT VARIABLES
export COMPOSE_PROJECT_NAME CONTAINER_USER INSTALL_DOCKER
export COMPOSE_STATE_DIR CLEANUP_TIMER_FILE CACHE_CLEANUP_TIME CACHE_CLEANUP_STRATEGY
export PERSISTENT_UV_CACHE
export HOST_VOLUME_PATH DATA_SUBDIR SSH_SUBDIR UV_CACHE_SUBDIR CACHE_SUBDIR
