#!/usr/bin/env bash
set -Eeuo pipefail

# cache-cleanup-hook.sh
#
# Strategy 2: delete everything under ~/.cache (but never delete the root dir)
# Strategy 1: delete only files not accessed in the last CACHE_CLEANUP_TIME seconds
#             (but never delete the root dir)

: "${CLEANUP_TARGET_USER:?CLEANUP_TARGET_USER is required}"
: "${CLEANUP_TARGET_HOME:?CLEANUP_TARGET_HOME is required}"

LOG_FILE="${CACHE_CLEANUP_LOG_FILE:-/run/control/log/cache-cleanup.log}"
touch "${LOG_FILE}" 2>/dev/null || true

log() {
  local msg="$1"
  printf '%s\n' "${msg}" >>"${LOG_FILE}" 2>/dev/null || true
}

ts() { date -Is; }

if [[ -z "${CLEANUP_TARGET_HOME}" || "${CLEANUP_TARGET_HOME}" == "/" ]]; then
  log "$(ts) [ERROR] Refusing to run: CLEANUP_TARGET_HOME='${CLEANUP_TARGET_HOME}'"
  exit 1
fi

cache_link="${CLEANUP_TARGET_HOME}/.cache"

# Ensure the path exists as a directory or symlink; if missing, recreate a directory.
if [[ ! -e "${cache_link}" ]]; then
  mkdir -p -- "${cache_link}" 2>/dev/null || true
fi

# Resolve symlink target so find traverses the real directory.
resolved_cache_dir="$(readlink -f -- "${cache_link}" 2>/dev/null || true)"
if [[ -z "${resolved_cache_dir}" || "${resolved_cache_dir}" == "/" ]]; then
  log "$(ts) [ERROR] Refusing to run: resolved_cache_dir='${resolved_cache_dir}' from cache_link='${cache_link}'"
  exit 1
fi

# Ensure the resolved directory exists (important when ~/.cache is a symlink).
mkdir -p -- "${resolved_cache_dir}" 2>/dev/null || true

if [[ ! -d "${resolved_cache_dir}" ]]; then
  log "$(ts) [ERROR] Cache path is not a directory: cache_link='${cache_link}' resolved='${resolved_cache_dir}'"
  exit 1
fi

strategy="${CACHE_CLEANUP_STRATEGY:-2}"
timeout_s="${CACHE_CLEANUP_TIME:-3600}"

log "$(ts) [INFO] Hook start strategy='${strategy}' user='${CLEANUP_TARGET_USER}' cache_link='${cache_link}' resolved='${resolved_cache_dir}' timeout='${timeout_s}'"

case "${strategy}" in
  2)
    # Delete everything inside the cache dir, but NEVER the root itself.
    if find "${resolved_cache_dir}" -xdev -mindepth 1 -delete; then
      log "$(ts) [INFO] Cache cleaned successfully (strategy=2)"
    else
      log "$(ts) [ERROR] Cache cleanup failed (strategy=2)"
      exit 1
    fi
    ;;
  1)
    now_s="$(date +%s)"
    cutoff_s="$(( now_s - timeout_s ))"

    deleted_files="0"

    while IFS= read -r -d '' f; do
      atime_s="$(stat -c %X -- "${f}" 2>/dev/null || echo "")"
      if [[ "${atime_s}" =~ ^[0-9]+$ ]] && (( atime_s < cutoff_s )); then
        rm -f -- "${f}" 2>/dev/null || true
        deleted_files="$(( deleted_files + 1 ))"
      fi
    done < <(find "${resolved_cache_dir}" -xdev -type f -print0 2>/dev/null || true)

    # Delete empty directories INSIDE the cache dir, but never the cache root.
    find "${resolved_cache_dir}" -xdev -mindepth 1 -type d -empty -delete 2>/dev/null || true

    log "$(ts) [INFO] Stale cleanup done (strategy=1) deleted_files=${deleted_files} cutoff_epoch=${cutoff_s}"
    ;;
  *)
    log "$(ts) [WARN] Unknown strategy='${strategy}', doing nothing"
    ;;
esac

# Final guarantee: keep the cache root directory present.
mkdir -p -- "${resolved_cache_dir}" 2>/dev/null || true
