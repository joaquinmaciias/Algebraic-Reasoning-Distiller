#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  cache-cleanup-timer.sh --mode=1|2 --state-file=/run/control/var/cache-cleanup-time --timeout=3600 \
    --hook=/run/control/src/cache-cleanup-hook.sh --user=guest --home=/home/guest [--tick=10]
EOF
  exit 2
}

MODE=""
STATE_FILE=""
TIMEOUT=""
HOOK=""
TARGET_USER=""
TARGET_HOME=""
TICK_SECONDS="10"

LOG_DIR="${CONTROL_LOG_DIR:-/run/control/log}"
TIMER_LOG_FILE="${LOG_DIR}/cache-cleanup-timer.log"
HOOK_LOG_FILE="${CACHE_CLEANUP_LOG_FILE:-/run/control/log/cache-cleanup.log}"

for arg in "$@"; do
  case "${arg}" in
    --mode=*) MODE="${arg#*=}" ;;
    --state-file=*) STATE_FILE="${arg#*=}" ;;
    --timeout=*) TIMEOUT="${arg#*=}" ;;
    --hook=*) HOOK="${arg#*=}" ;;
    --user=*) TARGET_USER="${arg#*=}" ;;
    --home=*) TARGET_HOME="${arg#*=}" ;;
    --tick=*) TICK_SECONDS="${arg#*=}" ;;
    *) usage ;;
  esac
done

if [[ -z "${MODE}" || -z "${STATE_FILE}" || -z "${TIMEOUT}" || -z "${HOOK}" || -z "${TARGET_USER}" || -z "${TARGET_HOME}" ]]; then
  usage
fi

if [[ "${MODE}" != "1" && "${MODE}" != "2" ]]; then
  usage
fi

mkdir -p -- "$(dirname -- "${STATE_FILE}")" "${LOG_DIR}"

ts() { date -Is; }

lock_file="${STATE_FILE}.lock"

read_state() {
  if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}" 2>/dev/null || true
  else
    echo ""
  fi
}

init_state() {
  local raw
  raw="$(read_state)"
  if [[ ! "${raw}" =~ ^-?[0-9]+$ ]]; then
    printf '%s\n' "${TIMEOUT}" > "${STATE_FILE}"
    echo "$(ts) [INFO] Initialized state to ${TIMEOUT}" >>"${TIMER_LOG_FILE}"
  fi
}

has_foreground_tty_process() {
  ps -u "${TARGET_USER}" -o tty=,stat=,comm= 2>/dev/null \
    | awk '
        $1 ~ /^pts\// && $2 ~ /\+/ {
          cmd=$3
          if (cmd !~ /^-?(bash|sh|dash|zsh)$/) { found=1 }
        }
        END { exit(found?0:1) }
      '
}

run_cleanup() {
  echo "$(ts) [INFO] Triggering cleanup (running hook as ${TARGET_USER})" \
    >>"${TIMER_LOG_FILE}"
  chmod +x -- "${HOOK}" 2>/dev/null || true

  su - "${TARGET_USER}" -s /bin/bash -c "
    export CACHE_CLEANUP_LOG_FILE='${HOOK_LOG_FILE}';
    export CLEANUP_TARGET_USER='${TARGET_USER}';
    export CLEANUP_TARGET_HOME='${TARGET_HOME}';
    export CACHE_CLEANUP_STRATEGY='${MODE}';
    export CACHE_CLEANUP_TIME='${TIMEOUT}';
    exec /usr/bin/env bash '${HOOK}'
    # exec '${HOOK}'
  " >>"${TIMER_LOG_FILE}" 2>&1
}

init_state

echo "$(ts) [INFO] Started timer mode='${MODE}' state='${STATE_FILE}' timeout='${TIMEOUT}' tick='${TICK_SECONDS}' user='${TARGET_USER}'" \
  >>"${TIMER_LOG_FILE}"

while true; do
  sleep "${TICK_SECONDS}"

  if has_foreground_tty_process; then
    flock -x "${lock_file}" -c "printf '%s\n' '${TIMEOUT}' >'${STATE_FILE}'"
    echo "$(ts) [INFO] Foreground tty process detected; reset state to ${TIMEOUT}" >>"${TIMER_LOG_FILE}"
    continue
  fi

  if [[ "${MODE}" == "1" ]]; then
    # Strategy 1: on each idle tick, perform stale cleanup instead of updating countdown.
    if ! run_cleanup; then
      echo "$(ts) [ERROR] Cleanup hook failed (mode=1)" >>"${TIMER_LOG_FILE}"
    fi
    # Keep state initialized (non-counting, but present)
    flock -x "${lock_file}" -c "printf '%s\n' '${TIMEOUT}' >'${STATE_FILE}'"
    continue
  fi

  # Mode 2: normal countdown logic
  flock -x "${lock_file}" -c "
    raw=\$(cat '${STATE_FILE}' 2>/dev/null || true)
    case \"\$raw\" in
      ''|*[!0-9-]* ) raw='${TIMEOUT}' ;;
    esac
    next=\$(( raw - ${TICK_SECONDS} ))
    printf '%s\n' \"\$next\" >'${STATE_FILE}'
  "

  remaining="$(read_state)"
  if [[ "${remaining}" =~ ^-?[0-9]+$ ]] && (( remaining <= 0 )); then
    if ! run_cleanup; then
      echo "$(ts) [ERROR] Cleanup hook failed (mode=2; continuing; resetting anyway)" >>"${TIMER_LOG_FILE}"
    fi
    flock -x "${lock_file}" -c "printf '%s\n' '${TIMEOUT}' >'${STATE_FILE}'"
    echo "$(ts) [INFO] Countdown reset to ${TIMEOUT} after cleanup" >>"${TIMER_LOG_FILE}"
  fi
done
