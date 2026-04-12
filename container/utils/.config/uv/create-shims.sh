#!/usr/bin/env bash
# create-shims.sh — generate uv shims for WSL/Linux
# Shims created in ~/.local/uv-shims:
#   python        -> uv run python (stable resolver; respects pins)
#   pip           -> uv pip (supports -p/--python)
#   version       -> prints: "uv X.Y.Z" and then "Python A.B.C" (or "(none)")
#   venv          -> create + activate .venv in the current shell (function)
#   lpin          -> print/set local pin (nearest .python-version ancestor)
#   gpin          -> print/set global pin
#   interpreters  -> list installed interpreters (deduped, patch-level; marks best match)
#   uncache       -> garbage-collect uv cache (keeps wheels for venv-installed dists)
#   lock          -> add deps to pyproject from env/src imports and run uv lock

__uv_shims__old_opts="$(set +o)"
__uv_shims__old_nounset="off"
__uv_shims__old_pipefail="off"

if [[ "$-" == *u* ]]; then
  __uv_shims__old_nounset="on"
fi
if set -o | awk '$1=="pipefail"{print $2; exit}' | grep -qx on; then
  __uv_shims__old_pipefail="on"
fi

set -uo pipefail


UV_SHIM_DIR="$HOME/.local/uv-shims"
mkdir -p "$UV_SHIM_DIR"

# -------------------------
# Helpers (host script)
# -------------------------
_read_global_pin() {
  for c in \
    "${XDG_CONFIG_HOME:-$HOME/.config}/uv/.python-version" \
    "$HOME/.uv/.python-version" \
    "$HOME/.config/uv/python/version"
  do
    [[ -f "$c" ]] && { sed -n '1p' "$c"; return 0; }
  done
  return 1
}

_pick_bin_for_pin() {
  # Given "3.12[.Z]" prefer python3.12 / python312 / python3.12; else python / python3
  local pin="${1:-}" mm cand
  if [[ "$pin" =~ ^([0-9]+)\.([0-9]+) ]]; then
    mm="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    for cand in "python$mm" "python${mm/./}" "python3.${BASH_REMATCH[2]}"; do
      if command -v "$cand" >/dev/null 2>&1; then
        command -v "$cand"; return 0
      fi
    done
  fi
  command -v python  >/dev/null 2>&1 && { command -v python;  return 0; }
  command -v python3 >/dev/null 2>&1 && { command -v python3; return 0; }
  return 1
}

TIMEOUT_BIN="$(command -v timeout 2>/dev/null || true)"
_tmo1() { [[ -n "$TIMEOUT_BIN" ]] && "$TIMEOUT_BIN" 1s "$@" || "$@"; }
_tmo2() { [[ -n "$TIMEOUT_BIN" ]] && "$TIMEOUT_BIN" 2s "$@" || "$@"; }


# -------------------------
# _uv_install_python (shim): ensure required python exists (interactive)
# -------------------------
cat > "$UV_SHIM_DIR/_uv_install_python" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

_is_tty() {
  [[ -t 0 && -t 1 ]]
}

_prompt_yn() {
  local prompt="${1:?}"
  local ans=""
  while true; do
    printf '%s' "${prompt}" >/dev/tty
    IFS= read -r ans </dev/tty || ans=""
    case "${ans}" in
      y|Y) echo "y"; return 0 ;;
      n|N) echo "n"; return 0 ;;
      *) ;;
    esac
  done
}

_find_boundary_root() {
  local d=""
  d="$(pwd -P 2>/dev/null || pwd)"
  while true; do
    if [[ -f "${d}/pyproject.toml" || -f "${d}/uv.toml" || -d "${d}/.git" ]]; then
      printf '%s\n' "${d}"
      return 0
    fi
    if [[ "${d}" == "/" ]]; then
      printf '/\n'
      return 0
    fi
    d="$(dirname -- "${d}")"
  done
}

_first_pin_in_dir_chain() {
  local root="${1:?}"
  local d=""
  local f=""

  d="$(pwd -P 2>/dev/null || pwd)"
  while true; do
    for f in "${d}/.python-versions" "${d}/.python-version"; do
      if [[ -f "${f}" ]]; then
        sed -n 's/[[:space:]]*$//; /^[[:space:]]*#/d; /^[[:space:]]*$/d; 1p' "${f}" \
          || true
        return 0
      fi
    done

    if [[ "${d}" == "${root}" || "${d}" == "/" ]]; then
      return 1
    fi
    d="$(dirname -- "${d}")"
  done
}

_any_python_exists() {
  # Align detection with what shims will actually do, but without downloads.
  # If uv can't run python quickly, treat as "no interpreter installed".
  if command -v timeout >/dev/null 2>&1; then
    timeout 1s env UV_PYTHON_DOWNLOADS=never uv --no-progress run python -V \
      >/dev/null 2>&1
    return $?
  fi

  env UV_PYTHON_DOWNLOADS=never uv --no-progress run python -V >/dev/null 2>&1
}

_pin_python_exists() {
  local pin="${1:?}"

  # Check if a Python matching the pin is already available without downloading.
  if command -v timeout >/dev/null 2>&1; then
    timeout 1s env UV_PYTHON_DOWNLOADS=never uv --no-progress python find "${pin}" \
      >/dev/null 2>&1
    return $?
  fi

  env UV_PYTHON_DOWNLOADS=never uv --no-progress python find "${pin}" >/dev/null 2>&1
}

_install_requested() {
  local req="${1:?}"
  uv --no-progress python install "${req}"
}

_install_latest_stable() {
  uv --no-progress --no-config python install
}

_uv_install_python() {
  if ! _is_tty; then
    return 0
  fi

  local root=""
  local pin=""
  root="$(_find_boundary_root)"
  pin="$(_first_pin_in_dir_chain "${root}" 2>/dev/null || true)"

  if [[ -n "${pin}" ]]; then
    # If the pinned interpreter is already installed, don't prompt.
    if _pin_python_exists "${pin}"; then
      return 0
    fi

    local ans=""
    ans="$(_prompt_yn "This project requires Python version ${pin}. Do you want to install it? [y/n]: ")"
    if [[ "${ans}" == "y" ]]; then
      _install_requested "${pin}"
      return 0
    fi
    return 1
  fi

  if _any_python_exists; then
    return 0
  fi

  # Only ask to install latest stable if explicitly enabled.
  if [[ "${ASK_TO_INSTALL_PYTHON:-false}" != "true" ]]; then
    return 0
  fi

  while true; do
    case "$(_prompt_yn "No Python interpreter installed. Do you want to install latest stable version? [y/n]: ")" in
      y)
        _install_latest_stable
        return 0
        ;;
      n)
        return 1
        ;;
    esac
  done
}

_uv_install_python "$@"
EOF
chmod +x "$UV_SHIM_DIR/_uv_install_python"



# -------------------------
# python (via uv run; resolves stable interpreter)
# -------------------------
cat > "$UV_SHIM_DIR/python" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

PY="$(uv run python -c 'import sys; print(sys.executable)')"
exec uv run --python "$PY" python "$@"
EOF
chmod +x "$UV_SHIM_DIR/python"


# -------------------------
# pip (maps to uv pip; accepts -p/--python)
# -------------------------
cat > "$UV_SHIM_DIR/pip" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

ARGS=(); PY=""
while [[ $# -gt 0 ]]; do
  case "${1-}" in
    -p|--python) shift || true; PY="${1-}";;
    *) ARGS+=("${1-}");;
  esac
  shift || true
done
if [[ -z "$PY" ]]; then
  PY="$(uv run python -c 'import sys; print(sys.executable)')"
fi
exec uv pip --no-progress "${ARGS[@]}" --python "$PY"
EOF
chmod +x "$UV_SHIM_DIR/pip"



# -------------------------
# version (two lines; matches uv resolver or active venv)
# -------------------------
cat > "$UV_SHIM_DIR/version" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

# 1) uv version
uv self version

# 2) Python version — prefer active venv; else uv-run; else heuristic; else "(none)"
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  exec "$VIRTUAL_ENV/bin/python" --version
fi

if PYVER="$(UV_PYTHON_DOWNLOADS=never uv --no-progress run python -V 2>/dev/null || true)"; then
  if [[ -n "$PYVER" ]]; then
    echo "$PYVER"
    exit 0
  fi
fi

_pick_bin_for_pin() {
  local pin="${1:-}" mm cand
  if [[ "$pin" =~ ^([0-9]+)\.([0-9]+) ]]; then
    mm="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    for cand in "python$mm" "python${mm/./}" "python3.${BASH_REMATCH[2]}"; do
      if command -v "$cand" >/dev/null 2>&1; then
        command -v "$cand"; return 0
      fi
    done
  fi
  command -v python  >/dev/null 2>&1 && { command -v python;  return 0; }
  command -v python3 >/dev/null 2>&1 && { command -v python3; return 0; }
  return 1
}

PIN=""
if [[ -f ".python-version" ]]; then
  PIN="$(sed -n '1p' .python-version)"
else
  for c in \
    "${XDG_CONFIG_HOME:-$HOME/.config}/uv/.python-version" \
    "$HOME/.uv/.python-version" \
    "$HOME/.config/uv/python/version"
  do
    [[ -f "$c" ]] && { PIN="$(sed -n '1p' "$c")"; break; }
  done
fi

BIN="$(_pick_bin_for_pin "$PIN" 2>/dev/null || true)"
if [[ -n "${BIN:-}" ]]; then
  exec "$BIN" --version
fi

echo "Python (none)"
EOF
chmod +x "$UV_SHIM_DIR/version"







# -------------------------
# venv (function): create/activate venv
# -------------------------
venv() {

    local _is_interactive="0"
    case "$-" in *i*) _is_interactive="1" ;; esac

    local _old_opts=""
    _old_opts="$(set +o)"
    if [[ "${_is_interactive}" == "0" ]]; then
        set -euo pipefail
    fi

    local _was_history="off"
    if [[ "$(set -o | awk '$1=="history"{print $2; exit}')" == "on" ]]; then
        _was_history="on"
    fi

    # Prevent `eval` restore from polluting interactive history.
    # Remove history toggles from the saved option script and restore history separately.
    local _old_opts_nohist=""
    _old_opts_nohist="$(printf '%s\n' "${_old_opts}" | sed '/^set [+-]o history$/d')"

    _restore_opts() {
        local opts="${1:?}"
        local was_history="${2:?}"

        # Disable history while applying many `set` commands.
        set +o history
        eval "${opts}"

        if [[ "${was_history}" == "on" ]]; then
            set -o history
        else
            set +o history
        fi
    }

    if command -v _uv_install_python >/dev/null 2>&1; then
        _uv_install_python || { _restore_opts "${_old_opts_nohist}" "${_was_history}"; return 1; }
    else
        local _shim_dir=""
        _shim_dir="${HOME}/.local/uv-shims"
        if [[ -x "${_shim_dir}/_uv_install_python" ]]; then
            "${_shim_dir}/_uv_install_python" || { _restore_opts "${_old_opts_nohist}" "${_was_history}"; return 1; }
        fi
    fi

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        printf '[venv] Already inside a virtual environment: %s\n' \
            "${VIRTUAL_ENV}" 1>&2
        _restore_opts "${_old_opts_nohist}" "${_was_history}"
        return 0
    fi

    local py=""
    local act=".venv"

    while [[ $# -gt 0 ]]; do
        case "${1-}" in
            -p|--python)
                shift || true
                if [[ $# -le 0 ]]; then
                    printf '[ERROR] missing value for --python\n' 1>&2
                    _restore_opts "${_old_opts_nohist}" "${_was_history}"
                    return 1
                fi
                py="${1-}"
                shift || true
                ;;
            --)
                shift || true
                break
                ;;
            -*)
                printf '[ERROR] unknown option: %s\n' "${1-}" 1>&2
                _restore_opts "${_old_opts_nohist}" "${_was_history}"
                return 1
                ;;
            *)
                act="${1-}"
                shift || true
                break
                ;;
        esac
    done

    local -a extra_args=()
    while [[ $# -gt 0 ]]; do
        extra_args+=("$1")
        shift || true
    done

    local activate_path="${act}/bin/activate"

    if [[ ! -f "${activate_path}" ]]; then
        local -a cmd=(uv --no-progress venv)
        if [[ -n "${py}" ]]; then
            cmd+=(--python "${py}")
        fi

        # Use absolute path only for creation output
        local act_abs=""
        if [[ "${act}" = /* ]]; then
            act_abs="${act}"
        else
            act_abs="$(pwd)/${act}"
        fi

        cmd+=("${extra_args[@]}" "${act_abs}")

        # Filter uv output: drop "Activate with" line
        "${cmd[@]}" 2>&1 | sed '/^Activate with:/d'
    fi

    if [[ ! -f "${activate_path}" ]]; then
        printf '[ERROR] expected %s not found\n' "${activate_path}" 1>&2
        _restore_opts "${_old_opts_nohist}" "${_was_history}"
        return 1
    fi

    # Activate in the current shell
    # shellcheck disable=SC1090
    source "${activate_path}"

    # ---- PROMPT FIX (matches `uv venv`) ----
    local dir_name=""
    dir_name="$(basename "$(pwd)")"

    export VIRTUAL_ENV_PROMPT="(${dir_name}) "

    if [[ -n "${_OLD_VIRTUAL_PS1:-}" ]]; then
        PS1="${VIRTUAL_ENV_PROMPT}${_OLD_VIRTUAL_PS1}"
    else
        PS1="${VIRTUAL_ENV_PROMPT}${PS1}"
    fi
    # ---------------------------------------

    _restore_opts "${_old_opts_nohist}" "${_was_history}"
}













# -------------------------
# lpin (local pin)
# -------------------------
cat > "$UV_SHIM_DIR/lpin" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

# If a version is provided, pin it here first.
if [[ $# -gt 0 ]]; then
  uv python pin "$1" >/dev/null
fi

# Find nearest ancestor (including PWD) that has .python-version
find_lpin_root() {
  local dir="$PWD"
  while :; do
    if [[ -f "$dir/.python-version" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    local parent="${dir%/*}"
    [[ "$dir" == "$parent" ]] && return 1   # reached /
    dir="$parent"
  done
}

# In a pinned dir → print just "<version>"
if [[ -f ".python-version" ]]; then
  head -n 1 .python-version
  exit 0
fi

# Under a pinned ancestor → print "(~path) <version>"
if root="$(find_lpin_root)"; then
  version="$(head -n 1 "$root/.python-version")"
  # Shorten $HOME prefix to "~" (no slash → "~data/…", as you wanted)
  display="$root"
  if [[ "$display" == "$HOME" ]]; then
    display="~"
  else
    display="${display/#$HOME\//~}"
  fi
  printf '(%s) %s\n' "$display" "$version"
else
  # No pinned parent → "(none)"
  echo "(none)"
fi
EOF
chmod +x "$UV_SHIM_DIR/lpin"


# -------------------------
# gpin (global pin)
# -------------------------
cat > "$UV_SHIM_DIR/gpin" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

if [[ $# -gt 0 ]]; then uv python pin "$1" --global >/dev/null; fi
for c in \
  "${XDG_CONFIG_HOME:-$HOME/.config}/uv/.python-version" \
  "$HOME/.uv/.python-version" \
  "$HOME/.config/uv/python/version"
do
  [[ -f "$c" ]] && { sed -n '1p' "$c"; exit 0; }
done
echo "(none)"
EOF
chmod +x "$UV_SHIM_DIR/gpin"

# -------------------------
# interpreters (fast + resilient; patch-level; dedup; smart star)
# -------------------------
cat > "$UV_SHIM_DIR/interpreters" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

TIMEOUT_BIN="$(command -v timeout 2>/dev/null || true)"
tmo1() { [[ -n "$TIMEOUT_BIN" ]] && "$TIMEOUT_BIN" 1s "$@" || "$@"; }
tmo2() { [[ -n "$TIMEOUT_BIN" ]] && "$TIMEOUT_BIN" 2s "$@" || "$@"; }

_resolve() { readlink -f "$1" 2>/dev/null || realpath -e "$1" 2>/dev/null || printf '%s' "$1"; }
_should_skip_path() { local p="$1"; [[ "${UV_SHIMS_SCAN_MNT:-0}" = 0 && "$p" == /mnt/* ]] && return 0; return 1; }
array_push() { eval "$1+=(\"\$2\")"; }

# Current interpreter (respect pins; avoid downloads)
CUR=""
if CUR_PATH="$(UV_PYTHON_DOWNLOADS=never uv --no-progress run python - <<'PY' 2>/dev/null || true
import sys; print(sys.executable)
PY
)"; then CUR="$(_resolve "$CUR_PATH")"; fi

# Read pins for star selection
_read_global_pin() {
  for c in \
    "${XDG_CONFIG_HOME:-$HOME/.config}/uv/.python-version" \
    "$HOME/.uv/.python-version" \
    "$HOME/.config/uv/python/version"
  do [[ -f "$c" ]] && { sed -n '1p' "$c"; return 0; }; done
  return 1
}
LPIN=""; GPIN=""
[[ -f ".python-version" ]] && LPIN="$(sed -n '1p' .python-version || true)"
GPIN="$(_read_global_pin || true)"
PIN="${LPIN:-$GPIN}"

PIN_FULL=""; PIN_PREFIX=""
if [[ "$PIN" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  PIN_FULL="cpython-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
  PIN_PREFIX="cpython-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
elif [[ "$PIN" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  PIN_PREFIX="cpython-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
elif [[ "$PIN" =~ ^(cpython-[0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
  PIN_FULL="${BASH_REMATCH[1]}"
  PIN_PREFIX="${PIN_FULL%.*}"
elif [[ "$PIN" =~ ^(cpython-[0-9]+\.[0-9]+)$ ]]; then
  PIN_PREFIX="${BASH_REMATCH[1]}"
fi

declare -a SHORTS PATHS
declare -A SEEN   # by resolved sys.executable
MAXLEN=0
LIST_OK=0

_add_record() {
  local path="$1" ver="$2"
  local real_norm="$(_resolve "$path")"
  [[ -z "${SEEN[$real_norm]:-}" ]] || return 0
  SEEN["$real_norm"]=1
  local short="cpython-$ver"
  array_push SHORTS "$short"
  array_push PATHS  "$real_norm"
  (( ${#short} > MAXLEN )) && MAXLEN=${#short}
}

parse_line() {
  local ent="$1" rawp="$2"
  [[ -n "$rawp" ]] || return 1
  local p="${rawp#"${rawp%%[![:space:]]*}"}"
  p="${p%% *->*}"
  [[ -n "$p" ]] || return 1
  if [[ "$p" != /* ]]; then p="$(realpath -m -- "$p" 2>/dev/null || printf '%s' "$p")"; fi
  _should_skip_path "$p" && return 1
  [[ -x "$p" ]] || return 1

  local real_py ver real_norm
  real_py="$(tmo1 "$p" -c 'import sys;print(sys.executable)' 2>/dev/null || true)" || true
  [[ -n "$real_py" ]] || return 1
  ver="$(tmo1 "$p" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")' 2>/dev/null || true)" || true
  [[ -n "$ver" ]] || return 1
  real_norm="$(_resolve "$real_py")"
  _add_record "$real_norm" "$ver"
}

if OUT="$(tmo2 uv --no-progress python list --only-installed 2>/dev/null || true)"; then
  if [[ -n "$OUT" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      a="${line%%[[:space:]]*}"; b="${line#"$a"}"; b="${b##[[:space:]]}"
      parse_line "$a" "$b" || true
    done <<<"$OUT"
    LIST_OK=1
  fi
fi

# fallback probing (PATH + uv-managed) if the list call wasn't usable
_add_entry_probe() {
  local bin="$1"; [[ -x "$bin" ]] || return 0; _should_skip_path "$bin" && return 0
  local real_py ver real_norm
  real_py="$(tmo1 "$bin" -c 'import sys;print(sys.executable)' 2>/dev/null || true)" || true
  [[ -n "$real_py" ]] || return 0
  ver="$(tmo1 "$bin" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")' 2>/dev/null || true)" || true
  [[ -n "$ver" ]] || return 0
  real_norm="$(_resolve "$real_py")"
  _add_record "$real_norm" "$ver"
}
if [[ "$LIST_OK" -eq 0 ]]; then
  if [[ "${UV_SHIMS_FAST:-0}" = "1" ]]; then
    for cand in python python3 python3.12 python3.11 python3.10; do
      bin="$(command -v "$cand" 2>/dev/null || true)"
      [[ -n "$bin" ]] && _add_entry_probe "$bin"
    done
  else
    while IFS= read -r cand; do
      bin="$(command -v "$cand" 2>/dev/null || true)"
      [[ -n "$bin" ]] && _add_entry_probe "$bin"
    done < <(LC_ALL=C compgen -c | grep -E '^python(3(\.[0-9]+)?)?$|^python3\.[0-9]+$' | sort -u)
    if [[ "${UV_SHIMS_SKIP_UVDATA:-0}" != 1 ]]; then
      _uv_data="${XDG_DATA_HOME:-$HOME/.local/share}/uv/python"
      if [[ -d "$_uv_data" ]]; then
        while IFS= read -r p; do _add_entry_probe "$p"; done < <(
          find "$_uv_data" -type f -path '*/bin/python*' -executable 2>/dev/null | sort -u
        )
      fi
    fi
  fi
fi

# Sort by version and decide star index
if ((${#SHORTS[@]}==0)); then
  echo "(no interpreters found)"; exit 0
fi
mapfile -t KEYS < <(printf '%s\n' "${!SHORTS[@]}")
IFS=$'\n' KEYS=($(for i in "${KEYS[@]}"; do echo -e "${SHORTS[i]}\t$i"; done | sort -V | awk -F'\t' '{print $2}')); unset IFS

STAR_IDX=-1
# prefer current interpreter
if [[ -n "$CUR" ]]; then
  for i in "${KEYS[@]}"; do
    [[ "$CUR" == "${PATHS[i]}" ]] && { STAR_IDX="$i"; break; }
  done
fi
# exact pinned X.Y.Z
if [[ "$STAR_IDX" -lt 0 && -n "$PIN_FULL" ]]; then
  for i in "${KEYS[@]}"; do
    [[ "${SHORTS[i]}" == "$PIN_FULL" ]] && STAR_IDX="$i"
  done
fi
# highest patch for pinned X.Y
if [[ "$STAR_IDX" -lt 0 && -n "$PIN_PREFIX" ]]; then
  for i in "${KEYS[@]}"; do
    [[ "${SHORTS[i]}" == ${PIN_PREFIX}* ]] && STAR_IDX="$i"
  done
fi

for i in "${KEYS[@]}"; do
  mark=' '
  [[ "$i" == "$STAR_IDX" ]] && mark='*'
  printf "%s %s %s\n" "$mark" "${SHORTS[i]}" "${PATHS[i]}"
done
EOF
chmod +x "$UV_SHIM_DIR/interpreters"







# -------------------------
# uncache (shim): uncache uv cache using hardlink GC + venv-installed wheels keep
# -------------------------
cat > "$UV_SHIM_DIR/uncache" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=false "${UV_SHIM_DIR}/_uv_install_python" || exit 1

: "${UV_CACHE_DIR:=$HOME/.cache/uv}"

DEBUG=0  # set to 1 to activate logs
log() {
  if (( DEBUG )); then
    printf '[uv-cache-gc] %s\n' "$*" >&2
  fi
  return 0
}

normalize_project_name() {
  local name="$1"
  name="${name,,}"
  name="${name//_/-}"
  name="${name//./-}"
  name="${name//+/-}"
  printf '%s' "$name"
}

has_linked_files() {
  local dir="$1"
  find "$dir" -type f -print0 2>/dev/null \
    | xargs -0 -r stat -c '%h' -- 2>/dev/null \
    | awk '$1 > 1 { found=1; exit } END { exit !found }'
}

extract_names_from_archive_object() {
  # Print normalized dist names found in *.dist-info/METADATA under obj_dir
  local obj_dir="$1"

  mapfile -d '' -t metadata_files < <(
    find "$obj_dir" -type f -name METADATA -print0 2>/dev/null \
      | tr '\0' '\n' \
      | awk '$0 ~ /\.dist-info\/METADATA$/ { print }' \
      | tr '\n' '\0'
  )

  log "  METADATA files found: ${#metadata_files[@]} (under $obj_dir)"
  if (( ${#metadata_files[@]} > 0 )); then
    local show_n=5
    if (( ${#metadata_files[@]} < show_n )); then
      show_n="${#metadata_files[@]}"
    fi
    local i=0
    for ((i = 0; i < show_n; i++)); do
      log "    METADATA: ${metadata_files[$i]}"
    done
  fi

  local metadata_path=""
  for metadata_path in "${metadata_files[@]}"; do
    local dist_name=""
    dist_name="$(
      awk -F': *' 'tolower($1)=="name" { print $2; exit }' "$metadata_path" \
        || true
    )"

    if [[ -z "$dist_name" ]]; then
      log "    WARN: could not parse Name: from $metadata_path"
      continue
    fi

    local norm=""
    norm="$(normalize_project_name "$dist_name")"
    log "    Parsed Name: '$dist_name' -> keep key: '$norm'"
    printf '%s\n' "$norm"
  done
}

if [[ ! -d "$UV_CACHE_DIR" ]]; then
  echo "UV cache directory does not exist: $UV_CACHE_DIR" >&2
  exit 1
fi

log "UV_CACHE_DIR=$UV_CACHE_DIR"

mapfile -d '' -t archive_roots < <(
  find "$UV_CACHE_DIR" -type d -name '*archive*' -print0 2>/dev/null
)
log "Found archive roots: ${#archive_roots[@]}"
for r in "${archive_roots[@]}"; do
  log "  archive_root: $r"
done

mapfile -d '' -t wheels_roots < <(
  find "$UV_CACHE_DIR" -type d -name '*wheels*' -print0 2>/dev/null
)
log "Found wheels roots: ${#wheels_roots[@]}"
for r in "${wheels_roots[@]}"; do
  log "  wheels_root: $r"
done

# -----------------------------------------------------------------------------
# 1) Prune archive objects that have ONLY single-linked files (nlink == 1)
#    Then remove the archive root itself if it becomes empty.
# -----------------------------------------------------------------------------
for archive_root in "${archive_roots[@]}"; do
  [[ -d "$archive_root" ]] || continue

  mapfile -d '' -t obj_dirs < <(
    find "$archive_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null
  )
  log "Scanning archive root: $archive_root (objects=${#obj_dirs[@]})"

  for obj_dir in "${obj_dirs[@]}"; do
    [[ -d "$obj_dir" ]] || continue

    if has_linked_files "$obj_dir"; then
      log "KEEP archive object (has linked files): $obj_dir"
      continue
    fi

    log "DELETE archive object (all nlink==1): $obj_dir"
    rm -rf -- "$obj_dir"
  done

  if [[ -d "$archive_root" ]] && [[ -z "$(ls -A "$archive_root" 2>/dev/null)" ]]; then
    log "DELETE empty archive root: $archive_root"
    rmdir -- "$archive_root" 2>/dev/null || true
  fi
done

# -----------------------------------------------------------------------------
# 2) Build keep-list from installed packages in UV-managed venvs found under ~
#    (not recursive per project: only scan "<dir>/.venv/..." if it exists)
# -----------------------------------------------------------------------------
keep_projects_file="$(mktemp)"
trap 'rm -f -- "$keep_projects_file"' EXIT

metadata_count=0
name_count=0

mapfile -t candidate_dirs < <(
  find -L ~ \
    \( -type d -name '.*' -prune \) \
    -o \
    \( \
      \( -type f \( \
          -name '.venv' \
          -o -name 'uv.lock' \
          -o -name 'pyproject.toml' \
          -o -name 'requirements.txt' \
        \) -printf '%h\n' \
      \) \
      -o \
      \( -type d -name '.venv' -printf '%h\n' \) \
    \) \
    | sort -u
)

for dir in "${candidate_dirs[@]}"; do
  [[ -d "$dir" ]] || continue

  # UV-managed environment check: only consider "<dir>/.venv" (no recursion)
  if [[ ! -d "$dir/.venv" ]]; then
    continue
  fi
  if [[ ! -f "$dir/.venv/pyvenv.cfg" ]]; then
    continue
  fi

  while IFS= read -r -d '' metadata_path; do
    metadata_count=$((metadata_count + 1))

    dist_name="$(
      awk -F': *' 'tolower($1)=="name" { print $2; exit }' "$metadata_path" \
        || true
    )"
    if [[ -z "$dist_name" ]]; then
      log "  WARN: could not parse Name: from $metadata_path"
      continue
    fi

    norm="$(normalize_project_name "$dist_name")"
    printf '%s\n' "$norm" >>"$keep_projects_file"
    name_count=$((name_count + 1))
  done < <(
    find "$dir/.venv" -type f \
      -path '*/lib/python*/site-packages/*.dist-info/METADATA' \
      -print0 2>/dev/null
  )
done

if [[ -s "$keep_projects_file" ]]; then
  sort -u "$keep_projects_file" -o "$keep_projects_file"
fi

log "Venv METADATA files seen: $metadata_count"
log "Names extracted (pre-dedupe): $name_count"
log "Keep-list (deduped):"
if [[ -s "$keep_projects_file" ]]; then
  while IFS= read -r name; do
    log "  keep: $name"
  done <"$keep_projects_file"
else
  log "  (empty)"
fi

# -----------------------------------------------------------------------------
# 3) Under any "*wheels*"/pypi/, keep ONLY wheels for keep-list names
# -----------------------------------------------------------------------------
for wheels_root in "${wheels_roots[@]}"; do
  pypi_root="$wheels_root/pypi"
  if [[ ! -d "$pypi_root" ]]; then
    log "Skip wheels root without pypi/: $wheels_root"
    continue
  fi

  mapfile -d '' -t proj_dirs < <(
    find "$pypi_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null
  )
  log "Scanning wheels pypi root: $pypi_root (projects=${#proj_dirs[@]})"

  for proj_dir in "${proj_dirs[@]}"; do
    [[ -d "$proj_dir" ]] || continue
    proj_name="$(basename -- "$proj_dir")"

    if [[ -s "$keep_projects_file" ]] && grep -Fxq -- "$proj_name" "$keep_projects_file"; then
      log "KEEP wheels project dir: $proj_dir"
      continue
    fi

    log "DELETE wheels project dir: $proj_dir"
    rm -rf -- "$proj_dir"
  done
done

log "Done."
EOF
chmod +x "$UV_SHIM_DIR/uncache"












cat > "$UV_SHIM_DIR/lock" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

UV_SHIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env ASK_TO_INSTALL_PYTHON=true "${UV_SHIM_DIR}/_uv_install_python" || exit 1

deps_mode="env"
write_txt="false"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

parse_args() {
  local arg="" rest="" i=""

  for arg in "$@"; do
    case "$arg" in
      --deps=env) deps_mode="env" ;;
      --deps=src) deps_mode="src" ;;
      --deps=*) die "invalid value for --deps (expected env or src)" ;;
      --txt) write_txt="true" ;;
      -[!-]*)
        rest="${arg#-}"
        for ((i = 0; i < ${#rest}; i++)); do
          case "${rest:$i:1}" in
            t) write_txt="true" ;;
            s) deps_mode="src" ;;
            *) die "unknown flag: -${rest:$i:1}" ;;
          esac
        done
        ;;
      *)
        die "unknown argument: $arg"
        ;;
    esac
  done
}

parse_args "$@"

[[ -n "${VIRTUAL_ENV:-}" ]] || die "no active environment (VIRTUAL_ENV is not set)"
command -v uv >/dev/null 2>&1 || die "uv not found in PATH"

normalize_name() {
  local raw_name="${1:-}"
  raw_name="${raw_name,,}"
  raw_name="$(printf '%s' "$raw_name" | sed -E 's/[-_.]+/-/g')"
  printf '%s\n' "$raw_name"
}

is_nvidia_dependency() {
  local name="${1:-}"
  [[ "$name" == *nvidia* || "$name" == *cuda* ]]
}

find_site_packages_dirs() {
  local venv_dir="${1:?}"
  find "$venv_dir" -type d \
    \( -name "site-packages" -o -name "dist-packages" \) \
    -print 2>/dev/null \
    | LC_ALL=C sort -u
}

name_from_dist_info() {
  local dist_dir="${1:?}"
  local meta_path="$dist_dir/METADATA"
  local name_line="" raw_name=""

  if [[ -f "$meta_path" ]]; then
    name_line="$(grep -m1 -E '^Name:' "$meta_path" || true)"
    if [[ -n "$name_line" ]]; then
      raw_name="$(printf '%s' "$name_line" | sed -E 's/^Name:[[:space:]]*//')"
      raw_name="$(printf '%s' "$raw_name" | sed -E 's/[[:space:]]+$//')"
      if [[ -n "$raw_name" ]]; then
        normalize_name "$raw_name"
        return 0
      fi
    fi
  fi
  return 1
}

fallback_name_from_metadata_dir_basename() {
  local base="${1:?}"
  base="${base%.dist-info}"
  base="${base%.egg-info}"

  local prefix="${base%*-}"
  local suffix="${base##*-}"
  if [[ "$base" != "$prefix" && "$suffix" =~ ^[0-9] ]]; then
    base="$prefix"
  fi

  normalize_name "$base"
}

version_from_metadata() {
  local dist_name_norm="${1:?}"
  local sp="" meta_dir="" meta_name="" meta_norm="" vline=""

  while IFS= read -r sp; do
    while IFS= read -r meta_dir; do
      meta_name="$(basename "$meta_dir")"
      meta_name="${meta_name%.dist-info}"
      meta_name="${meta_name%-*}"
      meta_norm="$(normalize_name "$meta_name")"
      if [[ "$meta_norm" != "$dist_name_norm" ]]; then
        continue
      fi
      vline="$(grep -m1 -E '^Version:' "$meta_dir/METADATA" 2>/dev/null || true)"
      if [[ -n "$vline" ]]; then
        printf '%s\n' "${vline#Version: }"
        return 0
      fi
    done < <(find "$sp" -maxdepth 1 -type d -name "*.dist-info" -print 2>/dev/null)
  done <<<"$site_dirs"

  return 1
}

gather_deps_from_env() {
  local sp="" d=""
  while IFS= read -r sp; do
    find "$sp" -maxdepth 1 -type d -name "*.dist-info" -print 2>/dev/null \
      | while IFS= read -r d; do
          name_from_dist_info "$d" || fallback_name_from_metadata_dir_basename "$(basename "$d")"
        done
  done <<<"$site_dirs"
}

find_py_files_null() {
  find . -type f -name "*.py" \
    -not -path "./.*/*" \
    -not -path "./.venv/*" \
    -not -path "./__pycache__/*" \
    -not -path "./.git/*" \
    -not -path "./.dvc/*" \
    -print0 2>/dev/null
}

grep_import_lines_from_files() {
  xargs -0 -r grep -I -h -E '^[[:space:]]*(import|from)[[:space:]]+' 2>/dev/null || true
}

scan_imported_modules() {
  find_py_files_null \
    | grep_import_lines_from_files \
    | sed -E 's/#.*$//' \
    | awk '
        /^[[:space:]]*import[[:space:]]+/ {
          sub(/^[[:space:]]*import[[:space:]]+/, "", $0)
          n=split($0, parts, ",")
          for (i=1; i<=n; i++) {
            p=parts[i]
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", p)
            sub(/[[:space:]]+as[[:space:]].*$/, "", p)
            sub(/\..*$/, "", p)
            if (p != "") print p
          }
          next
        }
        /^[[:space:]]*from[[:space:]]+/ {
          sub(/^[[:space:]]*from[[:space:]]+/, "", $0)
          sub(/[[:space:]]+import[[:space:]].*$/, "", $0)
          sub(/\..*$/, "", $0)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
          if ($0 != "") print $0
          next
        }
      ' \
    | LC_ALL=C sort -u
}

module_exists_in_site_packages() {
  local mod="${1:?}"
  local sp=""
  while IFS= read -r sp; do
    if [[ -d "$sp/$mod" || -f "$sp/$mod.py" ]]; then
      return 0
    fi
  done <<<"$site_dirs"
  return 1
}

resolve_dist_for_module_record() {
  local mod="${1:?}"
  local sp="" meta_dir="" dist_name="" record=""

  while IFS= read -r sp; do
    while IFS= read -r meta_dir; do
      record="$meta_dir/RECORD"
      [[ -f "$record" ]] || continue
      if grep -q -E "^${mod}/__init__\.py,|^${mod}/|^${mod}\.py," "$record" 2>/dev/null; then
        dist_name="$(
          name_from_dist_info "$meta_dir" \
            || fallback_name_from_metadata_dir_basename "$(basename "$meta_dir")"
        )"
        printf '%s\n' "$dist_name"
        return 0
      fi
    done < <(find "$sp" -maxdepth 1 -type d -name "*.dist-info" -print 2>/dev/null)
  done <<<"$site_dirs"

  return 1
}

gather_deps_from_src() {
  local mod="" dist=""
  scan_imported_modules |
    while IFS= read -r mod; do
      [[ -n "$mod" ]] || continue
      if ! module_exists_in_site_packages "$mod"; then
        continue
      fi
      if dist="$(resolve_dist_for_module_record "$mod")"; then
        printf '%s\n' "$dist"
      fi
    done
}

project_root="$(pwd)"

if [[ ! -f "$project_root/pyproject.toml" ]]; then
  echo "No pyproject.toml found in: $project_root"
  echo "Initializing with: uv init"
  (
    cd "$project_root"
    uv init >/dev/null
  )
fi

cd "$project_root"

tmp_all="$(mktemp)"
tmp_names="$(mktemp)"
tmp_main="$(mktemp)"
tmp_cuda="$(mktemp)"
trap 'rm -f "$tmp_all" "$tmp_names" "$tmp_main" "$tmp_cuda"' EXIT

site_dirs="$(find_site_packages_dirs "$VIRTUAL_ENV")"
[[ -n "$site_dirs" ]] || die "could not locate site-packages under: $VIRTUAL_ENV"

if [[ "$deps_mode" == "src" ]]; then
  gather_deps_from_src >"$tmp_all"
else
  gather_deps_from_env >"$tmp_all"
fi

LC_ALL=C sort -u "$tmp_all" >"$tmp_names"

: >"$tmp_main"
: >"$tmp_cuda"
while IFS= read -r name; do
  [[ -n "$name" ]] || continue
  if is_nvidia_dependency "$name"; then
    printf '%s\n' "$name" >>"$tmp_cuda"
  else
    printf '%s\n' "$name" >>"$tmp_main"
  fi
done <"$tmp_names"

LC_ALL=C sort -u "$tmp_main" -o "$tmp_main" 2>/dev/null || true
LC_ALL=C sort -u "$tmp_cuda" -o "$tmp_cuda" 2>/dev/null || true

if [[ -s "$tmp_main" ]]; then
  xargs -r -a "$tmp_main" -n 50 uv add --raw --no-sync -- >/dev/null
fi

if [[ -s "$tmp_cuda" ]]; then
  xargs -r -a "$tmp_cuda" -n 50 uv add --raw --optional cuda --no-sync -- >/dev/null
fi

if [[ "$write_txt" == "true" ]]; then
  : > requirements.txt
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    ver="$(version_from_metadata "$name")" || die "could not determine version for $name"
    printf '%s==%s\n' "$name" "$ver" >> requirements.txt
  done < <(cat "$tmp_main" "$tmp_cuda" | LC_ALL=C sort -u)
fi

uv lock
EOF

chmod +x "$UV_SHIM_DIR/lock"






# -------------------------
# Ensure PATH contains ~/.local/uv-shims
# -------------------------
PROFILE="$HOME/.bashrc"
[[ -n "${ZSH_VERSION:-}" ]] && PROFILE="$HOME/.zshrc"
if ! grep -q '# uv shims$' "$PROFILE"; then
  printf '%s\n' '[[ ":$PATH:" != *":$HOME/.local/uv-shims:"* ]] && export PATH="$HOME/.local/uv-shims:$PATH"  # uv shims' >> "$PROFILE"
  echo "[INFO] Added ~/.local/uv-shims to PATH in $PROFILE"
fi


set +u
set +o pipefail
eval "${__uv_shims__old_opts}"
unset __uv_shims__old_opts __uv_shims__old_nounset __uv_shims__old_pipefail
