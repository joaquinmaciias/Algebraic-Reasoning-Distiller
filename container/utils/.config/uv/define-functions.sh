#!/usr/bin/env bash
# create-shims.sh — uv shims as Bash functions (subshell-safe)
#
# After sourcing this file, you get these functions:
#   _uv_install_python
#   python, pip, version, venv
#   lpin, gpin, interpreters
#   uncache, lock
#
# Notes:
# - All functions except `venv` are defined as `name() ( ... )` so they always run
#   in a subshell with strict-mode + ERR trap.
# - `venv` is the only `{ ... }` function, and it does NOT run `set ...` at the
#   shell level. Heavy/fragile logic is done in subshell subfunctions.

# -------------------------
# Common strict-mode snippet (for subshell functions)
# -------------------------
__uv__strict() {
  set -Eeuo pipefail; shopt -s inherit_errexit 2>/dev/null || true
  trap 'echo "ERROR en ${FUNCNAME[0]:-MAIN} línea $LINENO: $BASH_COMMAND" >&2' ERR
}


# -------------------------
# Helpers (all subshell-style, per your rule)
# -------------------------

__uv__read_global_pin() (
  __uv__strict
  local c=""
  for c in \
    "${XDG_CONFIG_HOME:-$HOME/.config}/uv/.python-version" \
    "$HOME/.uv/.python-version" \
    "$HOME/.config/uv/python/version"
  do
    [[ -f "$c" ]] && { sed -n '1p' "$c"; return 0; }
  done
  return 1
)

# __uv__pick_bin_for_pin() (
#   __uv__strict
#   local pin="${1:-}" mm="" cand=""
#   if [[ "$pin" =~ ^([0-9]+)\.([0-9]+) ]]; then
#     mm="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
#     for cand in "python$mm" "python${mm/./}" "python3.${BASH_REMATCH[2]}"; do
#       if command -v "$cand" >/dev/null 2>&1; then
#         command -v "$cand"
#         return 0
#       fi
#     done
#   fi
#   command -v python  >/dev/null 2>&1 && { command -v python;  return 0; }
#   command -v python3 >/dev/null 2>&1 && { command -v python3; return 0; }
#   return 1
# )

# PYTHON CHECK

__uv__pick_bin_for_pin() (
  __uv__strict
  local pin="${1:-}"
  [[ -n "$pin" ]] || return 1

  # Intenta resolver el pin a un intérprete "managed" instalado.
  UV_PYTHON_DOWNLOADS=never uv --no-progress python find --managed-python "$pin" 2>/dev/null \
    || UV_PYTHON_DOWNLOADS=never uv --no-progress python find "$pin" 2>/dev/null
)






# -------------------------
# _uv_install_python
# -------------------------
_uv_install_python() (
  __uv__strict

  local allow_latest="${1:-}"
  if [[ "$allow_latest" == "true" ]]; then
    export ASK_TO_INSTALL_PYTHON="true"
  fi

  _is_tty() (
    __uv__strict
    [[ -t 0 && -t 1 ]]
  )

  _prompt_yn() (
    __uv__strict

    local prompt="${1:?missing prompt}"
    local ans=""
    while true; do
      printf '%s' "$prompt" >/dev/tty
      IFS= read -r ans </dev/tty || ans=""
      case "$ans" in
        y|Y) printf 'y\n'; return 0 ;;
        n|N) printf 'n\n'; return 0 ;;
        *) ;;
      esac
    done
  )

  _tmo() (
    __uv__strict

    local seconds="${1:?missing seconds}"
    shift
    if command -v timeout >/dev/null 2>&1; then
      timeout "$seconds" "$@"
    else
      "$@"
    fi
  )

  # Return 0 and print boundary root if a "project boundary" exists.
  # If none exists, return 1 (so callers can avoid scanning/prompts).
  _find_boundary_root() (
    __uv__strict

    local d=""
    d="$(pwd -P 2>/dev/null || pwd)"

    while true; do
        if [[ -f "$d/pyproject.toml" || -f "$d/uv.toml" || -d "$d/.git" \
           || -f "$d/.python-version" || -f "$d/.python-versions" ]]; then
        printf '%s\n' "$d"
        return 0
      fi
      if [[ "$d" == "/" ]]; then
        return 1
      fi
      d="$(dirname -- "$d")"
    done
  )

  _first_pin_in_dir_chain() (
    __uv__strict

    local root="${1:?missing root}"
    local d="" f=""

    d="$(pwd -P 2>/dev/null || pwd)"
    while true; do
      for f in "$d/.python-versions" "$d/.python-version"; do
        if [[ -f "$f" ]]; then
          sed -n 's/[[:space:]]*$//; /^[[:space:]]*#/d; /^[[:space:]]*$/d; 1p' \
            "$f" 2>/dev/null || true
          return 0
        fi
      done

      if [[ "$d" == "$root" || "$d" == "/" ]]; then
        return 1
      fi
      d="$(dirname -- "$d")"
    done
  )

    _pin_is_meaningful() (
    __uv__strict

    local pin="${1:-}"
    [[ -n "$pin" ]] || return 1

    local p
    p="$(
        printf '%s' "$pin" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//'
    )"
    if [[ "$p" == "none" || "$p" == "off" || "$p" == "clear" \
       || "$p" == "unset" || "$p" == "remove" || "$p" == "rm" \
       || "$p" == "any-any-none-any-any" ]]; then
      return 1
    fi

    [[ "$p" != *"any-any-none-any-any"* ]]
    )


  _any_python_exists() (
    __uv__strict

    # Avoid `uv run python` (can trigger requirement prompts).
    local out=""
    out="$(_tmo 1s uv --no-progress python list --only-installed 2>/dev/null || true)"
    [[ -n "$out" ]]
  )

  _pin_python_exists() (
    __uv__strict

    local pin="${1:?missing pin}"
    _tmo 1s uv --no-progress python find --no-python-downloads --managed-python "$pin" >/dev/null 2>&1
  )

  _install_requested() (
    __uv__strict

    local req="${1:?missing req}"
    uv --no-progress python install "$req"
  )

  _install_latest_stable() (
    __uv__strict

    uv --no-progress --no-config python install
  )

  if ! _is_tty; then
    return 0
  fi

  # Only enforce a "required python" if we are inside a detected project boundary.
  local root=""
  if root="$(_find_boundary_root 2>/dev/null || true)"; then
    :
  fi

  if [[ -n "$root" ]]; then
    local pin=""
    pin="$(_first_pin_in_dir_chain "$root" 2>/dev/null || true)"

    if _pin_is_meaningful "$pin"; then
      if _pin_python_exists "$pin"; then
        return 0
      fi

      local ans=""
      ans="$(_prompt_yn \
        "This project requires Python version ${pin}. Do you want to install it? [y/n]: ")"
      if [[ "$ans" == "y" ]]; then
        _install_requested "$pin"
       # If install fails, propagate failure
       _pin_python_exists "$pin" || return 1
       return 0
      fi

      # User declined required interpreter: stop caller.
      return 1
    fi
  fi

  if _any_python_exists; then
    return 0
  fi

  if [[ "${ASK_TO_INSTALL_PYTHON:-false}" != "true" ]]; then
    return 0
  fi

  local ans2=""
  ans2="$(_prompt_yn \
    "No Python interpreter installed. Do you want to install latest stable version? [y/n]: ")"
  if [[ "$ans2" == "y" ]]; then
    _install_latest_stable
  fi

  # Again: user saying "no" is not an error.
  return 0
)











# -------------------------
# python
# -------------------------
python() (
  __uv__strict
  _uv_install_python true || exit 1
  local py=""
  py="$(uv run python -c 'import sys; print(sys.executable)')"
  exec uv run --python "${py}" python "$@"
)

# -------------------------
# pip (maps to uv pip; accepts -p/--python)
# -------------------------
pip() (
  __uv__strict

  _uv_install_python true

  local py=""
  declare -a args=()

  while [[ $# -gt 0 ]]; do
  if [[ "${1-}" == "-p" || "${1-}" == "--python" ]]; then
      shift || true
      py="${1-}"
  else
      args+=("${1-}")
  fi
  shift || true
  done

  if [[ -z "$py" ]]; then
    py="$(
      env UV_PYTHON_DOWNLOADS=never uv --no-progress python find 2>/dev/null \
        || uv --no-progress run python -c 'import sys; print(sys.executable)' 2>/dev/null
    )"
  fi

  exec uv pip --no-progress "${args[@]}" --python "$py"
)


# -------------------------
# version
# -------------------------
version() (
  __uv__strict
  _uv_install_python true || exit 1

  uv self version

  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    exec "$VIRTUAL_ENV/bin/python" --version
  fi

  local pyver=""
  pyver="$(UV_PYTHON_DOWNLOADS=never uv --no-progress run python -V 2>/dev/null || true)"
  if [[ -n "${pyver}" ]]; then
    printf '%s\n' "${pyver}"
    exit 0
  fi

  # local pin="" bin=""
  # if [[ -f ".python-version" ]]; then
  #   pin="$(sed -n '1p' .python-version || true)"
  # else
  #   pin="$(__uv__read_global_pin 2>/dev/null || true)"
  # fi

  # bin="$(__uv__pick_bin_for_pin "${pin}" 2>/dev/null || true)"
  # if [[ -n "${bin}" ]]; then
  #   exec "${bin}" --version
  # fi

  # PYTHON CHECK

  local pin=""
  if [[ -f ".python-version" ]]; then
    pin="$(sed -n '1p' .python-version || true)"
  else
    pin="$(__uv__read_global_pin 2>/dev/null || true)"
  fi

  local py_path=""
  if [[ -n "$pin" ]]; then
    py_path="$(
      UV_PYTHON_DOWNLOADS=never uv --no-progress python find --managed-python "$pin" 2>/dev/null \
        || UV_PYTHON_DOWNLOADS=never uv --no-progress python find "$pin" 2>/dev/null \
        || true
    )"
  fi

  if [[ -n "$py_path" ]]; then
    exec uv run --python "$py_path" python -V
  fi


  echo "Python (none)"
)




# -------------------------
# venv (function): create/activate venv
# -------------------------
venv() {
  # No `set ...` in this top-level wrapper.

  __venv__strict() (
    __uv__strict
    "$@"
  )

  # Ensure required python (prompt-based) once.
  if ! __venv__strict _uv_install_python true; then
    return 1
  fi
  
  __venv__abs_path() (
    __uv__strict

    local p="${1:?missing path}"
    if [[ "$p" = /* ]]; then
      printf '%s\n' "$p"
    else
      printf '%s/%s\n' "$(pwd -P)" "$p"
    fi
  )

  __venv__create_if_missing() (
    __uv__strict

    local act="${1:?missing act}"
    local activate_path="${2:?missing activate_path}"
    local py="${3:-}"
    local explicit_name_passed="${4:?missing explicit_name_passed}"
    shift 4

    [[ -f "$activate_path" ]] && return 0

    local -a cmd=(uv --no-progress venv)
    if [[ -n "$py" ]]; then
      cmd+=(--python "$py")
    fi

    # Compute absolute path deterministically for display.
    local act_abs=""
    if [[ "$explicit_name_passed" == "1" ]]; then
      act_abs="$(__venv__abs_path "$act")"
    else
      act_abs="$(__venv__abs_path ".venv")"
    fi

    local uv_out=""
    if [[ "$explicit_name_passed" == "1" ]]; then
      uv_out="$("${cmd[@]}" "$@" "$act" 2>&1 | sed '/^Activate with:/d')"
    else
      uv_out="$(
        env UV_PYTHON_DOWNLOADS=never "${cmd[@]}" "$@" 2>&1 \
          | sed '/^Activate with:/d'
      )"

    fi

    if [[ -n "$uv_out" ]]; then
      printf '%s\n' "$uv_out" | sed -E \
        "s|^Creating virtual environment at: .*|Creating virtual environment at: ${act_abs}|"
    else
      printf 'Creating virtual environment at: %s\n' "$act_abs"
    fi
    # Must exist after creation, otherwise treat as failure.
    [[ -f "$activate_path" ]] || return 1
  )

  __venv__ensure_cfg_prompt() (
    __uv__strict

    local cfg="${1:?missing cfg}"
    local prompt_name="${2:?missing prompt_name}"

    [[ -f "$cfg" ]] || return 0

    local tmp="${cfg}.tmp.$$"
    awk -v prompt_name="$prompt_name" '
      BEGIN { found = 0 }
      /^[[:space:]]*prompt[[:space:]]*=/ {
        if (!found) {
          print "prompt = " prompt_name
          found = 1
        }
        next
      }
      { print }
      END {
        if (!found) {
          print "prompt = " prompt_name
        }
      }
    ' "$cfg" >"$tmp"
    mv -f "$tmp" "$cfg"
  )

  __venv__remove_cfg_prompt() (
    __uv__strict

    local cfg="${1:?missing cfg}"
    [[ -f "$cfg" ]] || return 0

    if ! grep -qE '^[[:space:]]*prompt[[:space:]]*=' "$cfg" 2>/dev/null; then
      return 0
    fi

    local tmp="${cfg}.tmp.$$"
    awk '
      /^[[:space:]]*prompt[[:space:]]*=/ { next }
      { print }
    ' "$cfg" >"$tmp"
    mv -f "$tmp" "$cfg"
  )

  __venv__assert_activate_exists() (
    __uv__strict
    [[ -f "${1:?missing activate_path}" ]]
  )

  __venv__is_existing_venv_dir() (
    __uv__strict

    local act="${1:?missing act}"
    [[ -d "$act" ]] || return 1
    [[ -f "$act/pyvenv.cfg" ]] || return 1
    return 0
  )

  __venv__confirm_replace() (
    __uv__strict
  
    local act="${1:?missing act}"
    local shown=""
  
    if [[ "$act" = /* ]]; then
      shown="$act"
    else
      shown="$(pwd -P)/${act#./}"
    fi
  
    local ans=""
    while true; do
      printf 'A virtual environment already exists at `%s`. Do you want to replace it? [y/n] ' \
        "$shown" 1>&2
  
      IFS= read -r ans </dev/tty || ans=""
      ans="$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')"
  
      case "$ans" in
        y|yes)
          return 0
          ;;
        n|no)
          return 1
          ;;
        *)
          ;;
      esac
    done
  )


  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    printf '[venv] Already inside a virtual environment: %s\n' \
      "${VIRTUAL_ENV}" 1>&2
    return 0
  fi

  local py=""
  local act=".venv"
  local add_cfg_prompt="1"          # no name => prompt=<cwd basename>
  local explicit_name_passed="0"    # whether user passed a positional path

  while [[ $# -gt 0 ]]; do
    case "${1-}" in
      -p|--python)
        shift || true
        if [[ $# -le 0 ]]; then
          printf '[ERROR] missing value for --python\n' 1>&2
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
        return 1
        ;;
      *)
        act="${1-}"
        add_cfg_prompt="0"       # explicit path => NO prompt=... in pyvenv.cfg
        explicit_name_passed="1"
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
  local cfg_path="${act}/pyvenv.cfg"
  local prompt_name=""
  prompt_name="$(basename "$(pwd -P)")"

  # Behavior difference:
  # - venv (no name): if .venv exists, just activate (no creation).
  # - venv <name>: if it exists, ask to replace; if "yes", pass --clear to uv.
  local -a clear_flag=()
  if [[ "$explicit_name_passed" == "1" ]]; then
    if __venv__strict __venv__is_existing_venv_dir "$act"; then
      if ! __venv__strict __venv__confirm_replace "$act"; then
        return 1
      fi
      clear_flag=(--clear)
    fi
  fi

  if [[ "$explicit_name_passed" == "0" ]]; then
    if [[ -f "$activate_path" ]]; then
      __venv__strict __venv__ensure_cfg_prompt "$cfg_path" "$prompt_name" || true
      export VIRTUAL_ENV_PROMPT="(${prompt_name}) "
      # shellcheck disable=SC1090
      source "${activate_path}"
      return 0
    fi
  fi

  if ! __venv__strict __venv__create_if_missing \
    "$act" "$activate_path" "$py" "$explicit_name_passed" \
    "${clear_flag[@]}" "${extra_args[@]}"; then
    return 1
  fi

  if ! __venv__strict __venv__assert_activate_exists "$activate_path"; then
    printf '[ERROR] expected %s not found\n' "${activate_path}" 1>&2
    return 1
  fi

  if [[ "$add_cfg_prompt" == "1" ]]; then
    __venv__strict __venv__ensure_cfg_prompt "$cfg_path" "$prompt_name" || true
  else
    __venv__strict __venv__remove_cfg_prompt "$cfg_path" || true
  fi

  # Activate in the current shell
  # shellcheck disable=SC1090
  if [[ "$add_cfg_prompt" == "1" ]]; then
    export VIRTUAL_ENV_PROMPT="(${prompt_name}) "
  else
    export VIRTUAL_ENV_PROMPT="($(basename -- "$act")) "
  fi
  source "${activate_path}"
}














# -------------------------
# lpin (local pin)
# -------------------------
lpin() (
  __uv__strict

  find_lpin_root() (
    __uv__strict

    local dir="$PWD"
    while :; do
      if [[ -f "$dir/.python-version" ]]; then
        printf '%s\n' "$dir"
        return 0
      fi
      local parent="${dir%/*}"
      [[ "$dir" == "$parent" ]] && return 1
      dir="$parent"
    done
  )

  _display_path() (
    __uv__strict

    local p="${1:?missing path}"
    if [[ "$p" == "$HOME" ]]; then
      printf '~\n'
      return 0
    fi
    printf '%s\n' "${p/#$HOME\//~}"
  )

  _clear_nearest_pin() (
    __uv__strict

    local root=""
    if root="$(find_lpin_root)"; then
      rm -f -- "$root/.python-version"
      return 0
    fi
    return 0
  )

  if [[ $# -gt 0 ]]; then
    local arg="$1"
    case "${arg,,}" in
      none|off|clear|unset|remove|rm)
        _clear_nearest_pin
        ;;
      *)
        uv python pin "$arg" >/dev/null
        ;;
    esac
  fi

  # In a pinned dir → print just "<version>"
  if [[ -f ".python-version" ]]; then
    head -n 1 .python-version
    exit 0
  fi

  # Under a pinned ancestor → print "(~path) <version>"
  local root=""
  if root="$(find_lpin_root)"; then
    local version="" display=""
    version="$(head -n 1 "$root/.python-version")"
    display="$(_display_path "$root")"
    printf '(%s) %s\n' "$display" "$version"
  else
    echo "(none)"
  fi
)


# -------------------------
# gpin (global pin)
# -------------------------
gpin() (
  __uv__strict
  : "${XDG_CONFIG_HOME:=$HOME/.config}"

  local arg="${1-}"
  local -a pin_files=(
    "$XDG_CONFIG_HOME/uv/.python-version"
    "$HOME/.uv/.python-version"
    "$HOME/.config/uv/python/version"
  )

  _print() (
    __uv__strict
    local f=""
    for f in "${pin_files[@]}"; do
      [[ -f "$f" ]] || continue
      sed -n '1p' "$f"
      return 0
    done
    echo "(none)"
  )

  _rm_all() (
    __uv__strict
    rm -f -- "${pin_files[@]}"
  )

  # Set / clear pin
  if [[ -n "$arg" ]]; then
    if [[ "$arg" == "none" ]]; then
      _rm_all
      _print
      exit 0
    fi

    # Allow interpreter path
    if [[ "$arg" == /* || "$arg" == ./* || "$arg" == ../* || -e "$arg" ]]; then
      local p=""
      if command -v realpath >/dev/null 2>&1; then
        p="$(realpath -m -- "$arg" 2>/dev/null || true)"
      fi
      [[ -n "$p" ]] || p="$arg"
      [[ -f "$p" && -x "$p" ]] || {
        echo "[gpin] ERROR: not an executable interpreter path: $arg" >&2
        exit 2
      }
      arg="$p"
    fi

    uv python pin "$arg" --global >/dev/null || {
      echo "[gpin] ERROR: uv could not pin: $arg" >&2
      exit 1
    }
  fi

  # Validate existing pin
  local val
  val="$(_print)"

  if [[ "$val" != "(none)" ]]; then
    if ! uv python find "$val" >/dev/null 2>&1; then
      # Broken global pin → remove it
      _rm_all
      echo "(none)"
      exit 0
    fi
  fi

  printf '%s\n' "$val"
)



# -------------------------
# interpreters
# -------------------------
interpreters() (
  __uv__strict
  _uv_install_python true || exit 1

  local timeout_bin=""
  timeout_bin="$(command -v timeout 2>/dev/null || printf '')"

  _maybe_tmo() (
    __uv__strict
    local seconds="${1:?missing timeout}"
    shift
    if [[ -n "$timeout_bin" ]]; then
      "$timeout_bin" "$seconds" "$@" 2>/dev/null \
        || "$timeout_bin" "${seconds%s}" "$@" 2>/dev/null \
        || true
    else
      "$@" 2>/dev/null
    fi
  )

  tmo1() (
    __uv__strict
    _maybe_tmo "1s" "$@"
  )

  tmo2() (
    __uv__strict
    _maybe_tmo "2s" "$@"
  )

  _resolve() (
    __uv__strict

    readlink -f "$1" 2>/dev/null \
      || realpath -e "$1" 2>/dev/null \
      || printf '%s' "$1"
  )

  _should_skip_path() (
    __uv__strict

    local p="${1:?missing path}"
    if [[ "${UV_SHIMS_SCAN_MNT:-0}" = 0 && "$p" == /mnt/* ]]; then
      return 0
    fi
    return 1
  )

  _probe_bin() (
    __uv__strict

    local bin="${1:?missing bin}"
    [[ -x "$bin" ]] || return 1
    if _should_skip_path "$bin"; then
      return 1
    fi

    local real_py="" ver="" real_norm=""
    real_py="$(tmo1 "$bin" -c 'import sys;print(sys.executable)' || true)"
    [[ -n "$real_py" ]] || return 1

    ver="$(tmo1 "$bin" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")' || true)"
    [[ -n "$ver" ]] || return 1

    real_norm="$(_resolve "$real_py")"
    [[ -n "$real_norm" ]] || return 1

    # stdout: "<version>\t<resolved_executable_path>"
    ver="${ver//$'\r'/}"
    real_norm="${real_norm//$'\r'/}"
    printf '%s\t%s\n' "$ver" "$real_norm"
  )

  # --- current interpreter (respect pins; avoid downloads)
  local CUR="" CUR_PATH=""
  CUR_PATH="$(
    UV_PYTHON_DOWNLOADS=never uv --no-progress run python - <<'PY' 2>/dev/null || true
import sys
print(sys.executable)
PY
  )"
  if [[ -n "$CUR_PATH" ]]; then
    CUR="$(_resolve "$CUR_PATH")"
  fi

  # --- pins
  local LPIN="" GPIN="" PIN=""
  if [[ -f ".python-version" ]]; then
    LPIN="$(sed -n '1p' .python-version 2>/dev/null || printf '')"
  fi
  GPIN="$(__uv__read_global_pin 2>/dev/null || printf '')"
  PIN="${LPIN:-$GPIN}"

  local PIN_FULL="" PIN_PREFIX=""
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

  # --- state lives ONLY here (parent subshell)
  declare -a SHORTS=()
  declare -a PATHS=()
  declare -A SEEN=()

  _ingest_record() (
    __uv__strict

    local ver="${1:?missing ver}"
    local real_norm="${2:?missing path}"
    printf '%s\t%s\n' "$ver" "$real_norm"
  )

  # Helper: add if unseen (done in parent to preserve state)
  _add_if_new() {
    local ver="$1"
    local real_norm="$2"
    [[ -n "$ver" && -n "$real_norm" ]] || return 0
    [[ -z "${SEEN[$real_norm]:-}" ]] || return 0
    SEEN["$real_norm"]=1
    SHORTS+=("cpython-$ver")
    PATHS+=("$real_norm")
  }

  # --- try uv list
  local LIST_OK=0 OUT=""
  OUT="$(tmo2 uv --no-progress python list --only-installed || true)"
  if [[ -n "$OUT" ]]; then
    while IFS= builtin read -r line; do
      [[ -n "$line" ]] || continue

      local a="" b="" p=""
      a="${line%%[[:space:]]*}"
      b="${line#"$a"}"
      b="${b##[[:space:]]}"
      p="${b%% *->*}"
      p="${p#"${p%%[![:space:]]*}"}"
      [[ -n "$p" ]] || continue

      if [[ "$p" != /* ]]; then
        p="$(realpath -m -- "$p" 2>/dev/null || printf '%s' "$p")"
      fi

      local rec=""
      rec="$(_probe_bin "$p" 2>/dev/null || true)"
      if [[ -n "$rec" ]]; then
        local ver="" real_norm=""
        IFS=$'\t' builtin read -r ver real_norm <<<"$rec"
        _add_if_new "$ver" "$real_norm"
      fi
    done <<<"$OUT" || true
    LIST_OK=1
  fi

  # --- fallback probing
  if [[ "$LIST_OK" -eq 0 ]]; then
    if [[ "${UV_SHIMS_FAST:-0}" = "1" ]]; then
      local cand="" bin="" rec="" ver="" real_norm=""
      for cand in python python3 python3.12 python3.11 python3.10; do
        bin="$(command -v "$cand" 2>/dev/null || printf '')"
        [[ -n "$bin" ]] || continue
        rec="$(_probe_bin "$bin" 2>/dev/null || true)"
        [[ -n "$rec" ]] || continue
        IFS=$'\t' builtin read -r ver real_norm <<<"$rec"
        _add_if_new "$ver" "$real_norm"
      done
    else
      local cand="" bin="" rec="" ver="" real_norm=""
      while IFS= builtin read -r cand; do
        bin="$(command -v "$cand" 2>/dev/null || printf '')"
        [[ -n "$bin" ]] || continue
        rec="$(_probe_bin "$bin" 2>/dev/null || true)"
        [[ -n "$rec" ]] || continue
        IFS=$'\t' builtin read -r ver real_norm <<<"$rec"
        _add_if_new "$ver" "$real_norm"
      done < <(
        LC_ALL=C compgen -c \
          | grep -E '^python(3(\.[0-9]+)?)?$|^python3\.[0-9]+$' \
          | sort -u
      ) || true

      if [[ "${UV_SHIMS_SKIP_UVDATA:-0}" != 1 ]]; then
        local _uv_data="" p="" rec="" ver="" real_norm=""
        _uv_data="${XDG_DATA_HOME:-$HOME/.local/share}/uv/python"
        if [[ -d "$_uv_data" ]]; then
          while IFS= builtin read -r p; do
            rec="$(_probe_bin "$p" 2>/dev/null || true)"
            [[ -n "$rec" ]] || continue
            IFS=$'\t' builtin read -r ver real_norm <<<"$rec"
            _add_if_new "$ver" "$real_norm"
          done < <(
            find "$_uv_data" -type f -path '*/bin/python*' -executable 2>/dev/null \
              | sort -u
          ) || true
        fi
      fi
    fi
  fi

  if ((${#SHORTS[@]} == 0)); then
    echo "(no interpreters found)"
    exit 0
  fi

  # --- sort by version
  local -a KEYS=()
  mapfile -t KEYS < <(printf '%s\n' "${!SHORTS[@]}")
  IFS=$'\n' KEYS=($(
    for i in "${KEYS[@]}"; do
      printf '%s\t%s\n' "${SHORTS[i]}" "$i"
    done | sort -V | awk -F'\t' '{print $2}'
  ))
  unset IFS

  # --- star selection
  local STAR_IDX=-1 i=""
  if [[ -n "$CUR" ]]; then
    for i in "${KEYS[@]}"; do
      if [[ "$CUR" == "${PATHS[i]}" ]]; then
        STAR_IDX="$i"
        break
      fi
    done
  fi
  if [[ "$STAR_IDX" -lt 0 && -n "$PIN_FULL" ]]; then
    for i in "${KEYS[@]}"; do
      [[ "${SHORTS[i]}" == "$PIN_FULL" ]] && STAR_IDX="$i"
    done
  fi
  if [[ "$STAR_IDX" -lt 0 && -n "$PIN_PREFIX" ]]; then
    for i in "${KEYS[@]}"; do
      [[ "${SHORTS[i]}" == ${PIN_PREFIX}* ]] && STAR_IDX="$i"
    done
  fi

  for i in "${KEYS[@]}"; do
    local mark=' '
    [[ "$i" == "$STAR_IDX" ]] && mark='*'
    printf "%s %s %s\n" "$mark" "${SHORTS[i]}" "${PATHS[i]}"
  done
)





uncache() (
  # __uv__strict
  _uv_install_python true || return 1

  : "${UV_CACHE_DIR:=$HOME/.cache/uv}"
  local DEBUG="${UV_SHIMS_DEBUG:-0}"

  log() (
    __uv__strict
    if (( DEBUG )); then
      printf '[uv-cache-nlink-gc] %s\n' "$*" >&2
    fi
    return 0
  )

  has_linked_files() (
    __uv__strict
    local dir="${1:?missing dir}"

    local hit=""
    hit="$(find "$dir" -xdev -type f -links +1 -print -quit 2>/dev/null || true)"

    [[ -n "$hit" ]]
  )

  _dir_size_bytes() (
    __uv__strict
    local d="${1:?missing dir}"
    du -sb -- "$d" 2>/dev/null | awk '{print $1}'
  )

  _fmt_gb() (
    __uv__strict
    local b="${1:?missing bytes}"
    awk -v b="$b" 'BEGIN { printf "%.2fGB", b / (1024*1024*1024) }'
  )

  _norm_proj() (
    __uv__strict
    local name="${1:?missing name}"
    name="${name,,}"
    name="${name//_/-}"
    name="${name//./-}"
    printf '%s' "$name"
  )

  if [[ ! -d "$UV_CACHE_DIR" ]]; then
    echo "UV cache directory does not exist: $UV_CACHE_DIR" >&2
    exit 1
  fi

  local before_bytes
  before_bytes="$(_dir_size_bytes "$UV_CACHE_DIR")"

  log "UV_CACHE_DIR=$UV_CACHE_DIR"

  # 1) archive/: borrar objetos sin hardlinks activos
  local archive_root
  while IFS= read -r -d '' archive_root; do
    [[ -d "$archive_root" ]] || continue

    local obj_dir
    while IFS= read -r -d '' obj_dir; do
      [[ -d "$obj_dir" ]] || continue

      if has_linked_files "$obj_dir"; then
        log "KEEP archive object: $obj_dir"
      else
        log "DELETE archive object: $obj_dir"
        rm -rf -- "$obj_dir"
      fi
    done < <(
      find "$archive_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null
    )

    # borrar root vacío
    [[ -z "$(ls -A "$archive_root" 2>/dev/null)" ]] && rmdir "$archive_root" 2>/dev/null || true
  done < <(
    find "$UV_CACHE_DIR" -type d -name '*archive*' -print0 2>/dev/null
  )

  # 2) wheels/pypi/: borrar proyectos sin hardlinks activos
  local wheels_root
  while IFS= read -r -d '' wheels_root; do
    local pypi_root="$wheels_root/pypi"
    [[ -d "$pypi_root" ]] || continue

    local proj_dir
    while IFS= read -r -d '' proj_dir; do
      [[ -d "$proj_dir" ]] || continue

      if has_linked_files "$proj_dir"; then
        log "KEEP wheels project: $proj_dir"
      else
        log "DELETE wheels project: $proj_dir"
        rm -rf -- "$proj_dir"
      fi
    done < <(
      find "$pypi_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null
    )
  done < <(
    find "$UV_CACHE_DIR" -type d -name '*wheels*' -print0 2>/dev/null
  )

  # 2.5) simple-v*/pypi/*.rkyv: borrar rkyv de proyectos que no están en uso
  local used_tmp
  used_tmp="$(mktemp -p "${TMPDIR:-/tmp}" uv-used-projs.XXXXXX)"
  trap 'rm -f -- "$used_tmp"' RETURN 2>/dev/null || true

  # Construye un set de proyectos "en uso" a partir de wheels/pypi con hardlinks.
  while IFS= read -r -d '' wheels_root; do
    local pypi_root="$wheels_root/pypi"
    [[ -d "$pypi_root" ]] || continue

    local proj_dir
    while IFS= read -r -d '' proj_dir; do
      [[ -d "$proj_dir" ]] || continue

      if has_linked_files "$proj_dir"; then
        _norm_proj "$(basename -- "$proj_dir")" >>"$used_tmp"
      fi
    done < <(
      find "$pypi_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null
    )
  done < <(
    find "$UV_CACHE_DIR" -type d -name '*wheels*' -print0 2>/dev/null
  )

  sort -u -o "$used_tmp" "$used_tmp" 2>/dev/null || true

  local simple_root
  while IFS= read -r -d '' simple_root; do
    local pypi_simple="$simple_root/pypi"
    [[ -d "$pypi_simple" ]] || continue

    local rkyv
    while IFS= read -r -d '' rkyv; do
      [[ -f "$rkyv" ]] || continue

      local base proj
      base="$(basename -- "$rkyv")"
      proj="$(_norm_proj "${base%.rkyv}")"

      if grep -Fxq -- "$proj" "$used_tmp" 2>/dev/null; then
        log "KEEP simple rkyv: $rkyv"
      else
        log "DELETE simple rkyv: $rkyv"
        rm -f -- "$rkyv"
      fi
    done < <(
      find "$pypi_simple" -maxdepth 1 -type f -name '*.rkyv' -print0 2>/dev/null
    )

    [[ -z "$(ls -A "$pypi_simple" 2>/dev/null)" ]] && rmdir "$pypi_simple" 2>/dev/null || true
    [[ -z "$(ls -A "$simple_root" 2>/dev/null)" ]] && rmdir "$simple_root" 2>/dev/null || true
  done < <(
    find "$UV_CACHE_DIR" -type d -name 'simple-v*' -print0 2>/dev/null
  )

  # 3) stats finales
  local after_bytes
  after_bytes="$(_dir_size_bytes "$UV_CACHE_DIR")"

  local released=$(( before_bytes - after_bytes ))
  (( released < 0 )) && released=0

  printf 'uv cache released storage: %s\n' "$(_fmt_gb "$released")"
  printf 'uv cache remaining storage: %s\n' "$(_fmt_gb "$after_bytes")"
)









# -------------------------
# environments:
# -------------------------
environments() (
  _uv_install_python true || return 1
  : "${UV_ENVS_WORKERS:=16}"
  : "${UV_ENVS_ROOT:=$HOME}"

  # command python3 - "$UV_ENVS_ROOT" <<'PY'
  UV_PYTHON_DOWNLOADS=never uv --no-progress run python - "$UV_ENVS_ROOT" <<'PY'
from __future__ import annotations

import concurrent.futures
import os
import sys
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class EnvRow:
    """A discovered environment row."""
    path: str
    is_uv: bool


def _is_environment_dir(d: str) -> bool:
    """Return True if d looks like a Python venv directory.

    Signature:
      - pyvenv.cfg exists
      - bin/ exists
      - bin/activate exists
      - bin/python exists (file or symlink)
    """
    cfg: str = os.path.join(d, "pyvenv.cfg")
    bin_dir: str = os.path.join(d, "bin")
    act: str = os.path.join(bin_dir, "activate")
    py: str = os.path.join(bin_dir, "python")

    if not os.path.isfile(cfg):
        return False
    if not os.path.isdir(bin_dir):
        return False
    if not os.path.isfile(act):
        return False
    if not (os.path.isfile(py) or os.path.islink(py)):
        return False
    return True


def _iter_env_dirs(root_dir: str) -> Iterator[str]:
    """Find environment dirs under root with limited recursion inside dot-dirs.

    Rule:
      - For directories whose name starts with '.', only recurse ONE additional level.
    """
    queue: list[tuple[str, int]] = [(root_dir, -1)]

    while queue:
        cur, dot_budget = queue.pop()

        if _is_environment_dir(cur):
            yield cur

        try:
            with os.scandir(cur) as it:
                for ent in it:
                    if not ent.is_dir(follow_symlinks=False):
                        continue

                    is_dot: bool = ent.name.startswith(".")
                    if dot_budget == 0:
                        continue

                    if dot_budget == -1:
                        next_budget: int = 1 if is_dot else -1
                        queue.append((ent.path, next_budget))
                    else:
                        queue.append((ent.path, 0))
        except OSError:
            continue


def _abs_path(p: str) -> str:
    """Return an absolute path (best-effort, resolving symlinks)."""
    try:
        return os.path.realpath(p)
    except OSError:
        return os.path.abspath(p)


def _is_uv_by_python_symlink(env_dir: str) -> bool:
    """Detect uv-managed env by checking python* symlinks in env_dir/bin.

    Heuristic:
      - For any file in <env_dir>/bin with 'python' substring in its name:
          if it is a symlink AND its resolved target path contains 'uv',
          then env is considered uv-managed.

    Args:
        env_dir: Environment directory path.

    Returns:
        True if heuristic matches, otherwise False.
    """
    bin_dir: str = os.path.join(env_dir, "bin")
    try:
        with os.scandir(bin_dir) as it:
            for ent in it:
                name: str = ent.name
                if "python" not in name:
                    continue

                # Use path-based check because DirEntry.is_symlink() may follow
                # filesystem quirks; os.path.islink is definitive for the path.
                candidate: str = os.path.join(bin_dir, name)
                if not os.path.islink(candidate):
                    continue

                try:
                    target: str = os.path.realpath(candidate)
                except OSError:
                    continue

                if "uv" in target:
                    return True
    except OSError:
        return False

    return False


def _tag_env(env_dir: str) -> EnvRow:
    """Create an EnvRow for env_dir."""
    is_uv: bool = _is_uv_by_python_symlink(env_dir=env_dir)
    return EnvRow(path=_abs_path(env_dir), is_uv=is_uv)


def main(argv: list[str]) -> int:
    """Entry point: print absolute env paths, with [uv] tag when applicable."""
    root_dir: str = argv[1] if len(argv) > 1 else os.path.expanduser("~")

    envs: list[str] = list(_iter_env_dirs(root_dir=root_dir))
    if not envs:
        return 0

    max_workers_env: str = os.environ.get("UV_ENVS_WORKERS", "").strip()
    max_workers: int = int(max_workers_env) if max_workers_env else 16
    workers: int = max(1, min(max_workers, (os.cpu_count() or 4) * 2))

    seen: set[str] = set()
    rows: list[EnvRow] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_tag_env, d) for d in envs]
        for fut in concurrent.futures.as_completed(futs):
            try:
                row: EnvRow = fut.result()
            except Exception:
                continue

            if row.path in seen:
                continue
            seen.add(row.path)
            rows.append(row)

    rows.sort(key=lambda r: r.path)
    for r in rows:
        suffix: str = " [uv]" if r.is_uv else ""
        sys.stdout.write(f"{r.path}{suffix}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
PY
)





# -------------------------
# lock: add deps to pyproject from env or src imports, then uv lock
# Flags:
#   --src / -s   : derive deps from source imports (instead of env dist-info)
#   --txt / -t   : also write requirements.txt with pinned versions
# Default:
#   env mode (no flag) + uv lock
# Behavior:
#   Re-running lock replaces declared deps with exactly what's detected
#   (for the selected mode), not additive accumulation.
# -------------------------
lock() (
  # __uv__strict

  _uv_install_python true || exit 1

  die() {
    # Controlled failure: do not trigger ERR handler, and stop the subshell now.
    trap - ERR
    printf 'error: %s\n' "$*" >&2
    exit 1
  }

  parse_args() (
    __uv__strict

    local deps_mode="env"
    local write_txt="false"
    local arg="" rest="" i=""

    for arg in "$@"; do
      case "$arg" in
        --src) deps_mode="src" ;;
        --txt) write_txt="true" ;;
        -[!-]*)
          rest="${arg#-}"
          for ((i = 0; i < ${#rest}; i++)); do
            case "${rest:$i:1}" in
              s) deps_mode="src" ;;
              t) write_txt="true" ;;
              *) printf 'die:%s\n' "unknown flag: -${rest:$i:1}" ; return 0 ;;
            esac
          done
          ;;
        *)
          printf 'die:%s\n' "unknown argument: $arg"
          return 0
          ;;
      esac
    done

    printf 'deps_mode=%q\n' "$deps_mode"
    printf 'write_txt=%q\n' "$write_txt"
  )

  local parsed=""
  parsed="$(parse_args "$@")"
  if [[ "$parsed" == die:* ]]; then
    die "${parsed#die:}"
  fi
  eval "$parsed"

  local venv_dir="${VIRTUAL_ENV-}"
  [[ -n "$venv_dir" ]] || die "no active environment (VIRTUAL_ENV is not set)"
  command -v uv >/dev/null 2>&1 || die "uv not found in PATH"
  # command -v python >/dev/null 2>&1 || die "python not found in PATH"

  normalize_name() (
    __uv__strict

    local raw_name="${1:-}"
    raw_name="${raw_name,,}"
    raw_name="$(printf '%s' "$raw_name" | command sed -E 's/[-_.]+/-/g')"
    printf '%s\n' "$raw_name"
  )

  is_nvidia_dependency() (
    __uv__strict

    local name="${1:-}"
    [[ "$name" == *nvidia* || "$name" == *cuda* ]]
  )

  find_site_packages_dirs() (
    __uv__strict

    local venv_dir_in="${1:?missing venv_dir}"
    command find "$venv_dir_in" -type d \
      \( -name "site-packages" -o -name "dist-packages" \) \
      -print 2>/dev/null \
      | LC_ALL=C command sort -u
  )

  name_from_dist_info() (
    __uv__strict

    local dist_dir="${1:?missing dist_dir}"
    local meta_path="$dist_dir/METADATA"

    if [[ -f "$meta_path" ]]; then
      local raw_name=""
      raw_name="$(
        command awk -F': *' 'tolower($1)=="name" { print $2; exit }' \
          "$meta_path" 2>/dev/null || true
      )"
      raw_name="$(printf '%s' "$raw_name" | command sed -E 's/[[:space:]]+$//')"
      if [[ -n "$raw_name" ]]; then
        normalize_name "$raw_name"
        return 0
      fi
    fi
    return 1
  )

  fallback_name_from_metadata_dir_basename() (
    __uv__strict

    local base="${1:?missing base}"
    base="${base%.dist-info}"
    base="${base%.egg-info}"

    local prefix="${base%*-}"
    local suffix="${base##*-}"
    if [[ "$base" != "$prefix" && "$suffix" =~ ^[0-9] ]]; then
      base="$prefix"
    fi

    normalize_name "$base"
  )

  gather_deps_from_env() (
    __uv__strict

    local sp="" d=""
    for sp in "${SITE_DIRS[@]}"; do
      while IFS= read -r d; do
        name_from_dist_info "$d" \
          || fallback_name_from_metadata_dir_basename "$(basename "$d")"
      done < <(command find "$sp" -maxdepth 1 -type d -name "*.dist-info" -print 2>/dev/null)
    done
  )

  gather_deps_from_src_py() (
    __uv__strict

    UV_PYTHON_DOWNLOADS=never uv --no-progress run python - << 'PY'
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Iterable, Set
from importlib import metadata as md

EXCLUDE_DIRS: Set[str] = {
    ".venv",
    ".git",
    ".dvc",
    "__pycache__",
}

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if any(d in parts for d in EXCLUDE_DIRS):
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p

def top_level_from_imports(root: Path) -> Set[str]:
    mods: Set[str] = set()
    for file in iter_py_files(root):
        try:
            src = file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    name = (n.name or "").split(".", 1)[0].strip()
                    if name:
                        mods.add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                mod = (node.module or "").split(".", 1)[0].strip()
                if mod:
                    mods.add(mod)
    return mods

def normalize(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[-_.]+", "-", name)
    return name

def main() -> int:
    root = Path(os.getcwd())
    mods = top_level_from_imports(root)
    if not mods:
        return 0

    pkg_to_dists = md.packages_distributions()
    dists: Set[str] = set()

    for m in mods:
        for dist in pkg_to_dists.get(m, []):
            if dist:
                dists.add(normalize(dist))

    for d in sorted(dists):
        print(d)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
PY
  )

  version_from_dist_py() (
    __uv__strict

    local dist_name_norm="${1:?missing dist_name_norm}"
    UV_PYTHON_DOWNLOADS=never uv --no-progress run python - << PY
from __future__ import annotations

import re
from importlib import metadata as md

def normalize(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[-_.]+", "-", name)
    return name

target = "${dist_name_norm}"
for d in md.distributions():
    name = d.metadata.get("Name", "")
    if name and normalize(name) == target:
        print(d.version)
        raise SystemExit(0)

raise SystemExit(1)
PY
  )

  read_current_declared_deps() (
    __uv__strict

    UV_PYTHON_DOWNLOADS=never uv --no-progress run python - << 'PY'
from __future__ import annotations

import re
from pathlib import Path
import tomllib
from typing import Any, List

def normalize(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[-_.]+", "-", name)
    return name

def norm_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    for x in xs:
        if isinstance(x, str) and x.strip():
            out.append(normalize(x))
    return sorted(set(out))

p = Path("pyproject.toml")
if not p.exists():
    print("MAIN:")
    print("CUDA:")
    raise SystemExit(0)

data = tomllib.loads(p.read_text(encoding="utf-8"))
proj = data.get("project", {})
main = norm_list(proj.get("dependencies", []))
opt = proj.get("optional-dependencies", {})
cuda = norm_list(opt.get("cuda", []) if isinstance(opt, dict) else [])

print("MAIN:")
for d in main:
    print(d)
print("CUDA:")
for d in cuda:
    print(d)
PY
  )

  local project_root=""
  project_root="$(pwd)"

  if [[ ! -f "$project_root/pyproject.toml" ]]; then
    echo "No pyproject.toml found in: $project_root"
    echo "Initializing with: uv init"
    (
      set -Eeuo pipefail
      shopt -s inherit_errexit 2>/dev/null || true
      trap 'echo "ERROR en ${FUNCNAME[0]:-MAIN} línea $LINENO: $BASH_COMMAND" >&2' ERR
      cd "$project_root"
      uv init --bare --vcs none && python --version >> .python-version >/dev/null
    )
  fi

  cd "$project_root"

  local tmp_all="" tmp_names="" tmp_main="" tmp_cuda=""
  local tmp_cur_main="" tmp_cur_cuda="" tmp_remove_main="" tmp_remove_cuda=""
  local tmp_add_main="" tmp_add_cuda="" tmp_cur_dump=""

  trap '
    rm -f \
      "${tmp_all-}" "${tmp_names-}" "${tmp_main-}" "${tmp_cuda-}" \
      "${tmp_cur_main-}" "${tmp_cur_cuda-}" "${tmp_remove_main-}" "${tmp_remove_cuda-}" \
      "${tmp_add_main-}" "${tmp_add_cuda-}" "${tmp_cur_dump-}" \
      2>/dev/null || true
  ' EXIT

  tmp_all="$(mktemp)"
  tmp_names="$(mktemp)"
  tmp_main="$(mktemp)"
  tmp_cuda="$(mktemp)"
  tmp_cur_main="$(mktemp)"
  tmp_cur_cuda="$(mktemp)"
  tmp_remove_main="$(mktemp)"
  tmp_remove_cuda="$(mktemp)"
  tmp_add_main="$(mktemp)"
  tmp_add_cuda="$(mktemp)"
  tmp_cur_dump="$(mktemp)"

  local -a SITE_DIRS=()
  mapfile -t SITE_DIRS < <(find_site_packages_dirs "$venv_dir")
  ((${#SITE_DIRS[@]} > 0)) || die "could not locate site-packages under: $venv_dir"

  if [[ "$deps_mode" == "src" ]]; then
    gather_deps_from_src_py >"$tmp_all"
  else
    gather_deps_from_env >"$tmp_all"
  fi

  LC_ALL=C command sort -u "$tmp_all" >"$tmp_names"

  : >"$tmp_main"
  : >"$tmp_cuda"

  local name=""
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    if is_nvidia_dependency "$name"; then
      printf '%s\n' "$name" >>"$tmp_cuda"
    else
      printf '%s\n' "$name" >>"$tmp_main"
    fi
  done <"$tmp_names"

  LC_ALL=C command sort -u "$tmp_main" -o "$tmp_main" 2>/dev/null || true
  LC_ALL=C command sort -u "$tmp_cuda" -o "$tmp_cuda" 2>/dev/null || true

  # --- Sync pyproject to exactly detected deps (not additive) -----------------
  read_current_declared_deps >"$tmp_cur_dump"

  command awk 'f&&$0=="CUDA:"{exit} f{print} $0=="MAIN:"{f=1}' \
    "$tmp_cur_dump" | LC_ALL=C command sort -u >"$tmp_cur_main"
  command awk 'f{print} $0=="CUDA:"{f=1}' \
    "$tmp_cur_dump" | LC_ALL=C command sort -u >"$tmp_cur_cuda"

  # Remove deps currently declared but not desired.
  # comm requires both inputs sorted.
  command comm -23 "$tmp_cur_main" "$tmp_main" >"$tmp_remove_main" || true
  command comm -23 "$tmp_cur_cuda" "$tmp_cuda" >"$tmp_remove_cuda" || true

  if [[ -s "$tmp_remove_main" ]]; then
    command xargs -r -a "$tmp_remove_main" -n 50 uv remove --no-sync -- >/dev/null
  fi
  if [[ -s "$tmp_remove_cuda" ]]; then
    command xargs -r -a "$tmp_remove_cuda" -n 50 uv remove --optional cuda --no-sync -- >/dev/null
  fi

  # Add deps desired but not currently declared.
  command comm -13 "$tmp_cur_main" "$tmp_main" >"$tmp_add_main" || true
  command comm -13 "$tmp_cur_cuda" "$tmp_cuda" >"$tmp_add_cuda" || true

  if [[ -s "$tmp_add_main" ]]; then
    command xargs -r -a "$tmp_add_main" -n 50 uv add --raw --no-sync -- >/dev/null
  fi
  if [[ -s "$tmp_add_cuda" ]]; then
    command xargs -r -a "$tmp_add_cuda" -n 50 uv add --raw --optional cuda --no-sync -- >/dev/null
  fi

  if [[ "$write_txt" == "true" ]]; then
    : > requirements.txt
    local ver=""
    while IFS= read -r name; do
      [[ -n "$name" ]] || continue
      ver="$(version_from_dist_py "$name" 2>/dev/null || true)"
      [[ -n "$ver" ]] || die "could not determine version for $name"
      printf '%s==%s\n' "$name" "$ver" >> requirements.txt
    done < <(cat "$tmp_main" "$tmp_cuda" | LC_ALL=C command sort -u)
  fi

  uv lock
)
