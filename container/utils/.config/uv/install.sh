#!/usr/bin/env bash

# install.sh
# ----------------
# Auto-detect and strip CRLF if present (robust even when sourced)
SELF="${BASH_SOURCE[0]:-$0}"

if command grep -q $'\r' -- "$SELF"; then
  echo "[INFO] Converting DOS line endings to LF in $SELF..."
  command sed -i 's/\r$//' -- "$SELF"
fi


# # --- strict mode + useful traceback ---
# set -Eeuo pipefail
# shopt -s inherit_errexit 2>/dev/null || true
# set -o errtrace

# __traceback() {
#   local status="$?"
#   local i=0

#   echo "[ERROR] status : ${status}" >&2
#   echo "[ERROR] command: ${BASH_COMMAND}" >&2
#   echo "[ERROR] at     : ${BASH_SOURCE[1]}:${BASH_LINENO[0]} (func: ${FUNCNAME[1]:-MAIN})" >&2
#   echo "[ERROR] stack  :" >&2
#   for ((i=1; i<${#FUNCNAME[@]}; i++)); do
#     echo "  - ${BASH_SOURCE[$i]}:${BASH_LINENO[$((i-1))]}  ${FUNCNAME[$i]}" >&2
#   done
# }

# trap '__traceback' ERR

# # --- end traceback setup ---


# 1) Ensure local bin dir exists
UVBIN="$HOME/.local/bin"
mkdir -p "$UVBIN"


# 2) Install uv if missing
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] Installing uv..." >&2
  tmp_uv_installer="$(mktemp)"
  retries=5
  delay=2.5
  attempt=1
  while :; do
    curl_err_default="$(
      curl -fsSL -S --connect-timeout 10 --max-time 120 \
        -o "$tmp_uv_installer" \
        https://astral.sh/uv/install.sh 2>&1
    )" && break
    curl_err_v4="$(
      curl -4 -fsSL -S --connect-timeout 10 --max-time 120 \
        -o "$tmp_uv_installer" \
        https://astral.sh/uv/install.sh 2>&1
    )" && break
    echo "[WARN] Download failed (attempt ${attempt}/${retries})" >&2
    echo "[WARN]   default: ${curl_err_default}" >&2
    echo "[WARN]   ipv4   : ${curl_err_v4}" >&2

    if (( attempt >= retries )); then
      echo "[ERROR] Failed to download uv installer after ${retries} attempts" >&2
      rm -f "$tmp_uv_installer"
      exit 1
    fi
    attempt=$((attempt + 1))
    sleep "$delay"
  done
  if ! bash "$tmp_uv_installer" >/dev/null 2>&1; then
    echo "[ERROR] uv installer failed" >&2
    rm -f "$tmp_uv_installer"
    exit 1
  fi
  rm -f "$tmp_uv_installer"
fi


# 3) Ensure PATH lines in ~/.bashrc
BASHRC="$HOME/.bashrc"

LINE1='[[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && export PATH="$HOME/.local/bin:$PATH"  # uv bin'
LINE2='[ -f "$HOME/.config/uv/define-functions.sh" ] && chmod +x "$HOME/.config/uv/define-functions.sh" && ( set -e; "$HOME/.config/uv/define-functions.sh" ) && source "$HOME/.config/uv/define-functions.sh" >&2'

[ -f "$BASHRC" ] || : > "$BASHRC"

tmpfile="$(mktemp)"
awk -v l1="$LINE1" -v l2="$LINE2" '
BEGIN { f1=0; f2=0 }
{
  # Keep exact matches
  if ($0 == l1) { f1=1; print; next }
  if ($0 == l2) { f2=1; print; next }

  # Uncomment commented variants (with/without spaces)
  line = $0
  sub(/^[[:space:]]*#?[[:space:]]*/, "", line)
  if (line == l1) { f1=1; print l1; next }
  if (line == l2) { f2=1; print l2; next }

  print
}
END {
  if (!f1 || !f2) {
    print ""
    print ""
    if (!f1) print l1
    if (!f2) print l2
  }
}
' "$BASHRC" > "$tmpfile" && mv "$tmpfile" "$BASHRC"

echo "[INFO] Ensured PATH + define-functions sourcing lines in $BASHRC"


# 4) Refresh PATH in THIS shell (no sourcing of ~/.bashrc)
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac


# 5) Ensure at least one uv-managed Python is installed (install latest if none)
#    We treat "uv-managed" as: directories named cpython-* in uv's Python store.
XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
UV_PY_DIR="${XDG_DATA_HOME}/uv/python"

has_uv_managed_python="no"
if [ -d "$UV_PY_DIR" ]; then
  if find "$UV_PY_DIR" -maxdepth 1 -type d -name 'cpython-*' -print -quit \
    | command grep -q .; then
    has_uv_managed_python="yes"
  fi
fi

if [ "$has_uv_managed_python" != "yes" ]; then
  echo "[INFO] No uv-managed Python found in $UV_PY_DIR; installing latest via 'uv python install'..."
  if uv python install; then
    echo "[INFO] Installed latest Python via uv."
  else
    echo "[WARN] 'uv python install' failed. Check your network/proxy and run it manually."
  fi
else
  echo "[INFO] uv-managed Python already present; skipping Python install."
fi

echo "[OK] install.sh completed. Reload ~/.bashrc file to activate installed resources."
