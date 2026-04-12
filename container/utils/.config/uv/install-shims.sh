#!/usr/bin/env bash

# install.sh
# ----------------
# Auto-detect and strip CRLF if present (robust even when sourced)
SELF="${BASH_SOURCE[0]:-$0}"

if command grep -q $'\r' -- "$SELF"; then
  echo "[INFO] Converting DOS line endings to LF in $SELF..."
  command sed -i 's/\r$//' -- "$SELF"
fi


set -o pipefail
set -u

# 1) Ensure local bin dir exists
UVBIN="$HOME/.local/bin"
mkdir -p "$UVBIN"

# 2) Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] Installing uv..."
  curl -sSL https://astral.sh/uv/install.sh | bash >/dev/null 2>&1
fi

# 3) Ensure PATH lines in ~/.bashrc
BASHRC="$HOME/.bashrc"

LINE1='[[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && export PATH="$HOME/.local/bin:$PATH"  # uv bin'
LINE2='[[ ":$PATH:" != *":$HOME/.local/uv-shims:"* ]] && export PATH="$HOME/.local/uv-shims:$PATH"  # uv shims'
LINE3='[ -f "$HOME/.local/uv-shims/create_shims.sh" ] && chmod +x "$HOME/.local/uv-shims/create_shims.sh" && "$HOME/.local/uv-shims/create_shims.sh"'

[ -f "$BASHRC" ] || : > "$BASHRC"

tmpfile="$(mktemp)"
awk -v l1="$LINE1" -v l2="$LINE2" -v l3="$LINE3" '
BEGIN { f1=0; f2=0; f3=0 }
{
  # Keep exact matches
  if ($0 == l1) { f1=1; print; next }
  if ($0 == l2) { f2=1; print; next }
  if ($0 == l3) { f3=1; print; next }

  # Uncomment commented variants (with/without spaces)
  line = $0
  sub(/^[[:space:]]*#?[[:space:]]*/, "", line)
  if (line == l1) { f1=1; print l1; next }
  if (line == l2) { f2=1; print l2; next }
  if (line == l3) { f3=1; print l3; next }

  print
}
END {
  if (!f1 || !f2 || !f3) {
    print ""
    print ""
    if (!f1) print l1
    if (!f2) print l2
    if (!f3) print l3
  }
}
' "$BASHRC" > "$tmpfile" && mv "$tmpfile" "$BASHRC"

echo "[INFO] Ensured PATH + create_shims sourcing lines in $BASHRC"

# 4) Create shims if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/create-shims.sh" ]; then
  "$SCRIPT_DIR/create-shims.sh"
elif [ -f "$SCRIPT_DIR/create-shims.sh" ]; then
  bash "$SCRIPT_DIR/create-shims.sh"
else
  echo "[INFO] Skipping create-shims.sh (not found)"
fi

# 4.5) Ensure create_shims.sh exists inside uv-shims
UVSHIMS_DIR="$HOME/.local/uv-shims"
mkdir -p "$UVSHIMS_DIR"

# Copy the local create-shims.sh into uv-shims as create_shims.sh (underscore name)
if [ -f "$SCRIPT_DIR/create-shims.sh" ]; then
  cp -f "$SCRIPT_DIR/create-shims.sh" "$UVSHIMS_DIR/create_shims.sh"
  chmod +x "$UVSHIMS_DIR/create_shims.sh" || true
  echo "[INFO] Installed $UVSHIMS_DIR/create_shims.sh"
else
  echo "[INFO] Skipping uv-shims/create_shims.sh install (not found: $SCRIPT_DIR/create-shims.sh)"
fi

# 5) Refresh PATH in THIS shell (no sourcing of ~/.bashrc)
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
case ":$PATH:" in
  *":$HOME/.local/uv-shims:"*) : ;;
  *) export PATH="$HOME/.local/uv-shims:$PATH" ;;
esac

# 6) Ensure at least one uv-managed Python is installed (install latest if none)
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
    # If you rely on shims, refresh them after installation.
    if [ -x "$SCRIPT_DIR/create-shims.sh" ]; then
      "$SCRIPT_DIR/create-shims.sh"
    elif [ -f "$SCRIPT_DIR/create-shims.sh" ]; then
      bash "$SCRIPT_DIR/create-shims.sh"
    fi
  else
    echo "[WARN] 'uv python install' failed. Check your network/proxy and run it manually."
  fi
else
  echo "[INFO] uv-managed Python already present; skipping Python install."
fi

echo "[OK] install.sh completed. Reload ~/.bashrc file to activate installed resources."
