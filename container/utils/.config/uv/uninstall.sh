#!/usr/bin/env bash

# uninstall.sh
# ----------
# Auto‐detect and strip CRLF
if grep -q $'\r' "$0"; then
  echo "[INFO] Converting DOS line endings to LF in $0..."
  sed -i 's/\r$//' "$0"
fi

set -euo pipefail

# 1) Remove uv cache directory (avoids hanging on file locks)
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/uv"
if [ -d "$CACHE_DIR" ]; then
  echo "[INFO] Removing uv cache at $CACHE_DIR..."
  rm -rf "$CACHE_DIR"
fi

# 2) Remove config & data
rm -rf "$HOME/.config/uv" "$HOME/.local/share/uv"

# 3) Remove uv binary
rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"

# 4) Remove all shims
rm -rf "$HOME/.local/uv-shims"

# 5) Clean up PATH entries in your shell profile
PROFILE="$HOME/.bashrc"
[[ -n "${ZSH_VERSION:-}" ]] && PROFILE="$HOME/.zshrc"

# Delete any line that ends in "# uv bin" or "# uv shims"
if grep -Eq '# uv bin\s*$|# uv shims\s*$' "$PROFILE"; then
  sed -i.bak \
    -e '/# uv bin\s*$/d' \
    -e '/# uv shims\s*$/d' \
    "$PROFILE"
  echo "[INFO] Removed uv PATH entries from $PROFILE (backup at ${PROFILE}.bak)"
fi

echo "[INFO] uv uninstallation complete."
