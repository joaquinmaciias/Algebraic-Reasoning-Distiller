#!/usr/bin/env bash
set -euo pipefail

# uv-cache-venv-links.sh
#
# For each /*archive* subdir under $UV_CACHE_DIR (default ~/.cache/uv):
#   - Ignore wheels completely.
#   - Find files in the archive dir with nlink > 1, collect their inodes.
#   - Scan LINK_SEARCH_ROOTS once (per archive dir) and match by inode.
#   - Print ONLY the venv roots that contain matching hardlinks, deduped:
#       <archive_dir>
#         <venv_root>/
#
# Configure search roots (space-separated absolute paths):
#   LINK_SEARCH_ROOTS="/mnt/workdata/data $HOME" ./uv-cache-venv-links.sh
#
# Defaults:
#   - If /mnt/workdata/data exists: scan only /mnt/workdata/data (fast)
#   - Else: scan $HOME
#
# Notes:
#   - Assumes Linux venv layout: <venv>/lib/pythonX.Y/site-packages/...
#   - Paths may contain spaces but not newlines.

cache_dir="${UV_CACHE_DIR:-$HOME/.cache/uv}"

echo $cache_dir

if [[ ! -d "$cache_dir" ]]; then
  echo "Cache directory does not exist: $cache_dir" >&2
  exit 1
fi

if [[ -n "${LINK_SEARCH_ROOTS:-}" ]]; then
  # shellcheck disable=SC2206
  search_roots=(${LINK_SEARCH_ROOTS})
else
  if [[ -d "/mnt/workdata/data" ]]; then
    search_roots=("/mnt/workdata/data")
  else
    search_roots=("$HOME")
  fi
fi

valid_roots=()
for r in "${search_roots[@]}"; do
  if [[ -d "$r" ]]; then
    valid_roots+=("$r")
  fi
done

if [[ "${#valid_roots[@]}" -eq 0 ]]; then
  echo "No valid LINK_SEARCH_ROOTS to scan." >&2
  exit 1
fi

mapfile -d '' archive_dirs < <(
  find "$cache_dir" \
    -mindepth 1 -maxdepth 1 -type d \
    \( -name 'wheels*' -o -name '*wheels*' \) -prune -false -o \
    -name '*archive*' -print0
)

if [[ "${#archive_dirs[@]}" -eq 0 ]]; then
  echo "No *archive* subdirectories found under: $cache_dir" >&2
  exit 0
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

for adir in "${archive_dirs[@]}"; do
  echo "$adir"

  inode_file="$tmpdir/inodes.txt"
  out_file="$tmpdir/venvs.txt"
  : >"$inode_file"
  : >"$out_file"

  # Collect unique inodes of linked files under this archive dir.
  find "$adir" \
    \( -path '*/wheels/*' -o -path '*/wheels-*/*' -o -path '*/wheels*/*' \) \
    -prune -false -o \
    -type f -print0 \
    | xargs -0 -r stat -c '%i %h' -- 2>/dev/null \
    | awk '$2 > 1 { print $1 }' \
    | sort -u >"$inode_file"

  if [[ ! -s "$inode_file" ]]; then
    echo "  (no linked files)"
    continue
  fi

  for root in "${valid_roots[@]}"; do
    # Avoid scanning inside the cache itself if user includes it.
    if [[ "$root" == "$cache_dir" || "$root" == "$cache_dir/"* ]]; then
      continue
    fi

    # Scan once, filter by inode set, collapse to venv root.
    find "$root" -xdev \
      \( -path '*/wheels/*' -o -path '*/wheels-*/*' -o -path '*/wheels*/*' \) \
      -prune -false -o \
      -type f -printf '%i\t%p\n' 2>/dev/null \
      | awk -v inode_path="$inode_file" '
          BEGIN {
            while ((getline ino < inode_path) > 0) {
              want[ino] = 1
            }
            close(inode_path)
          }

          function venv_root(p,   r) {
            # Linux venv: <venv>/lib/pythonX.Y/site-packages/...
            r = p
            if (sub(/\/lib\/python[^/]*\/site-packages\/.*/, "", r)) {
              return r
            }
            return ""
          }

          {
            ino = $1
            p = $0
            sub(/^[^\t]*\t/, "", p)

            if (!(ino in want)) next

            vr = venv_root(p)
            if (vr == "") next

            print vr
          }
        ' >>"$out_file"
  done

  if [[ ! -s "$out_file" ]]; then
    echo "  (no venvs found under search roots)"
    continue
  fi

  sort -u "$out_file" | awk '{ print "  " $0 "/" }'
done
