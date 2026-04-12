#!/usr/bin/env bash
set -euo pipefail

# uv-cache-links.sh
# Fixed behavior (no flags):
#   - Scan ALL regular files under $UV_CACHE_DIR (or ~/.cache/uv).
#   - Group by "<top>/<second>" path components relative to cache root.
#   - Print: "<max_nlink>\t<linked>/<total>\t<group>"
#   - Sort: max_nlink desc, linked desc, total desc, group asc.

cache_dir="${UV_CACHE_DIR:-$HOME/.cache/uv}"

if [[ ! -d "$cache_dir" ]]; then
  echo "Cache directory does not exist: $cache_dir" >&2
  exit 1
fi

# Collect nlink + relative paths in a process-efficient way:
# - one `find`
# - one batched `stat` via xargs
# - one `awk` aggregation
find "$cache_dir" -type f -print0 \
  | xargs -0 -r stat -c '%h %n' -- \
  | awk -v root="$cache_dir/" '
    function group_key(path,   rel, n, parts) {
      rel = path
      sub(root, "", rel)
      n = split(rel, parts, "/")
      if (n == 1) {
        return parts[1] "/" parts[1]
      }
      return parts[1] "/" parts[2]
    }

    {
      nlink = $1
      # Reconstruct path (in case it contains spaces):
      path = $2
      for (i = 3; i <= NF; i++) {
        path = path " " $i
      }

      key = group_key(path)

      total[key] += 1
      if (nlink > 1) {
        linked[key] += 1
      }
      if (!(key in maxlink) || (nlink > maxlink[key])) {
        maxlink[key] = nlink
      }
    }

    END {
      for (k in total) {
        l = (k in linked) ? linked[k] : 0
        printf "%d\t%d/%d\t%s\n", maxlink[k], l, total[k], k
      }
    }
  ' \
  | sort -t $'\t' -k1,1nr -k2,2nr -k3,3nr -k4,4
