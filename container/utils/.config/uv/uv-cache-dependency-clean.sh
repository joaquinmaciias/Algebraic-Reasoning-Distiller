#!/usr/bin/env bash
set -euo pipefail

# uv-cache-dependency-clean.sh

DEBUG="${DEBUG:-0}"
# DEBUG=1
log() {
  if [[ "$DEBUG" == "1" ]]; then
    printf '[debug] %s\n' "$*" >&2
  fi
}

cache_dir="${UV_CACHE_DIR:-$HOME/.cache/uv}"
dry_run="${DRY_RUN:-0}"
keep_bootstrap="${KEEP_BOOTSTRAP:-1}"

log "Starting uv-cache-dependency-clean"
log "cache_dir=$cache_dir dry_run=$dry_run keep_bootstrap=$keep_bootstrap"

if [[ ! -d "$cache_dir" ]]; then
  echo "error: cache directory does not exist: $cache_dir" >&2
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
  echo "error: no valid LINK_SEARCH_ROOTS to scan." >&2
  exit 1
fi

log "Search roots:"
for r in "${valid_roots[@]}"; do
  log "  $r"
done

mapfile -d '' archive_dirs < <(
  find "$cache_dir" \
    -mindepth 1 -maxdepth 1 -type d \
    \( -name 'wheels*' -o -name '*wheels*' \) -prune -false -o \
    -name '*archive*' -print0
)

log "Found ${#archive_dirs[@]} archive dirs"
for d in "${archive_dirs[@]}"; do
  log "  archive: $d"
done

if [[ "${#archive_dirs[@]}" -eq 0 ]]; then
  echo "error: no *archive* subdirectories found under: $cache_dir" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

all_venvs_file="$tmpdir/all_venvs.txt"
: >"$all_venvs_file"

# ----------------------------------------------------------------------
# 1) Gather venv roots
# ----------------------------------------------------------------------
for adir in "${archive_dirs[@]}"; do
  inode_file="$tmpdir/inodes.txt"
  : >"$inode_file"

  log "Scanning archive dir: $adir"

  find "$adir" \
    \( -path '*/wheels/*' -o -path '*/wheels-*/*' -o -path '*/wheels*/*' \) \
    -prune -false -o \
    -type f -print0 \
    | xargs -0 -r stat -c '%i %h' -- 2>/dev/null \
    | awk '$2 > 1 { print $1 }' \
    | sort -u >"$inode_file"

  inode_count="$(wc -l <"$inode_file" || true)"
  log "  inodes with nlink>1: $inode_count"

  if [[ "$inode_count" -eq 0 ]]; then
    continue
  fi

  for root in "${valid_roots[@]}"; do
    log "  scanning root: $root"

    find "$root" -xdev \
      \( -path '*/.*' ! -path '*/.venv/*' ! -path '*/.venv' \) -prune -false -o \
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
        ' >>"$all_venvs_file"
  done
done

if [[ ! -s "$all_venvs_file" ]]; then
  echo "No venvs found referencing archive files under: $cache_dir"
  exit 0
fi

sort -u "$all_venvs_file" -o "$all_venvs_file"
venv_count="$(wc -l <"$all_venvs_file")"

log "Total unique venvs found: $venv_count"

echo "Venvs to clean:"
awk '{ print "  " $0 "/" }' "$all_venvs_file"

# ----------------------------------------------------------------------
# 2) Uninstall dependencies
# ----------------------------------------------------------------------
uninstall_from_venv() {
  local venv_root="$1"
  local py="$venv_root/bin/python"

  log "Processing venv: $venv_root"

  if [[ ! -f "$venv_root/pyvenv.cfg" ]]; then
    log "  skip: not a venv (missing pyvenv.cfg)"
    return 0
  fi
  if [[ ! -x "$py" ]]; then
    log "  skip: missing python: $py"
    return 0
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found in PATH" >&2
    return 1
  fi

  log "  venv python: $py"
  log "  uv: $(command -v uv)"

  local pkgs_file="$tmpdir/pkgs.$(printf '%s' "$venv_root" | tr '/ ' '__').txt"
  local freeze_err="$tmpdir/freeze.$(printf '%s' "$venv_root" | tr '/ ' '__').err"
  : >"$pkgs_file"
  : >"$freeze_err"

  log "  running: uv pip freeze --python $py"
  
  set +e
  uv pip freeze --python "$py" 1>"$tmpdir/freeze.out" 2>"$freeze_err"
  local freeze_rc="$?"
  set -e

  log "  freeze exit code: $freeze_rc"

  if [[ "$freeze_rc" -ne 0 ]]; then
    echo "error: uv pip freeze failed for venv: $venv_root" >&2
    echo "---- stderr ----" >&2
    sed -n '1,120p' "$freeze_err" >&2
    echo "--------------" >&2
    return 0
  fi

  awk '
    # Accept:
    #   name==version
    #   name @ url
    #   -e editable (ignore)
    #   --hash lines (ignore)
    /^[[:space:]]*$/ { next }
    /^-e[[:space:]]+/ { next }
    /^--hash=/ { next }
  
    {
      line=$0
      # name==version
      if (index(line, "==") > 0) {
        split(line, a, "==")
        name=a[1]
      } else if (index(line, " @ ") > 0) {
        split(line, a, " @ ")
        name=a[1]
      } else {
        next
      }
  
      # Trim spaces
      sub(/^[[:space:]]+/, "", name)
      sub(/[[:space:]]+$/, "", name)
  
      low=tolower(name)
      if (low == "pip" || low == "setuptools" || low == "wheel") next
  
      print name
    }
  ' "$tmpdir/freeze.out" >"$pkgs_file"


  local pkg_count
  pkg_count="$(wc -l <"$pkgs_file" || true)"
  log "  packages to uninstall: $pkg_count"

  if [[ "$pkg_count" -eq 0 ]]; then
    log "  nothing to uninstall"
    return 0
  fi

  if [[ "$dry_run" == "1" ]]; then
    log "  DRY_RUN enabled"
    awk '{ print "  would uninstall: " $0 }' "$pkgs_file"
    return 0
  fi

  log "  uninstalling packages (xargs batches)"
  if [[ "$DEBUG" == "1" ]]; then
  <"$pkgs_file" xargs -r -n 50 uv pip uninstall --python "$py" || true
  else
  <"$pkgs_file" xargs -r -n 50 uv pip uninstall --python "$py" >/dev/null || true
  fi

  log "  uninstall done"
}



while IFS= read -r venv; do
  uninstall_from_venv "$venv"
done <"$all_venvs_file"

log "Finished uv-cache-dependency-clean"
