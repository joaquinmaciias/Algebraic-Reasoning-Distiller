
__setup_traps() {
  set -Eeuo pipefail; shopt -s inherit_errexit 2>/dev/null || true
  trap 'echo "ERROR en ${FUNCNAME[0]:-MAIN} línea $LINENO: $BASH_COMMAND" >&2' ERR
}

register_github_repo() (
  __setup_traps
  set +o history

  local ssh_dir="$HOME/.ssh"
  local cfg="$ssh_dir/config"
  local repo_index="$ssh_dir/github-repo-index"
  local known_hosts="$ssh_dir/known_hosts"
  local key_index="$ssh_dir/key-index"

  local key_arg=""
  local remote_arg=""

  while [[ $# -gt 0 ]]; do
    case "${1-}" in
      --key=*)
        key_arg="${1#--key=}"
        ;;
      --remote=*)
        remote_arg="${1#--remote=}"
        ;;
      *)
        printf 'error: unknown argument: %s\n' "${1-}" >&2
        return 2
        ;;
    esac
    shift || true
  done

  if [[ -n "$remote_arg" ]]; then
    if [[ "$remote_arg" =~ [[:space:]] ]]; then
      printf 'error: --remote must not contain whitespace: %s\n' "$remote_arg" >&2
      return 2
    fi
    if [[ ! "$remote_arg" =~ ^git@github\.com:[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git$ ]]; then
      printf 'error: invalid GitHub SSH remote: %s\n' "$remote_arg" >&2
      printf 'error: expected format: git@github.com:<username>/<repo-name>.git\n' >&2
      return 2
    fi
  fi

  if [[ -n "$key_arg" ]]; then
    if [[ "$key_arg" == *"/"* ]]; then
      printf 'error: --key must be a basename (no /): %s\n' "$key_arg" >&2
      return 2
    fi
    if [[ "$key_arg" =~ [[:space:]] ]]; then
      printf 'error: --key must not contain whitespace: %s\n' "$key_arg" >&2
      return 2
    fi
    if [[ ! "$key_arg" =~ ^[A-Za-z0-9._-]+$ ]]; then
      printf 'error: --key contains invalid characters: %s\n' "$key_arg" >&2
      return 2
    fi
  fi

  mkdir -p -- "$ssh_dir" >/dev/null 2>&1 || {
    printf 'error: could not create %s\n' "$ssh_dir" >&2
    return 1
  }
  chmod 700 "$ssh_dir" >/dev/null 2>&1 || true

  if [[ ! -e "$cfg" ]]; then
    : >"$cfg" || {
      printf 'error: could not create %s\n' "$cfg" >&2
      return 1
    }
  fi
  chmod 600 "$cfg" >/dev/null 2>&1 || true

  if [[ ! -e "$repo_index" ]]; then
    : >"$repo_index" || {
      printf 'error: could not create %s\n' "$repo_index" >&2
      return 1
    }
  fi
  chmod 600 "$repo_index" >/dev/null 2>&1 || true

  if [[ ! -e "$known_hosts" ]]; then
    : >"$known_hosts" >/dev/null 2>&1 || true
  fi
  chmod 600 "$known_hosts" >/dev/null 2>&1 || true

  if [[ ! -e "$key_index" ]]; then
    : >"$key_index" || {
      printf 'error: could not create %s\n' "$key_index" >&2
      return 1
    }
  fi
  chmod 600 "$key_index" >/dev/null 2>&1 || true

  _rss__rand_digits_10() (
    __setup_traps
    local s=""
    if command -v date >/dev/null 2>&1; then
      s="$(date +%s%N 2>/dev/null || true)"
    fi
    if [[ -z "$s" ]]; then
      s="${RANDOM}${RANDOM}${RANDOM}${RANDOM}${RANDOM}"
    fi
    s="${s//[^0-9]/}"
    if [[ ${#s} -lt 10 ]]; then
      s="${s}0000000000"
    fi
    printf '%s\n' "${s: -10}"
  )

  _rss__origin_owner_repo() (
    __setup_traps
    local origin_url="${1:?missing origin url}"
    local s="$origin_url"
    s="${s%.git}"

    if [[ "$s" =~ ^git@[^:]+:(.+)/(.+)$ ]]; then
      printf '%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
      return 0
    fi
    if [[ "$s" =~ ^ssh://git@[^/]+/(.+)/(.+)$ ]]; then
      printf '%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
      return 0
    fi
    if [[ "$s" =~ ^https?://[^/]+/(.+)/(.+)$ ]]; then
      printf '%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
      return 0
    fi
    return 1
  )

  _rss__sanitize_host_from_path() (
    __setup_traps
    local p="${1:?missing path}"

    local host="github-${p}"
    host="${host//\//-}"
    host="${host// /-}"
    host="$(printf '%s' "$host" | tr -cd 'A-Za-z0-9-')"
    host="$(printf '%s' "$host" | sed -E 's/^-+//; s/-+$//; s/-+/-/g')"

    if [[ -z "$host" ]]; then
      printf 'github-repo\n'
      return 0
    fi
    printf '%s\n' "$host"
  )

  _rss__ensure_known_hosts_github() (
    __setup_traps
    if command -v ssh-keyscan >/dev/null 2>&1; then
      if ! ssh-keygen -F github.com -f "$known_hosts" >/dev/null 2>&1; then
        ssh-keyscan -t ed25519 github.com >>"$known_hosts" 2>/dev/null || true
        chmod 600 "$known_hosts" >/dev/null 2>&1 || true
      fi
    fi
  )

  _rss__repo_index_has_path() (
    __setup_traps
    local want_path="${1:?missing path}"
    awk -v want="$want_path" '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }
      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)
        if (t == "" || t ~ /^#/) next
        n=split(t, a, /[ \t]+/)
        if (n >= 3 && a[1] == want) { found=1; exit }
      }
      END { exit found?0:1 }
    ' "$repo_index"
  )

  _rss__repo_index_has_key() (
    __setup_traps
    local want_key="${1:?missing key}"
    awk -v want="$want_key" '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }
      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)
        if (t == "" || t ~ /^#/) next
        n=split(t, a, /[ \t]+/)
        if (n >= 3 && a[3] == want) { found=1; exit }
      }
      END { exit found?0:1 }
    ' "$repo_index"
  )

  _rss__remove_host_block() (
    __setup_traps
    local host="${1:?missing host}"
    local tmp=""
    tmp="$(mktemp)"

    awk -v target="$host" '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }

      BEGIN { skip=0 }

      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)

        if (t ~ /^Host[ \t]+/ && t !~ /^#/) {
          rest=t
          sub(/^Host[ \t]+/, "", rest)
          split(rest, a, /[ \t]+/)
          if (a[1] == target) {
            skip=1
            next
          }
          if (skip) skip=0
        }

        if (!skip) print line
      }
    ' "$cfg" >"$tmp"

    mv -f "$tmp" "$cfg"
    chmod 600 "$cfg" >/dev/null 2>&1 || true
  )

  _rss__upsert_repo_host_block() (
    __setup_traps
    local host="${1:?missing host}"
    local key_abs="${2:?missing key path}"

    _rss__remove_host_block "$host"

    {
      printf '\n'
      printf 'Host %s\n' "$host"
      printf '  HostName github.com\n'
      printf '  User git\n'
      printf '  IdentitiesOnly yes\n'
      printf '  IdentityFile %s\n' "$key_abs"
      printf '  StrictHostKeyChecking yes\n'
      printf '\n'
    } >>"$cfg"

    chmod 600 "$cfg" >/dev/null 2>&1 || true
  )

  _rss__format_ssh_config() (
    __setup_traps
    local tmp=""
    tmp="$(mktemp)"

    awk '
      BEGIN { blank=0 }
      {
        line=$0
        sub(/\r$/, "", line)
        if (line ~ /^[ \t]*$/) {
          if (!blank) { print ""; blank=1 }
          next
        }
        blank=0
        print line
      }
    ' "$cfg" >"$tmp"

    mv -f "$tmp" "$cfg"
    chmod 600 "$cfg" >/dev/null 2>&1 || true
  )

  _rss__sort_repo_index_by_path() (
    __setup_traps
    local tmp=""
    tmp="$(mktemp)"

    awk '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }
      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)
        if (t == "" || t ~ /^#/) next
        n=split(t, a, /[ \t]+/)
        if (n >= 3) print a[1] "\t" a[2] "\t" a[3]
      }
    ' "$repo_index" | LC_ALL=C sort -t $'\t' -k1,1 | awk -F $'\t' '
      { print $1 " " $2 " " $3 }
    ' >"$tmp"

    mv -f "$tmp" "$repo_index"
    chmod 600 "$repo_index" >/dev/null 2>&1 || true
  )

  _rss__prompt_continue_clone() (
    __setup_traps

    local ans=""
    local tty_in="/dev/tty"
    local tty_out="/dev/tty"

    while :; do
      if [[ -r "$tty_in" && -w "$tty_out" ]]; then
        printf 'Continue with cloning [y/n]? ' >"$tty_out"
        IFS= read -r ans <"$tty_in" || { printf 'error\n'; return 0; }
      else
        printf 'Continue with cloning [y/n]? ' >&2
        IFS= read -r ans || { printf 'error\n'; return 0; }
      fi

      case "${ans,,}" in
        y|yes) printf 'y\n'; return 0 ;;
        n|no)  printf 'n\n'; return 0 ;;
        *) printf 'Please answer y or n.\n' >&2 ;;
      esac
    done
  )

  _rss__add_key_to_key_index_if_missing() (
    __setup_traps
    local key_name="${1:?missing key name}"
    local key_priv="$ssh_dir/$key_name"

    [[ -f "$key_priv" ]] || return 1

    if ! grep -Fxq -- "$key_name" "$key_index" 2>/dev/null; then
      printf '%s\n' "$key_name" >>"$key_index"
      chmod 600 "$key_index" >/dev/null 2>&1 || true
    fi
    return 0
  )

  _rss__sync_key_index_from_repo_index() (
    __setup_traps

    declare -A seen=()
    local line=""
    while IFS= read -r line || [[ -n "${line:-}" ]]; do
      [[ -n "${line:-}" ]] || continue
      [[ "$line" == \#* ]] && continue
      [[ "$line" =~ [[:space:]] ]] && continue
      if [[ -f "$ssh_dir/$line" ]]; then
        seen["$line"]="1"
      fi
    done <"$key_index"

    local k=""
    while IFS= read -r k || [[ -n "${k:-}" ]]; do
      [[ -n "${k:-}" ]] || continue
      [[ "$k" =~ [[:space:]] ]] && continue

      if [[ -f "$ssh_dir/$k" && -z "${seen[$k]+x}" ]]; then
        printf '%s\n' "$k" >>"$key_index"
        seen["$k"]="1"
      fi
    done < <(
      awk '
        /^[[:space:]]*$/ { next }
        /^[[:space:]]*#/ { next }
        { if (NF >= 3) print $3 }
      ' "$repo_index" 2>/dev/null || true
    )

    chmod 600 "$key_index" >/dev/null 2>&1 || true
    return 0
  )

  _rss__check_repo_access_with_key() (
    __setup_traps
    local remote="${1:?missing remote}"
    local key_abs="${2:?missing key path}"

    command -v git >/dev/null 2>&1 || return 1

    _rss__ensure_known_hosts_github

    GIT_SSH_COMMAND="ssh -i $key_abs -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$known_hosts" \
      git ls-remote -h "$remote" >/dev/null 2>&1
  )

  _rss__sync_key_index_from_repo_index >/dev/null 2>&1 || true
  if command -v prune_github_credentials >/dev/null 2>&1; then
    prune_github_credentials >/dev/null 2>&1 || true
  fi

  # ------------------------------------------------------------
  # --remote mode (optional --key=<existing-key>)
  # ------------------------------------------------------------
  if [[ -n "$remote_arg" ]]; then
    command -v git >/dev/null 2>&1 || {
      printf 'error: git not found in PATH\n' >&2
      return 1
    }

    local start_dir=""
    start_dir="$(pwd -P)"

    local repo_dir=""
    repo_dir="${remote_arg##*/}"
    repo_dir="${repo_dir%.git}"
    [[ -n "$repo_dir" ]] || {
      printf 'error: could not derive repo directory name from --remote\n' >&2
      return 1
    }
    [[ ! -e "$repo_dir" ]] || {
      printf 'error: target path already exists: %s\n' "$start_dir/$repo_dir" >&2
      return 1
    }

    local owner_repo_for_endpoint=""
    owner_repo_for_endpoint="$(_rss__origin_owner_repo "$remote_arg")" || {
      printf 'error: could not derive owner/repo from --remote: %s\n' "$remote_arg" >&2
      return 1
    }

    local key_id="" key_name="" key_priv="" key_pub=""
    local created_key="0"

    if [[ -n "$key_arg" ]]; then
      key_name="$key_arg"
      key_priv="$ssh_dir/$key_name"
      key_pub="${key_priv}.pub"

      [[ -f "$key_priv" ]] || {
        printf 'error: ssh private key not found: %s\n' "$key_priv" >&2
        return 1
      }
      chmod 600 "$key_priv" >/dev/null 2>&1 || true

      if [[ ! -f "$key_pub" ]]; then
        printf 'warning: ssh public key not found: %s\n' "$key_pub" >&2
      fi

      if ! _rss__check_repo_access_with_key "$remote_arg" "$key_priv"; then
        printf 'error: cannot access repo with key %s\n' "$key_name" >&2
        return 1
      fi

      _rss__add_key_to_key_index_if_missing "$key_name" >/dev/null 2>&1 || true
    else
      command -v ssh-keygen >/dev/null 2>&1 || {
        printf 'error: ssh-keygen not found in PATH\n' >&2
        return 1
      }

      while :; do
        key_id="$(_rss__rand_digits_10)"
        key_name="id_ed25519_repo_${key_id}"
        _rss__repo_index_has_key "$key_name" && continue
        key_priv="$ssh_dir/$key_name"
        key_pub="${key_priv}.pub"
        [[ -e "$key_priv" || -e "$key_pub" ]] && continue
        break
      done

      ssh-keygen -t ed25519 -C "deploy-key-repo_${key_id}" -f "$key_priv" -N "" \
        >/dev/null 2>&1 || {
        printf 'error: ssh-keygen failed\n' >&2
        return 1
      }
      created_key="1"

      chmod 600 "$key_priv" >/dev/null 2>&1 || true
      chmod 644 "$key_pub" >/dev/null 2>&1 || true
      _rss__ensure_known_hosts_github

      printf 'Include the following public key in your GitHub repository (Settings → Deploy keys → Add deploy key).\n'
      printf '> ENDPOINT: https://github.com/%s/settings/keys\n' "$owner_repo_for_endpoint"
      printf '> PUBLIC KEY: %s\n' "$(cat "$key_pub")"
    fi

    local decision=""
    decision="$(_rss__prompt_continue_clone)"
    if [[ "$decision" == "n" ]]; then
      if [[ "$created_key" == "1" ]]; then
        rm -f -- "$key_priv" "$key_pub" >/dev/null 2>&1 || true
      fi
      _rss__format_ssh_config
      return 0
    fi
    if [[ "$decision" != "y" ]]; then
      printf 'error: could not read user input\n' >&2
      if [[ "$created_key" == "1" ]]; then
        rm -f -- "$key_priv" "$key_pub" >/dev/null 2>&1 || true
      fi
      return 1
    fi

    local intended_abs=""
    intended_abs="$start_dir/$repo_dir"

    local host=""
    host="$(_rss__sanitize_host_from_path "$intended_abs")"

    _rss__upsert_repo_host_block "$host" "$key_priv"
    _rss__format_ssh_config
    _rss__ensure_known_hosts_github

    local remote_for_clone=""
    remote_for_clone="git@${host}:${owner_repo_for_endpoint}.git"

    while :; do
      if git clone "$remote_for_clone" "$repo_dir"; then
        break
      fi

      printf 'Clone failed. If you just added the deploy key, wait a moment and try again.\n' >&2

      if [[ -d "$repo_dir" ]]; then
        rm -rf -- "$repo_dir" >/dev/null 2>&1 || true
      fi

      decision="$(_rss__prompt_continue_clone)"
      if [[ "$decision" == "n" ]]; then
        _rss__remove_host_block "$host" || true
        if [[ "$created_key" == "1" ]]; then
          rm -f -- "$key_priv" "$key_pub" >/dev/null 2>&1 || true
        fi
        _rss__format_ssh_config
        return 0
      fi
      if [[ "$decision" != "y" ]]; then
        printf 'error: could not read user input\n' >&2
        _rss__remove_host_block "$host" || true
        if [[ "$created_key" == "1" ]]; then
          rm -f -- "$key_priv" "$key_pub" >/dev/null 2>&1 || true
        fi
        _rss__format_ssh_config
        return 1
      fi
    done

    cd -- "$repo_dir" || {
      printf 'error: could not cd into %s\n' "$repo_dir" >&2
      _rss__remove_host_block "$host" || true
      if [[ "$created_key" == "1" ]]; then
        rm -f -- "$key_priv" "$key_pub" >/dev/null 2>&1 || true
      fi
      _rss__format_ssh_config
      return 1
    }

    local cwd_remote=""
    cwd_remote="$(pwd -P)"

    if _rss__repo_index_has_key "$key_name"; then
      printf 'error: key already present in %s: %s\n' "$repo_index" "$key_name" >&2
      return 1
    fi
    if _rss__repo_index_has_path "$cwd_remote"; then
      printf 'error: %s already has an entry in %s\n' "$cwd_remote" "$repo_index" >&2
      return 1
    fi

    printf '%s %s %s\n' "$cwd_remote" "$host" "$key_name" >>"$repo_index"
    _rss__sort_repo_index_by_path
    _rss__sync_key_index_from_repo_index >/dev/null 2>&1 || true
    _rss__format_ssh_config

    printf 'Repository registered in ssh config.\n'
    return 0
  fi

  # ------------------------------------------------------------
  # Non-remote modes (unchanged)
  # ------------------------------------------------------------
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    printf 'error: current directory is not a git repository\n' >&2
    return 1
  }

  local cwd=""
  cwd="$(pwd -P)"

  local host=""
  host="$(_rss__sanitize_host_from_path "$cwd")"

  local key_name="" key_priv="" key_pub=""
  if [[ -z "$key_arg" ]]; then
    _rss__repo_index_has_path "$cwd" && {
      printf 'error: %s already has an entry in %s\n' "$cwd" "$repo_index" >&2
      return 1
    }

    local origin_url=""
    origin_url="$(git remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || {
      printf 'error: origin remote not found\n' >&2
      return 1
    }

    local owner_repo=""
    owner_repo="$(_rss__origin_owner_repo "$origin_url")" || {
      printf 'error: could not parse owner/repo from origin: %s\n' "$origin_url" >&2
      return 1
    }

    local key_id=""
    while :; do
      key_id="$(_rss__rand_digits_10)"
      key_name="id_ed25519_repo_${key_id}"
      _rss__repo_index_has_key "$key_name" && continue
      key_priv="$ssh_dir/$key_name"
      key_pub="${key_priv}.pub"
      [[ -e "$key_priv" || -e "$key_pub" ]] && continue
      break
    done

    ssh-keygen -t ed25519 -C "deploy-key-repo_${key_id}" -f "$key_priv" -N "" || {
      printf 'error: ssh-keygen failed\n' >&2
      return 1
    }

    chmod 600 "$key_priv" >/dev/null 2>&1 || true
    chmod 644 "$key_pub" >/dev/null 2>&1 || true
    _rss__ensure_known_hosts_github

    _rss__upsert_repo_host_block "$host" "$key_priv"

    printf '%s %s %s\n' "$cwd" "$host" "$key_name" >>"$repo_index"
    _rss__sort_repo_index_by_path

    git remote set-url origin "git@${host}:${owner_repo}.git" || {
      printf 'error: failed to set origin url\n' >&2
      return 1
    }

    printf 'Include the following public key in your GitHub repository (Settings → Deploy keys → Add deploy key).\n'
    printf '> PUBLIC KEY: %s\n' "$(cat "$key_pub")"
  else
    key_name="$key_arg"
    key_priv="$ssh_dir/$key_name"
    key_pub="${key_priv}.pub"

    [[ -f "$key_priv" ]] || {
      printf 'error: ssh key not found: %s\n' "$key_priv" >&2
      return 1
    }

    _rss__repo_index_has_key "$key_name" && {
      printf 'error: key already present in %s: %s\n' "$repo_index" "$key_name" >&2
      return 1
    }
    _rss__repo_index_has_path "$cwd" && {
      printf 'error: %s already has an entry in %s\n' "$cwd" "$repo_index" >&2
      return 1
    }

    local origin_url=""
    origin_url="$(git remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || {
      printf 'error: origin remote not found\n' >&2
      return 1
    }

    local owner_repo=""
    owner_repo="$(_rss__origin_owner_repo "$origin_url")" || {
      printf 'error: could not parse owner/repo from origin: %s\n' "$origin_url" >&2
      return 1
    }

    _rss__ensure_known_hosts_github
    _rss__upsert_repo_host_block "$host" "$key_priv"

    printf '%s %s %s\n' "$cwd" "$host" "$key_name" >>"$repo_index"
    _rss__sort_repo_index_by_path

    git remote set-url origin "git@${host}:${owner_repo}.git" || {
      printf 'error: failed to set origin url\n' >&2
      return 1
    }

    printf 'Repository registered in ssh config.\n'
  fi

  _rss__sync_key_index_from_repo_index >/dev/null 2>&1 || true
  if command -v prune_ssh >/dev/null 2>&1; then
    prune_ssh >/dev/null 2>&1 || true
  fi

  chmod 600 "$cfg" >/dev/null 2>&1 || true
  chmod 600 "$repo_index" >/dev/null 2>&1 || true
  chmod 600 "$known_hosts" >/dev/null 2>&1 || true
  chmod 600 "$key_index" >/dev/null 2>&1 || true

  _rss__format_ssh_config
  return 0
)




















# -----------------------------------------------------------------------------
# prune_github_credentials
#
# Per-repo hosts:
# - github-repo-index lines are: <abs_repo_path> <hostname> <keyname>
#
# Behavior:
# - Remove entries from github-repo-index whose repo path no longer exists OR
#   whose key file ~/.ssh/<keyname> is missing.
#   For each removed entry, also:
#     - remove Host <hostname> block from ~/.ssh/config
#     - remove <keyname> from ~/.ssh/key-index (if present)
#   NOTE: does NOT delete any key files from disk.
# - Remove Host blocks (only those whose Host starts with "github-") whose
#   IdentityFile points to a key that is NOT referenced by github-repo-index.
#   Also remove the corresponding key name from ~/.ssh/key-index.
#   NOTE: does NOT delete any key files from disk.
# - Compress blank lines in ~/.ssh/config so there are never >1 consecutive.
# -----------------------------------------------------------------------------
prune_github_credentials() (
  __setup_traps

  local ssh_dir="$HOME/.ssh"
  local config_file="$ssh_dir/config"
  local repo_index="$ssh_dir/github-repo-index"
  local key_index="$ssh_dir/key-index"

  if [[ ! -d "$ssh_dir" ]]; then
    echo "error: ~/.ssh not found" >&2
    return 1
  fi
  if [[ ! -f "$repo_index" ]]; then
    echo "error: ~/.ssh/github-repo-index not found" >&2
    return 1
  fi
  if [[ ! -f "$config_file" ]]; then
    echo "error: ~/.ssh/config not found" >&2
    return 1
  fi
  if [[ ! -r "$repo_index" || ! -w "$repo_index" ]]; then
    echo "error: ~/.ssh/github-repo-index is not readable/writable" >&2
    ls -l "$repo_index" >&2 || true
    return 1
  fi
  if [[ ! -r "$config_file" || ! -w "$config_file" ]]; then
    echo "error: ~/.ssh/config is not readable/writable" >&2
    ls -l "$config_file" >&2 || true
    return 1
  fi
  if [[ -f "$key_index" && ( ! -r "$key_index" || ! -w "$key_index" ) ]]; then
    echo "error: ~/.ssh/key-index is not readable/writable" >&2
    ls -l "$key_index" >&2 || true
    return 1
  fi

  local tmp_repo tmp_cfg invalid_keys referenced_keys remove_hosts removed_keys
  local unref_hosts_removed unref_keys_removed
  local changed="0"

  tmp_repo="$(mktemp)"
  tmp_cfg="$(mktemp)"
  invalid_keys="$(mktemp)"
  referenced_keys="$(mktemp)"
  remove_hosts="$(mktemp)"
  removed_keys="$(mktemp)"
  unref_hosts_removed="$(mktemp)"
  unref_keys_removed="$(mktemp)"

  _prune_rm_keypair_best_effort() (
    __setup_traps
    local key_path="${1-}"
    if [[ -n "$key_path" && -f "$key_path" ]]; then
      rm -f -- "$key_path" || true
    fi
    if [[ -n "$key_path" && -f "${key_path}.pub" ]]; then
      rm -f -- "${key_path}.pub" || true
    fi
  )

  _prune_compress_blank_lines_file() (
    __setup_traps
    local path="${1:?missing path}"
    local tmp=""
    tmp="$(mktemp)"
    awk '
      BEGIN { blank = 0 }
      /^[[:space:]]*$/ {
        if (!blank) { print ""; blank = 1 }
        next
      }
      { print; blank = 0 }
    ' "$path" >"$tmp"
    cat "$tmp" >"$path"
    rm -f "$tmp"
  )

  _prune_sort_uniq_file() (
    __setup_traps
    local path="${1:?missing path}"
    local tmp=""
    tmp="$(mktemp)"
    awk 'NF { print }' "$path" | sort -u >"$tmp" 2>/dev/null || true
    cat "$tmp" >"$path"
    rm -f "$tmp"
  )

  _prune_remove_host_block() (
    __setup_traps
    local host="${1:?missing host}"
    local tmp=""
    tmp="$(mktemp)"

    awk -v target="$host" '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }

      BEGIN { skip=0 }

      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)

        if (t ~ /^Host[ \t]+/ && t !~ /^#/) {
          rest=t
          sub(/^Host[ \t]+/, "", rest)
          split(rest, a, /[ \t]+/)
          if (a[1] == target) {
            skip=1
            next
          }
          if (skip) skip=0
        }

        if (!skip) print line
      }
    ' "$config_file" >"$tmp"

    cat "$tmp" >"$config_file"
    rm -f "$tmp"
  )

  _prune_key_index_remove_keyname() (
    __setup_traps
    local keyname="${1:?missing keyname}"
    [[ -f "$key_index" ]] || return 0
    local tmp=""
    tmp="$(mktemp)"
    awk -v k="$keyname" '
      function trim(s){ sub(/^[ \t\r\n]+/, "", s); sub(/[ \t\r\n]+$/, "", s); return s }
      /^[[:space:]]*$/ { next }
      /^[[:space:]]*#/ { next }
      {
        t=trim($0)
        if (t == k) next
        print t
      }
    ' "$key_index" >"$tmp"
    cat "$tmp" >"$key_index"
    rm -f "$tmp"
  )

  # -----------------------------------------
  # 1) Prune github-repo-index invalid paths OR missing key files.
  #    Record:
  #      invalid key paths (abs) -> invalid_keys
  #      hosts to remove -> remove_hosts
  #      referenced key paths (abs) -> referenced_keys
  # -----------------------------------------
  awk -v ssh_dir="$ssh_dir" \
      -v out_invalid="$invalid_keys" \
      -v out_ref="$referenced_keys" \
      -v out_rmhosts="$remove_hosts" '
    function trim(s) { sub(/^[ \t\r\n]+/, "", s); sub(/[ \t\r\n]+$/, "", s); return s }
    /^[[:space:]]*$/ { next }
    /^[[:space:]]*#/ { next }
    {
      repo = $1
      host = $2
      key  = $3

      repo = trim(repo)
      host = trim(host)
      key  = trim(key)

      if (repo == "" || host == "" || key == "") next

      key_path = ssh_dir "/" key

      # Missing key => invalidate entry.
      cmdk = "[ -f \"" key_path "\" ]"
      rck = system(cmdk)
      if (rck != 0) {
        print key_path >> out_invalid
        print host >> out_rmhosts
        next
      }

      # Keep only absolute repo paths in index; otherwise consider invalid.
      if (substr(repo, 1, 1) != "/") {
        print key_path >> out_invalid
        print host >> out_rmhosts
        next
      }

      cmd = "[ -d \"" repo "\" ]"
      rc = system(cmd)
      if (rc != 0) {
        print key_path >> out_invalid
        print host >> out_rmhosts
        next
      }

      print repo " " host " " key
      print key_path >> out_ref
    }
  ' "$repo_index" >"$tmp_repo"

  _prune_sort_uniq_file "$invalid_keys"
  _prune_sort_uniq_file "$referenced_keys"
  _prune_sort_uniq_file "$remove_hosts"

  if [[ -s "$invalid_keys" || -s "$remove_hosts" ]]; then
    changed="1"
  fi

  cat "$tmp_repo" >"$repo_index"
  rm -f "$tmp_repo"

  # Record keys tied to invalid entries (but do NOT delete key files).
  if [[ -s "$invalid_keys" ]]; then
    while IFS= read -r key_path || [[ -n "${key_path:-}" ]]; do
      [[ -n "${key_path:-}" ]] || continue
      printf '%s\n' "$key_path" >>"$removed_keys"
      _prune_key_index_remove_keyname "$(basename "$key_path")"
    done <"$invalid_keys"
  fi

  # Remove host blocks tied to invalid entries (including missing keys).
  if [[ -s "$remove_hosts" ]]; then
    while IFS= read -r h || [[ -n "${h:-}" ]]; do
      [[ -n "${h:-}" ]] || continue
      _prune_remove_host_block "$h"
    done <"$remove_hosts"
  fi

  # -----------------------------------------
  # 2) Remove any per-repo Host blocks whose IdentityFile key is unreferenced.
  #    Also record removed hosts + key paths (abs) (but do NOT delete keys).
  # -----------------------------------------
  : >"$unref_hosts_removed"
  : >"$unref_keys_removed"

  awk -v refs="$referenced_keys" \
      -v out_hosts="$unref_hosts_removed" \
      -v out_keys="$unref_keys_removed" '
    BEGIN {
      while ((getline r < refs) > 0) {
        gsub(/^[ \t]+|[ \t]+$/, "", r)
        if (r != "") ref[r] = 1
      }
      close(refs)
    }

    function ltrim(s) { sub(/^[ \t]+/, "", s); return s }
    function rtrim(s) { sub(/[ \t]+$/, "", s); return s }
    function trim(s)  { return rtrim(ltrim(s)) }

    function expand_home(p,  q) {
      q = trim(p)
      sub(/[ \t]+#.*/, "", q)
      if (q ~ /^~\//) {
        q = ENVIRON["HOME"] "/" substr(q, 3)
      } else if (q == "~") {
        q = ENVIRON["HOME"]
      } else if (q ~ /^\$HOME\//) {
        q = ENVIRON["HOME"] "/" substr(q, 7)
      } else if (q == "$HOME") {
        q = ENVIRON["HOME"]
      }
      return q
    }

    BEGIN { in_host=0; cur_host=""; cur_keep=1; nlines=0; removed_this=0; rm_key="" }

    function flush_block() {
      if (in_host) {
        if (cur_keep) {
          for (i=1; i<=nlines; i++) print lines[i]
        } else {
          if (removed_this && cur_host != "") {
            print cur_host >> out_hosts
          }
          if (removed_this && rm_key != "") {
            print rm_key >> out_keys
          }
        }
      }
      delete lines
      nlines=0
      in_host=0
      cur_host=""
      cur_keep=1
      removed_this=0
      rm_key=""
    }

    {
      line=$0
      sub(/\r$/, "", line)
      t=trim(line)

      if (t ~ /^Host[ \t]+/ && t !~ /^#/) {
        flush_block()

        rest=t
        sub(/^Host[ \t]+/, "", rest)
        split(rest, a, /[ \t]+/)
        cur_host=a[1]
        in_host=1
        cur_keep=1
        removed_this=0
        rm_key=""
        nlines=0
      }

      if (!in_host) {
        print line
        next
      }

      nlines++
      lines[nlines]=line

      if (t ~ /^IdentityFile[ \t]+/ && t !~ /^#/) {
        p=t
        sub(/^IdentityFile[ \t]+/, "", p)
        p=expand_home(p)

        if (cur_host ~ /^github-/) {
          if (!(p in ref)) {
            cur_keep=0
            removed_this=1
            rm_key=p
          }
        }
      }
    }

    END { flush_block() }
  ' "$config_file" >"$tmp_cfg"

  cat "$tmp_cfg" >"$config_file"
  rm -f "$tmp_cfg"

  _prune_sort_uniq_file "$unref_hosts_removed"
  _prune_sort_uniq_file "$unref_keys_removed"

  if [[ -s "$unref_hosts_removed" || -s "$unref_keys_removed" ]]; then
    changed="1"
  fi

  if [[ -s "$unref_keys_removed" ]]; then
    while IFS= read -r key_path || [[ -n "${key_path:-}" ]]; do
      [[ -n "${key_path:-}" ]] || continue
      printf '%s\n' "$key_path" >>"$removed_keys"
      _prune_key_index_remove_keyname "$(basename "$key_path")"
    done <"$unref_keys_removed"
  fi

  _prune_compress_blank_lines_file "$config_file"
  chmod 600 "$config_file" 2>/dev/null || true
  chmod 600 "$repo_index" 2>/dev/null || true
  [[ -f "$key_index" ]] && chmod 600 "$key_index" 2>/dev/null || true

  _prune_sort_uniq_file "$removed_keys"

  if [[ "$changed" == "1" ]]; then
    echo "Pruned ~/.ssh/github-repo-index and cleaned per-repo Host blocks."
    if [[ -s "$removed_keys" ]]; then
      while IFS= read -r key_path || [[ -n "${key_path:-}" ]]; do
        [[ -n "${key_path:-}" ]] || continue
        printf -- "- %s\n" "$key_path"
      done <"$removed_keys"
    fi
    if [[ -s "$remove_hosts" ]]; then
      while IFS= read -r h || [[ -n "${h:-}" ]]; do
        [[ -n "${h:-}" ]] || continue
        printf -- "- host: %s\n" "$h"
      done <"$remove_hosts"
    fi
    if [[ -s "$unref_hosts_removed" ]]; then
      while IFS= read -r h || [[ -n "${h:-}" ]]; do
        [[ -n "${h:-}" ]] || continue
        printf -- "- host: %s\n" "$h"
      done <"$unref_hosts_removed"
    fi
  else
    echo "Repository index is consistent (no invalid repo paths or missing key references)."
  fi

  rm -f "$tmp_repo" "$tmp_cfg" "$invalid_keys" "$referenced_keys" \
        "$remove_hosts" "$removed_keys" "$unref_hosts_removed" \
        "$unref_keys_removed"
)






prune_ssh_keys() (
  __setup_traps

  local ssh_dir="$HOME/.ssh"
  local key_index="$ssh_dir/key-index"

  mkdir -p -- "$ssh_dir" >/dev/null 2>&1 || return 1
  chmod 700 "$ssh_dir" >/dev/null 2>&1 || true

  if [[ ! -e "$key_index" ]]; then
    : >"$key_index" || return 1
  fi
  chmod 600 "$key_index" >/dev/null 2>&1 || true

  local tmp=""
  tmp="$(mktemp)"

  # Track removals (private keys only)
  local removed_tmp=""
  removed_tmp="$(mktemp)"
  : >"$removed_tmp"

  # 1) Format ~/.ssh/key-index:
  # - no more than one consecutive blank line
  # - keep lines that are:
  #   * blank
  #   * start with '#'
  #   * exact key filename existing in ~/.ssh (private key file)
  # - drop everything else
  local prev_blank="0"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    if [[ "$line" == "" ]]; then
      if [[ "$prev_blank" == "0" ]]; then
        printf '\n' >>"$tmp"
        prev_blank="1"
      fi
      continue
    fi

    prev_blank="0"

    if [[ "$line" == \#* ]]; then
      printf '%s\n' "$line" >>"$tmp"
      continue
    fi

    # Any whitespace makes it invalid as a "key name per line".
    if [[ "$line" =~ [[:space:]] ]]; then
      continue
    fi

    if [[ -f "$ssh_dir/$line" ]]; then
      printf '%s\n' "$line" >>"$tmp"
    fi
  done <"$key_index"

  cat "$tmp" >"$key_index"
  rm -f "$tmp" >/dev/null 2>&1 || true

  # 2) Build declared keys set from formatted key-index
  declare -A declared=()
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    [[ -n "$line" ]] || continue
    [[ "$line" == \#* ]] && continue
    declared["$line"]="1"
  done <"$key_index"

  # 3) Remove from ~/.ssh all keys not declared in key-index.
  # We consider a "key" any regular file that has a matching ".pub",
  # and we delete both the private and its ".pub" when unlisted.
  shopt -s nullglob
  local f=""
  for f in "$ssh_dir"/*; do
    local base=""
    base="$(basename -- "$f")"

    # Skip non-regular files
    [[ -f "$f" ]] || continue

    # Skip known non-key files
    case "$base" in
      config|known_hosts|authorized_keys|github-repo-index|key-index|*.log|*.bak)
        continue
        ;;
    esac

    # Skip public key files; handled with their private counterpart
    if [[ "$base" == *.pub ]]; then
      continue
    fi

    # Only treat as "private key" if it has a corresponding public key file
    if [[ ! -f "$f.pub" ]]; then
      continue
    fi

    if [[ -z "${declared[$base]+x}" ]]; then
      rm -f -- "$f" "$f.pub" >/dev/null 2>&1 || true
      printf '%s\n' "$base" >>"$removed_tmp"
    fi
  done

  # Also remove stray .pub files whose private key is missing/unlisted
  for f in "$ssh_dir"/*.pub; do
    local pub_base=""
    pub_base="$(basename -- "$f")"
    local priv_base="${pub_base%.pub}"

    [[ -f "$ssh_dir/$priv_base" ]] && continue
    if [[ -z "${declared[$priv_base]+x}" ]]; then
      rm -f -- "$f" >/dev/null 2>&1 || true
    fi
  done

  chmod 600 "$key_index" >/dev/null 2>&1 || true

  # Report removals (private keys only)
  if [[ -s "$removed_tmp" ]]; then
    sort -u "$removed_tmp" >/dev/null 2>&1 || true
    echo "Pruned SSH keys (removed unlisted private keys from ~/.ssh):"
    sort -u "$removed_tmp" | while IFS= read -r k || [[ -n "${k:-}" ]]; do
      [[ -n "${k:-}" ]] || continue
      printf -- "- %s\n" "$k"
    done
  else
    echo "No SSH keys pruned (no unlisted private keys found)."
  fi

  rm -f "$removed_tmp" >/dev/null 2>&1 || true
  return 0
)





prune_ssh() (
  prune_github_credentials
  prune_ssh_keys
)