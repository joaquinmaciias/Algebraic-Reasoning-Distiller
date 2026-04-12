register-github-repo() {
  # register-github-repo
  #
  # Usage:
  #   register-github-repo
  #   register-github-repo --key=<keyname>
  #   register-github-repo --remote=git@github.com:<username>/<repo-name>.git
  #
  # Important:
  # - No `set -e`, to avoid exiting an interactive shell/container on controlled errors.
  # - History is disabled during execution to avoid polluting ~/.bash_history.

  local _rss__was_history="off"
  local _rss__was_nounset="off"
  local _rss__was_pipefail="off"

  if [[ "$(set -o | awk '$1=="history"{print $2; exit}')" == "on" ]]; then
    _rss__was_history="on"
  fi
  if [[ "$-" == *u* ]]; then
    _rss__was_nounset="on"
  fi
  if [[ "$(set -o | awk '$1=="pipefail"{print $2; exit}')" == "on" ]]; then
    _rss__was_pipefail="on"
  fi

  set +o history
  set -u
  set -o pipefail

  _rss__exit() {
    local code="${1:-0}"

    if [[ "$_rss__was_nounset" != "on" ]]; then
      set +u
    fi
    if [[ "$_rss__was_pipefail" != "on" ]]; then
      set +o pipefail
    fi
    if [[ "$_rss__was_history" == "on" ]]; then
      set -o history
    fi

    return "$code"
  }

  local ssh_dir="$HOME/.ssh"
  local cfg="$ssh_dir/config"
  local repo_index="$ssh_dir/github-repo-index"

  local key_arg=""
  local remote_arg=""

  prune-github-credentials

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
        _rss__exit 2
        return $?
        ;;
    esac
    shift || true
  done

  if [[ -n "$remote_arg" && -n "$key_arg" ]]; then
    printf 'error: --remote and --key are not compatible\n' >&2
    _rss__exit 2
    return $?
  fi

  if [[ -n "$remote_arg" ]]; then
    if [[ "$remote_arg" =~ [[:space:]] ]]; then
      printf 'error: --remote must not contain whitespace: %s\n' "$remote_arg" >&2
      _rss__exit 2
      return $?
    fi
    if [[ ! "$remote_arg" =~ ^git@github\.com:[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git$ ]]; then
      printf 'error: invalid GitHub SSH remote: %s\n' "$remote_arg" >&2
      printf 'error: expected format: git@github.com:<username>/<repo-name>.git\n' >&2
      _rss__exit 2
      return $?
    fi
  fi

  if [[ -n "$key_arg" ]]; then
    if [[ "$key_arg" == *"/"* ]]; then
      printf 'error: --key must be a basename (no /): %s\n' "$key_arg" >&2
      _rss__exit 2
      return $?
    fi
    if [[ "$key_arg" =~ [[:space:]] ]]; then
      printf 'error: --key must not contain whitespace: %s\n' "$key_arg" >&2
      _rss__exit 2
      return $?
    fi
    if [[ ! "$key_arg" =~ ^[A-Za-z0-9._-]+$ ]]; then
      printf 'error: --key contains invalid characters: %s\n' "$key_arg" >&2
      _rss__exit 2
      return $?
    fi
  fi

  mkdir -p "$ssh_dir" >/dev/null 2>&1 || {
    printf 'error: could not create %s\n' "$ssh_dir" >&2
    _rss__exit 1
    return $?
  }
  chmod 700 "$ssh_dir" >/dev/null 2>&1 || true

  if [[ ! -e "$cfg" ]]; then
    : >"$cfg" || {
      printf 'error: could not create %s\n' "$cfg" >&2
      _rss__exit 1
      return $?
    }
  fi
  chmod 600 "$cfg" >/dev/null 2>&1 || true

  (
    set -o noclobber
    : >"$repo_index"
  ) >/dev/null 2>&1 || true

  if [[ ! -e "$repo_index" ]]; then
    printf 'error: could not create %s\n' "$repo_index" >&2
    _rss__exit 1
    return $?
  fi
  chmod 600 "$repo_index" >/dev/null 2>&1 || true

  # --------------------------------------------------------------------------
  # Ensure GitHub host key is present when StrictHostKeyChecking yes is used.
  # This prevents first-run failures when ~/.ssh/known_hosts does not exist.
  # --------------------------------------------------------------------------
  local known_hosts="$ssh_dir/known_hosts"
  if [[ ! -e "$known_hosts" ]]; then
    : >"$known_hosts" >/dev/null 2>&1 || true
  fi
  chmod 600 "$known_hosts" >/dev/null 2>&1 || true
  if command -v ssh-keyscan >/dev/null 2>&1; then
    if ! ssh-keygen -F github.com -f "$known_hosts" >/dev/null 2>&1; then
      ssh-keyscan -t ed25519 github.com >>"$known_hosts" 2>/dev/null || true
      chmod 600 "$known_hosts" >/dev/null 2>&1 || true
    fi
  fi

  _rss__origin_owner_repo() {
    local origin_url="$1"
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
  }

  _rss__repo_index_has_path() {
    local want_path="$1"
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
        if (n >= 2 && a[1] == want) { found=1; exit }
      }
      END { exit found?0:1 }
    ' "$repo_index"
  }

  _rss__repo_index_has_key() {
    local want_key="$1"
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
        if (n >= 2 && a[2] == want) { found=1; exit }
      }
      END { exit found?0:1 }
    ' "$repo_index"
  }

  _rss__ensure_github_host_block_exists() {
    if awk '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }
      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)
        if (t ~ /^Host[ \t]+github\.com([ \t]|$)/ && t !~ /^#/) { found=1; exit }
      }
      END { exit found?0:1 }
    ' "$cfg"; then
      return 0
    fi

    {
      printf '\n'
      printf 'Host github.com\n'
      printf '  HostName github.com\n'
      printf '  User git\n'
      printf '  IdentitiesOnly yes\n'
      printf '  StrictHostKeyChecking yes\n'
      printf '\n'
    } >>"$cfg" || return 1

    return 0
  }

  _rss__append_identityfile_to_github_block() {
    local key_abs="$1"

    local tmp=""
    tmp="$(mktemp)" || return 1

    if ! awk -v key="$key_abs" '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }

      BEGIN {
        in_github=0
        inserted=0
      }

      function flush_insert_if_needed() {
        if (in_github && !inserted) {
          print "  IdentityFile " key
          inserted=1
        }
      }

      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)

        if (t ~ /^Host[ \t]+/ && t !~ /^#/) {
          if (in_github) {
            flush_insert_if_needed()
            in_github=0
          }

          rest=t
          sub(/^Host[ \t]+/, "", rest)
          split(rest, a, /[ \t]+/)
          if (a[1] == "github.com") {
            in_github=1
            inserted=0
          }
          print line
          next
        }

        if (in_github && t ~ /^(IdentitiesOnly|StrictHostKeyChecking)[ \t]+/ && t !~ /^#/) {
          flush_insert_if_needed()
          print line
          next
        }

        print line
      }

      END {
        if (in_github) {
          flush_insert_if_needed()
        }
      }
    ' "$cfg" >"$tmp"; then
      rm -f "$tmp" >/dev/null 2>&1 || true
      return 1
    fi

    if ! mv -f "$tmp" "$cfg"; then
      rm -f "$tmp" >/dev/null 2>&1 || true
      return 1
    fi

    return 0
  }

  _rss__github_block_has_identityfile_for_key() {
    local key_abs="$1"
    local key_base="$2"
    local key_tilde="~/.ssh/${key_base}"

    awk -v want1="$key_abs" -v want2="$key_tilde" '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }

      BEGIN { in_github=0; found=0 }

      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)

        if (t ~ /^Host[ \t]+/ && t !~ /^#/) {
          rest=t
          sub(/^Host[ \t]+/, "", rest)
          split(rest, a, /[ \t]+/)
          in_github = (a[1] == "github.com") ? 1 : 0
        }

        if (in_github && t ~ /^IdentityFile[ \t]+/ && t !~ /^#/) {
          p=t
          sub(/^IdentityFile[ \t]+/, "", p)
          p=trim(p)
          if (p == want1 || p == want2) {
            found=1
            exit
          }
        }
      }

      END { exit found?0:1 }
    ' "$cfg"
  }

  _rss__sort_repo_index_by_path() {
    local tmp=""
    tmp="$(mktemp)" || return 1

    if ! awk '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }
      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)
        if (t == "" || t ~ /^#/) next
        n=split(t, a, /[ \t]+/)
        if (n >= 2) print a[1] "\t" a[2]
      }
    ' "$repo_index" | LC_ALL=C sort -t $'\t' -k1,1 | awk -F $'\t' '
      { print $1 " " $2 }
    ' >"$tmp"; then
      rm -f "$tmp" >/dev/null 2>&1 || true
      return 1
    fi

    if ! mv -f "$tmp" "$repo_index"; then
      rm -f "$tmp" >/dev/null 2>&1 || true
      return 1
    fi

    return 0
  }

  _rss__prompt_continue_clone() {
    local ans=""
    while :; do
      printf 'Continue with cloning [y/n]? '
      IFS= read -r ans || return 1
      case "${ans,,}" in
        y|yes) return 0 ;;
        n|no) return 2 ;;
        *) printf 'Please answer y or n.\n' >&2 ;;
      esac
    done
  }

  # ------------------------------------------------------------
  # --remote mode (DO ALL DIRECTORY CHECKS BEFORE KEY CREATION)
  # ------------------------------------------------------------
  if [[ -n "$remote_arg" ]]; then
    if ! command -v ssh-keygen >/dev/null 2>&1; then
      printf 'error: ssh-keygen not found in PATH\n' >&2
      _rss__exit 1
      return $?
    fi
    if ! command -v git >/dev/null 2>&1; then
      printf 'error: git not found in PATH\n' >&2
      _rss__exit 1
      return $?
    fi

    local start_dir=""
    start_dir="$(pwd -P)"

    local repo_dir=""
    repo_dir="${remote_arg##*/}"
    repo_dir="${repo_dir%.git}"

    if [[ -z "$repo_dir" ]]; then
      printf 'error: could not derive repo directory name from --remote\n' >&2
      _rss__exit 1
      return $?
    fi

    if [[ -e "$repo_dir" ]]; then
      printf 'error: target path already exists: %s\n' "$repo_dir" >&2
      _rss__exit 1
      return $?
    fi

    # Only now create key pair + touch ssh config.
    local key_id=""
    local key_name=""
    local key_priv=""
    local key_pub=""

    while :; do
      key_id="$(LC_ALL=C tr -dc '0-9' </dev/urandom | head -c 10)"
      key_name="id_ed25519_repo_${key_id}"

      if _rss__repo_index_has_key "$key_name"; then
        continue
      fi

      key_priv="$ssh_dir/$key_name"
      key_pub="${key_priv}.pub"

      if [[ -e "$key_priv" || -e "$key_pub" ]]; then
        continue
      fi

      break
    done

    if ! ssh-keygen -t ed25519 \
      -C "deploy-key-repo_${key_id}" \
      -f "$key_priv" \
      -N ""; then
      printf 'error: ssh-keygen failed\n' >&2
      _rss__exit 1
      return $?
    fi

    chmod 600 "$key_priv" >/dev/null 2>&1 || true
    chmod 644 "$key_pub" >/dev/null 2>&1 || true

    if ! _rss__ensure_github_host_block_exists; then
      printf 'error: failed to ensure github.com entry in %s\n' "$cfg" >&2
      _rss__exit 1
      return $?
    fi

    if ! _rss__append_identityfile_to_github_block "$key_priv"; then
      printf 'error: failed to add IdentityFile to github.com entry in %s\n' "$cfg" >&2
      _rss__exit 1
      return $?
    fi

    # Print endpoint + public key (only in --remote mode)
    local owner_repo_for_endpoint=""
    owner_repo_for_endpoint="$(_rss__origin_owner_repo "$remote_arg")" || {
      printf 'error: could not derive owner/repo from --remote: %s\n' "$remote_arg" >&2
      _rss__exit 1
      return $?
    }

    printf 'Include the following public key in your GitHub repository (Settings → Deploy keys → Add deploy key).\n'
    printf '> ENDPOINT: https://github.com/%s/settings/keys\n' "$owner_repo_for_endpoint"
    printf '> PUBLIC KEY: %s\n' "$(cat "$key_pub")"

    while :; do
      local rc_prompt=0
      _rss__prompt_continue_clone
      rc_prompt=$?

      if [[ $rc_prompt -eq 2 ]]; then
        if command -v prune-github-credentials >/dev/null 2>&1; then
          prune-github-credentials >/dev/null 2>&1 || true
        fi
        _rss__exit 0
        return $?
      fi
      if [[ $rc_prompt -ne 0 ]]; then
        printf 'error: could not read user input\n' >&2
        _rss__exit 1
        return $?
      fi

      if git clone "$remote_arg" "$repo_dir"; then
        break
      fi

      printf 'Clone failed. If you just added the deploy key, wait a moment and try again.\n' >&2
      if [[ -d "$repo_dir" ]]; then
        rm -rf -- "$repo_dir" >/dev/null 2>&1 || true
      fi
    done

    if ! cd -- "$repo_dir"; then
      printf 'error: could not cd into %s\n' "$repo_dir" >&2
      _rss__exit 1
      return $?
    fi

    local cwd_remote=""
    cwd_remote="$(pwd -P)"

    if [[ ! -f "$key_priv" ]]; then
      printf 'error: ssh key not found: %s\n' "$key_priv" >&2
      _rss__exit 1
      return $?
    fi

    if _rss__repo_index_has_key "$key_name"; then
      printf 'error: key already present in %s: %s\n' "$repo_index" "$key_name" >&2
      _rss__exit 1
      return $?
    fi
    if _rss__repo_index_has_path "$cwd_remote"; then
      printf 'error: %s already has an entry in %s\n' "$cwd_remote" "$repo_index" >&2
      _rss__exit 1
      return $?
    fi

    {
      printf '%s %s\n' "$cwd_remote" "$key_name"
    } >>"$repo_index" || {
      printf 'error: could not write to %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    }

    if ! _rss__sort_repo_index_by_path; then
      printf 'error: failed to sort %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    fi

    chmod 600 "$cfg" >/dev/null 2>&1 || true
    chmod 600 "$repo_index" >/dev/null 2>&1 || true

    printf 'Repository registered in ssh config.\n'

    _rss__exit 0
    return $?
  fi

  # ------------------------------------------------------------
  # Non-remote modes: require we are inside a git repo
  # ------------------------------------------------------------
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'error: current directory is not a git repository\n' >&2
    _rss__exit 1
    return $?
  fi

  local cwd=""
  cwd="$(pwd -P)"

  local key_name=""
  local key_priv=""
  local key_pub=""

  if [[ -z "$key_arg" ]]; then
    if _rss__repo_index_has_path "$cwd"; then
      printf 'error: %s already has an entry in %s\n' "$cwd" "$repo_index" >&2
      _rss__exit 1
      return $?
    fi

    local origin_url=""
    origin_url="$(git remote get-url origin 2>/dev/null || true)"
    if [[ -z "$origin_url" ]]; then
      printf 'error: origin remote not found\n' >&2
      _rss__exit 1
      return $?
    fi

    local owner_repo=""
    if ! owner_repo="$(_rss__origin_owner_repo "$origin_url")"; then
      printf 'error: could not parse owner/repo from origin: %s\n' "$origin_url" >&2
      _rss__exit 1
      return $?
    fi

    local key_id=""
    while :; do
      key_id="$(LC_ALL=C tr -dc '0-9' </dev/urandom | head -c 10)"
      key_name="id_ed25519_repo_${key_id}"

      if _rss__repo_index_has_key "$key_name"; then
        continue
      fi

      key_priv="$ssh_dir/$key_name"
      key_pub="${key_priv}.pub"

      if [[ -e "$key_priv" || -e "$key_pub" ]]; then
        continue
      fi
      break
    done

    if ! ssh-keygen -t ed25519 \
      -C "deploy-key-repo_${key_id}" \
      -f "$key_priv" \
      -N ""; then
      printf 'error: ssh-keygen failed\n' >&2
      _rss__exit 1
      return $?
    fi

    chmod 600 "$key_priv" >/dev/null 2>&1 || true
    chmod 644 "$key_pub" >/dev/null 2>&1 || true

    if ! _rss__ensure_github_host_block_exists; then
      printf 'error: failed to ensure github.com entry in %s\n' "$cfg" >&2
      _rss__exit 1
      return $?
    fi

    if ! _rss__append_identityfile_to_github_block "$key_priv"; then
      printf 'error: failed to add IdentityFile to github.com entry in %s\n' "$cfg" >&2
      _rss__exit 1
      return $?
    fi

    {
      printf '%s %s\n' "$cwd" "$key_name"
    } >>"$repo_index" || {
      printf 'error: could not write to %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    }

    if ! _rss__sort_repo_index_by_path; then
      printf 'error: failed to sort %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    fi

    if ! git remote set-url origin "git@github.com:${owner_repo}.git"; then
      printf 'error: failed to set origin url\n' >&2
      _rss__exit 1
      return $?
    fi

    printf 'Include the following public key in your GitHub repository (Settings → Deploy keys → Add deploy key).\n'
    printf '> PUBLIC KEY: %s\n' "$(cat "$key_pub")"
  else
    key_name="$key_arg"
    key_priv="$ssh_dir/$key_name"
    local key_base=""
    key_base="$(basename -- "$key_name")"

    if [[ ! -f "$key_priv" ]]; then
      printf 'error: ssh key not found: %s\n' "$key_priv" >&2
      _rss__exit 1
      return $?
    fi

    if [[ ! -e "$repo_index" ]]; then
      printf 'error: missing %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    fi
    if _rss__repo_index_has_key "$key_name"; then
      printf 'error: key already present in %s: %s\n' "$repo_index" "$key_name" >&2
      _rss__exit 1
      return $?
    fi
    if _rss__repo_index_has_path "$cwd"; then
      printf 'error: %s already has an entry in %s\n' "$cwd" "$repo_index" >&2
      _rss__exit 1
      return $?
    fi

    local origin_url=""
    origin_url="$(git remote get-url origin 2>/dev/null || true)"
    if [[ -z "$origin_url" ]]; then
      printf 'error: origin remote not found\n' >&2
      _rss__exit 1
      return $?
    fi

    local owner_repo=""
    if ! owner_repo="$(_rss__origin_owner_repo "$origin_url")"; then
      printf 'error: could not parse owner/repo from origin: %s\n' "$origin_url" >&2
      _rss__exit 1
      return $?
    fi

    if ! awk '
      function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s){ return rtrim(ltrim(s)) }
      {
        line=$0
        sub(/\r$/, "", line)
        t=trim(line)
        if (t ~ /^Host[ \t]+github\.com([ \t]|$)/ && t !~ /^#/) { found=1; exit }
      }
      END { exit found?0:1 }
    ' "$cfg"; then
      printf 'error: missing "Host github.com" entry in %s\n' "$cfg" >&2
      _rss__exit 1
      return $?
    fi

    if ! _rss__github_block_has_identityfile_for_key "$key_priv" "$key_base"; then
      printf 'error: "Host github.com" does not reference IdentityFile %s\n' "$key_priv" >&2
      _rss__exit 1
      return $?
    fi

    {
      printf '%s %s\n' "$cwd" "$key_name"
    } >>"$repo_index" || {
      printf 'error: could not write to %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    }

    if ! _rss__sort_repo_index_by_path; then
      printf 'error: failed to sort %s\n' "$repo_index" >&2
      _rss__exit 1
      return $?
    fi

    if ! git remote set-url origin "git@github.com:${owner_repo}.git"; then
      printf 'error: failed to set origin url\n' >&2
      _rss__exit 1
      return $?
    fi

    printf 'Repository registered in ssh config.\n'
  fi

  chmod 600 "$cfg" >/dev/null 2>&1 || true
  chmod 600 "$repo_index" >/dev/null 2>&1 || true

  _rss__exit 0
  return $?
}




























# -----------------------------------------------------------------------------
# create-github-keys
#
# Create a GitHub deploy-key pair and add it as an extra IdentityFile under
# the existing "Host github.com" entry in ~/.ssh/config (creating that entry
# if missing).
#
# Usage:
#   create-github-keys
# -----------------------------------------------------------------------------
create-github-keys() {
  set -uo pipefail

  local ssh_dir="$HOME/.ssh"
  local config_file="$ssh_dir/config"

  mkdir -p "$ssh_dir"
  chmod 700 "$ssh_dir" 2>/dev/null || true

  if ! command -v ssh-keygen >/dev/null 2>&1; then
    echo "error: ssh-keygen not found in PATH" >&2
    return 1
  fi

  # Prefer openssl for clean textual randomness; fall back to python3 if needed.
  if ! command -v openssl >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "error: need either openssl or python3 to generate randomness" >&2
    return 1
  fi

  # If config exists but is not readable/writable, fail early.
  if [[ -e "$config_file" && ! -r "$config_file" ]]; then
    echo "error: ~/.ssh/config exists but is not readable" >&2
    ls -l "$config_file" >&2 || true
    return 1
  fi
  if [[ -e "$config_file" && ! -w "$config_file" ]]; then
    echo "error: ~/.ssh/config exists but is not writable" >&2
    ls -l "$config_file" >&2 || true
    return 1
  fi

  _create_ssh_host_rand_digits_10() {
    local out=""
    if command -v openssl >/dev/null 2>&1; then
      while :; do
        out="$(openssl rand -hex 16 | tr -cd '0-9')"
        if [[ ${#out} -ge 10 ]]; then
          printf '%s' "${out:0:10}"
          return 0
        fi
      done
    else
      python3 - <<'PY'
import secrets, string
alphabet = string.digits
print("".join(secrets.choice(alphabet) for _ in range(10)))
PY
    fi
  }

  _squeeze_blank_lines_in_file() {
    local file_path="$1"
    local tmp_path
    tmp_path="$(mktemp)"
    awk '
      BEGIN { blank = 0 }
      /^[[:space:]]*$/ {
        if (!blank) { print ""; blank = 1 }
        next
      }
      { print; blank = 0 }
    ' "$file_path" >"$tmp_path"
    cat "$tmp_path" >"$file_path"
    rm -f "$tmp_path"
  }

  _ensure_github_host_has_identityfile() {
    local cfg_path="$1"
    local new_key_path="$2"
    local tmp_path
    tmp_path="$(mktemp)"

    awk -v keypath="$new_key_path" '
      function flush_block() {
        if (in_block) {
          idx = ++n
          keys[idx] = cur_key
          blocks[idx] = cur_block
          cur_block = ""
          cur_key = ""
          in_block = 0
        }
      }

      function add_identityfile_to_block(block,    i, line, out, saw_user, last_idf, inserted, already) {
        split(block, L, "\n")
        out = ""
        saw_user = 0
        last_idf = 0
        inserted = 0
        already = 0

        for (i = 1; i <= length(L); i++) {
          line = L[i]
          if (line == "" && i == length(L)) continue

          if (line ~ /^[[:space:]]*IdentityFile[[:space:]]+/) {
            last_idf = i
            if (line ~ ("(^|[[:space:]])" keypath "([[:space:]]|$)")) {
              already = 1
            }
          }
          if (line ~ /^[[:space:]]*User[[:space:]]+/) {
            saw_user = 1
          }
        }

        if (already) {
          return block
        }

        # Insert after last IdentityFile, else after User, else after Host line.
        for (i = 1; i <= length(L); i++) {
          line = L[i]
          if (line == "" && i == length(L)) continue

          out = out line "\n"

          if (!inserted) {
            if (last_idf > 0 && i == last_idf) {
              out = out "  IdentityFile " keypath "\n"
              inserted = 1
            } else if (last_idf == 0 && saw_user && line ~ /^[[:space:]]*User[[:space:]]+/) {
              out = out "  IdentityFile " keypath "\n"
              inserted = 1
            } else if (last_idf == 0 && !saw_user && line ~ /^[[:space:]]*Host[[:space:]]+/) {
              out = out "  IdentityFile " keypath "\n"
              inserted = 1
            }
          }
        }

        return out
      }

      BEGIN {
        preamble = ""
        cur_block = ""
        cur_key = ""
        in_block = 0
        saw_host = 0
        n = 0
      }

      $0 ~ /^[[:space:]]*Host[[:space:]]+/ {
        saw_host = 1
        flush_block()

        in_block = 1
        cur_block = $0 "\n"

        line = $0
        sub(/^[[:space:]]*Host[[:space:]]+/, "", line)
        split(line, a, /[[:space:]]+/)
        cur_key = a[1]
        next
      }

      {
        if (!saw_host) {
          preamble = preamble $0 "\n"
        } else if (in_block) {
          cur_block = cur_block $0 "\n"
        } else {
          preamble = preamble $0 "\n"
        }
      }

      END {
        flush_block()

        github_found = 0
        for (i = 1; i <= n; i++) {
          if (keys[i] == "github.com") {
            blocks[i] = add_identityfile_to_block(blocks[i])
            github_found = 1
          }
        }

        printf "%s", preamble

        for (i = 1; i <= n; i++) {
          gsub(/\n+$/, "\n", blocks[i])
          printf "%s", blocks[i]
          if (i < n) printf "\n"
        }

        if (!github_found) {
          if (preamble != "" || n > 0) printf "\n"
          printf "Host github.com\n"
          printf "  HostName github.com\n"
          printf "  User git\n"
          printf "  IdentityFile %s\n", keypath
          printf "  IdentitiesOnly yes\n"
          printf "  StrictHostKeyChecking yes\n"
        }
      }
    ' "$cfg_path" >"$tmp_path"

    cat "$tmp_path" >"$cfg_path"
    rm -f "$tmp_path"
  }

  # --- generate unique key name + files --------------------------------------
  local digits key_path pub_path comment

  while :; do
    digits="$(_create_ssh_host_rand_digits_10)"
    key_path="$ssh_dir/id_ed25519_${digits}"
    pub_path="${key_path}.pub"

    # Avoid collisions with existing files
    if [[ -e "$key_path" || -e "$pub_path" ]]; then
      continue
    fi

    # Avoid collisions with existing ~/.ssh/config references
    if [[ -f "$config_file" ]]; then
      if grep -Eq "^[[:space:]]*IdentityFile[[:space:]]+.*id_ed25519_${digits}([[:space:]]+|$)" \
        "$config_file" 2>/dev/null; then
        continue
      fi
      if grep -Eq "deploy-key-repo_${digits}([[:space:]]+|$)" "$config_file" 2>/dev/null; then
        continue
      fi
    fi

    break
  done

  comment="deploy-key-repo_${digits}"

  ssh-keygen -t ed25519 -C "$comment" -f "$key_path" -N "" >/dev/null

  chmod 600 "$key_path"
  chmod 644 "$pub_path"

  # --- ensure config exists + add IdentityFile under Host github.com ----------
  if [[ ! -f "$config_file" ]]; then
    : >"$config_file"
  fi
  chmod 600 "$config_file"

  _ensure_github_host_has_identityfile "$config_file" "$key_path"
  _squeeze_blank_lines_in_file "$config_file"
  chmod 600 "$config_file"

  # --- print public key + next steps ----------------------------------------
  local pub_key
  pub_key="$(cat "$pub_path")"

  echo "Include the following public key in your GitHub repository (Settings → Deploy keys → Add deploy key):"
  echo "> PUBLIC KEY: $pub_key"
  echo "Then clone the repository, change directory to its root and run:"
  echo "> register-github-repo --key=id_ed25519_${digits}"
}





















# -----------------------------------------------------------------------------
# prune-github-credentials
#
# Same behavior as before, but with improved messages:
# - If any key is removed, print:
#     Pruned ~/.ssh/github-repo-index and cleaned Host github.com IdentityFile keys.
#     - <key1>
#     - <key2>
# - If no key is removed, print:
#     Nothing to prune (github-repo-index has no invalid repo paths).
# -----------------------------------------------------------------------------
prune-github-credentials() {
  set -uo pipefail

  local ssh_dir="$HOME/.ssh"
  local config_file="$ssh_dir/config"
  local repo_index="$ssh_dir/github-repo-index"

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

  local tmp_repo tmp_cfg invalid_keys referenced_keys gh_keys gh_remove_keys
  local removed_keys
  tmp_repo="$(mktemp)"
  tmp_cfg="$(mktemp)"
  invalid_keys="$(mktemp)"
  referenced_keys="$(mktemp)"
  gh_keys="$(mktemp)"
  gh_remove_keys="$(mktemp)"
  removed_keys="$(mktemp)"

  _prune_rm_keypair_best_effort() {
    local key_path="$1"
    if [[ -n "$key_path" && -f "$key_path" ]]; then
      rm -f -- "$key_path" || true
    fi
    if [[ -n "$key_path" && -f "${key_path}.pub" ]]; then
      rm -f -- "${key_path}.pub" || true
    fi
  }

  _prune_compress_blank_lines_file() {
    local path="$1"
    local tmp
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
  }

  _prune_sort_uniq_file() {
    local path="$1"
    local tmp
    tmp="$(mktemp)"
    awk 'NF { print }' "$path" | sort -u >"$tmp" 2>/dev/null || true
    cat "$tmp" >"$path"
    rm -f "$tmp"
  }

  # -----------------------------------------
  # 1) Prune github-repo-index invalid paths -> invalid_keys
  # 2) Build referenced_keys from remaining lines
  # -----------------------------------------
  awk -v ssh_dir="$ssh_dir" -v out_invalid="$invalid_keys" -v out_ref="$referenced_keys" '
    function trim(s) { sub(/^[ \t\r\n]+/, "", s); sub(/[ \t\r\n]+$/, "", s); return s }
    /^[[:space:]]*$/ { next }
    /^[[:space:]]*#/ { next }
    {
      repo = $1
      key  = $2
      repo = trim(repo)
      key  = trim(key)
      if (repo == "" || key == "") {
        next
      }

      if (substr(repo, 1, 1) != "/") {
        print ssh_dir "/" key >> out_invalid
        next
      }

      cmd = "[ -d \"" repo "\" ]"
      rc = system(cmd)
      if (rc != 0) {
        print ssh_dir "/" key >> out_invalid
        next
      }

      print repo " " key
      print ssh_dir "/" key >> out_ref
    }
  ' "$repo_index" >"$tmp_repo"

  _prune_sort_uniq_file "$invalid_keys"
  _prune_sort_uniq_file "$referenced_keys"

  cat "$tmp_repo" >"$repo_index"
  rm -f "$tmp_repo"

  # Delete keys tied to invalid repo paths (and record them).
  if [[ -s "$invalid_keys" ]]; then
    while IFS= read -r key_path || [[ -n "${key_path:-}" ]]; do
      [[ -n "${key_path:-}" ]] || continue
      printf '%s\n' "$key_path" >>"$removed_keys"
      _prune_rm_keypair_best_effort "$key_path"
    done <"$invalid_keys"
  fi

  # -----------------------------------------
  # Collect IdentityFile keys from Host github.com (expanded paths)
  # -----------------------------------------
  awk '
    BEGIN { in_github = 0 }

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

    /^[[:space:]]*Host[[:space:]]+/ {
      line = $0
      sub(/^[[:space:]]*Host[[:space:]]+/, "", line)
      split(line, a, /[[:space:]]+/)
      alias = a[1]
      in_github = (alias == "github.com") ? 1 : 0
      next
    }

    {
      if (in_github && $0 ~ /^[[:space:]]*IdentityFile[[:space:]]+/) {
        line = $0
        sub(/^[[:space:]]*IdentityFile[[:space:]]+/, "", line)
        p = expand_home(line)
        if (p != "") print p
      }
    }
  ' "$config_file" >"$gh_keys"
  _prune_sort_uniq_file "$gh_keys"

  # -----------------------------------------
  # Determine which github.com keys to remove:
  # - keys in invalid_keys
  # - keys not present in referenced_keys (i.e., unreferenced)
  # -----------------------------------------
  : >"$gh_remove_keys"
  if [[ -s "$invalid_keys" ]]; then
    cat "$invalid_keys" >>"$gh_remove_keys"
  fi

  awk -v refs="$referenced_keys" -v ssh_dir="$ssh_dir" '
    BEGIN {
      while ((getline r < refs) > 0) {
        gsub(/^[ \t]+|[ \t]+$/, "", r)
        if (r != "") ref[r] = 1
      }
      close(refs)
    }
    {
      k = $0
      if (k ~ ("^" ssh_dir "/id_ed25519_")) {
        if (!(k in ref)) {
          print k
        }
      }
    }
  ' "$gh_keys" >>"$gh_remove_keys"

  _prune_sort_uniq_file "$gh_remove_keys"

  # -----------------------------------------
  # Remove those IdentityFile lines from Host github.com in config
  # -----------------------------------------
  awk -v rmfile="$gh_remove_keys" '
    BEGIN {
      in_github = 0
      while ((getline k < rmfile) > 0) {
        gsub(/^[ \t]+|[ \t]+$/, "", k)
        if (k != "") rm[k] = 1
      }
      close(rmfile)
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

    /^[[:space:]]*Host[[:space:]]+/ {
      line = $0
      sub(/^[[:space:]]*Host[[:space:]]+/, "", line)
      split(line, a, /[[:space:]]+/)
      alias = a[1]
      in_github = (alias == "github.com") ? 1 : 0
      print
      next
    }

    {
      if (in_github && $0 ~ /^[[:space:]]*IdentityFile[[:space:]]+/) {
        line = $0
        sub(/^[[:space:]]*IdentityFile[[:space:]]+/, "", line)
        p = expand_home(line)
        if (p in rm) {
          next
        }
      }
      print
    }
  ' "$config_file" >"$tmp_cfg"

  cat "$tmp_cfg" >"$config_file"
  rm -f "$tmp_cfg"

  _prune_compress_blank_lines_file "$config_file"
  chmod 600 "$config_file" 2>/dev/null || true

  # Delete github.com keys that are now unreferenced/invalid (and record them).
  if [[ -s "$gh_remove_keys" ]]; then
    while IFS= read -r key_path || [[ -n "${key_path:-}" ]]; do
      [[ -n "${key_path:-}" ]] || continue
      printf '%s\n' "$key_path" >>"$removed_keys"
      _prune_rm_keypair_best_effort "$key_path"
    done <"$gh_remove_keys"
  fi

  _prune_sort_uniq_file "$removed_keys"

  if [[ -s "$removed_keys" ]]; then
    echo "Pruned ~/.ssh/github-repo-index and cleaned Host github.com IdentityFile keys."
    while IFS= read -r key_path || [[ -n "${key_path:-}" ]]; do
      [[ -n "${key_path:-}" ]] || continue
      printf -- "- %s\n" "$key_path"
    done <"$removed_keys"
  else
    echo "Nothing to prune (github-repo-index has no invalid repo paths)."
  fi

  rm -f "$invalid_keys" "$referenced_keys" "$gh_keys" "$gh_remove_keys" "$removed_keys"
}
