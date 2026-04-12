# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# # If not running interactively, don't do anything
# case $- in
#     *i*) ;;
#       *) return;;
# esac

# # Disable strict-mode in interactive shells (prevents random prompt-hook exits).
# case "$-" in
#   *i*) set +e; set +u ;;
# esac

# Disable strict-mode in interactive shells (prevents random prompt-hook exits).
case "$-" in
  *i*) set +e; set +u ;;
    *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        # We have color support; assume it's compliant with Ecma-48
        # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
        # a case would tend to support setf rather than setaf.)
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    # PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '  # green
    PS1='${debian_chroot:+($debian_chroot)}\[\033[38;2;255;204;117m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ ' # orange
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# remove unused dependencies from the uv cache
command -v uncache >/dev/null 2>&1 && uncache >/dev/null  # silent success & visible failure
if command -v uv >/dev/null 2>&1; then
  mkdir -p -- "${UV_CACHE_DIR:-$HOME/.cache/uv}" 2>/dev/null || true
  uv cache prune >/dev/null 2>&1 || echo "uv cache prune failed" >&2
fi

### CUSTOM

[[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && export PATH="$HOME/.local/bin:$PATH"  # uv bin
[ -f "$HOME/.config/uv/define-functions.sh" ] && chmod +x "$HOME/.config/uv/define-functions.sh" && ( set -e; "$HOME/.config/uv/define-functions.sh" ) && source "$HOME/.config/uv/define-functions.sh" >&2

chmod +x "$HOME/.config/uv/uv-cache-dependency-clean.sh"

# aliases
alias up="chmod +x up.sh && ./up.sh"
alias attach="chmod +x attach.sh && ./attach.sh"
alias down="chmod +x down.sh && ./down.sh"
alias py="python"
alias cls="clear"
alias storage="du -hc --max-depth=0 /home/${USER}/.local/share/uv/python/ /home/${USER}/.local/share/uv/tools/ /home/${USER}/.cache/ /mnt/workdata/uv_cache/ /mnt/workdata/data/"

exit() {
  if [[ "${PERSISTENT_UV_CACHE:-true}" != "true" ]]; then
    DEBUG=0 "$HOME/.config/uv/uv-cache-dependency-clean.sh" >/dev/null 2>&1 || true
    find /mnt/workdata/uv_cache -mindepth 1 -delete >/dev/null 2>&1 || true
  fi
  command -v uncache >/dev/null 2>&1 && uncache >/dev/null # silent success & visible failure
  builtin exit
}

# uncomment this to avoid displaying README.md at start up
# DISPLAY_INFO_AT_STARTUP=false
case "${DISPLAY_INFO_AT_STARTUP:-false}" in
  true|1|yes|on)
    if [[ "${PERSISTENT_UV_CACHE:-true}" != "true" ]]; then
      DEBUG=0 "$HOME/.config/uv/uv-cache-dependency-clean.sh" >/dev/null 2>&1 || true
      find /mnt/workdata/uv_cache -mindepth 1 -delete >/dev/null 2>&1 || true
    fi
    echo ""
    echo "INSTALLED PYTHON INTERPRETERS:"
    command -v uncache >/dev/null 2>&1 && { uncache >/dev/null & disown; }
    interpreters || true
    echo ""
    echo "EFFECTIVE STORAGE:"
    storage
    echo ""
    echo "INSTALLED UV TOOLS:"
    uv tool list
    echo ""
    ;;
esac
DISPLAY_INFO_AT_STARTUP=false


# Define ssh functions
source "${HOME}/.config/ssh/define-ssh-functions.sh"
prune_ssh >/dev/null 2>&1 || true



# ---- cache cleanup activity ping (global across all terminals) ----
# Requires env:
#   CLEANUP_TIMER_FILE=/run/control/var/cache-cleanup-time
#   CACHE_CLEANUP_TIME=<seconds>

__cache_cleanup__ping() {
  # Best-effort: never break interactive shell if something fails.
  local state_file lock_file timeout_s
  state_file="${CLEANUP_TIMER_FILE:-}"
  timeout_s="${CACHE_CLEANUP_TIME:-}"
  [[ -n "${state_file}" ]] || return 0
  [[ "${timeout_s}" =~ ^[0-9]+$ ]] || return 0

  lock_file="${state_file}.lock"
  mkdir -p -- "$(dirname -- "${state_file}")" 2>/dev/null || true

  if command -v flock >/dev/null 2>&1; then
    flock -x "${lock_file}" -c "printf '%s\n' '${timeout_s}' >'${state_file}'" \
      2>/dev/null || true
  else
    printf '%s\n' "${timeout_s}" >"${state_file}" 2>/dev/null || true
  fi
  return 0
}

__cache_cleanup__install_prompt_command() {
  local pc
  pc="${PROMPT_COMMAND-}"

  if [[ -z "${pc}" ]]; then
    PROMPT_COMMAND="__cache_cleanup__ping"
    return 0
  fi

  case ";${pc};" in
    *";__cache_cleanup__ping;"*) : ;;
    *)
      PROMPT_COMMAND="${pc} ; __cache_cleanup__ping"
      ;;
  esac
}


__cache_cleanup__install_prompt_command
# ---- end cache cleanup hook ----







# Ensure errexit is disabled in interactive shells to prevent the session from
# exiting when a command returns a non-zero status
set +o errexit