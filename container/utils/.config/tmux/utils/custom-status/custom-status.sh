#!/usr/bin/env bash

# Helper: get a tmux option or fallback to default
get_tmux_option() {
  local option=$1
  local default_value="$2"
  local option_value
  option_value=$(tmux show-options -gqv "$option")
  if [ -n "$option_value" ]; then
    echo "$option_value"
    return
  fi
  echo "$default_value"
}

# Helper: normalize boolean-ish values
is_true() {
  case "$1" in
    1|true|on|yes|True|TRUE|ON|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

# Space-right-pad to width
pad_to_width() {
  local s="$1" w="$2" n=${#1}
  if (( n < w )); then
    printf "%s%*s" "$s" $((w - n)) ""
  else
    printf "%s" "$s"
  fi
}

# --- Defaults/Options ---

bg=$(get_tmux_option "@minimal-tmux-bg" "#698DDA")
fg=$(get_tmux_option "@minimal-tmux-fg" "#000000")

use_arrow=$(get_tmux_option "@minimal-tmux-use-arrow" "false")
if is_true "$use_arrow"; then
  larrow=$(get_tmux_option "@minimal-tmux-left-arrow" "")
  rarrow=$(get_tmux_option "@minimal-tmux-right-arrow" "")
else
  larrow=""
  rarrow=""
fi

status=$(get_tmux_option "@minimal-tmux-status" "bottom")
justify=$(get_tmux_option "@minimal-tmux-justify" "centre")

indicator_state=$(get_tmux_option "@minimal-tmux-indicator" "true")
indicator_str=$(get_tmux_option "@minimal-tmux-indicator-str" "tmux")

right_state=$(get_tmux_option "@minimal-tmux-right" "true")
left_state=$(get_tmux_option "@minimal-tmux-left" "true")

# --- Fixed visible width for symmetry ---
CONTENT_W=12
time_fmt=" %a %Y/%m/%d %H:%M"

# Precompute a padded indicator of exactly CONTENT_W chars
padded_indicator="$(pad_to_width " ${indicator_str} " ${CONTENT_W})"

# --- LEFT: clock normally; highlighted “tmux” while prefix (fixed width) ---
if is_true "$left_state"; then
  status_left="#{?client_prefix,#[bg=${bg}]#[fg=${fg}]#[bold]${padded_indicator}#[nobold]#[fg=default]#[bg=default],${time_fmt}}"
  if [ -n "$rarrow" ]; then
    status_left="${status_left}#[fg=${bg}]#[bg=default]${rarrow}"
  fi
else
  status_left=""
fi

# --- RIGHT: session name, fixed width via printf, never highlighted ---
if is_true "$right_state"; then
  # Left arrow to mirror the left side’s right arrow (keeps total widths equal)
  status_right=""
  if [ -n "$larrow" ]; then
    status_right="${status_right}${larrow}"
  fi

  # Use printf '%20.20s' to left-pad and truncate to CONTENT_W.
  # Surround session with spaces to match clock spacing.
  right_fixed='#(sh -lc '\''s=$(tmux display -p "#S"); printf "%'"${CONTENT_W}"'.'"${CONTENT_W}"'s" " ${s} "'\'' )'

  # No styling on the right → never highlighted
  status_right="${status_right}${right_fixed}"
else
  status_right=""
fi

# Allow extras (append after our fixed blocks)
status_right_extra="$status_right$(get_tmux_option '@minimal-tmux-status-right-extra' '')"
status_left_extra="$status_left$(get_tmux_option '@minimal-tmux-status-left-extra' '')"

# Window formats
window_status_format=$(get_tmux_option "@minimal-tmux-window-status-format" " #I:#W ")
expanded_icon=$(get_tmux_option "@minimal-tmux-expanded-icon" "󰊓 ")
show_expanded_icon_for_all_tabs=$(get_tmux_option "@minimal-tmux-show-expanded-icon-for-all-tabs" "false")

# --- Apply ---
tmux set-option -g status on
tmux set-option -g status-position "$status"
tmux set-option -g status-style "bg=default,fg=default"
tmux set-option -g status-justify "$justify"

# Max lengths high enough not to truncate our fixed blocks
tmux set-option -g status-left-length 80
tmux set-option -g status-right-length 80

tmux set-option -g status-left "$status_left_extra"
tmux set-option -g status-right "$status_right_extra"

tmux set-option -g window-status-format "$window_status_format"
if is_true "$show_expanded_icon_for_all_tabs"; then
  tmux set-option -g window-status-format " ${window_status_format}#{?window_zoomed_flag,${expanded_icon},}"
fi

tmux set-option -g window-status-current-format \
  "#[fg=${bg}]${larrow}#[bg=${bg}]#[fg=${fg}]${window_status_format}#{?window_zoomed_flag,${expanded_icon},}#[fg=${bg}]#[bg=default]${rarrow}"
