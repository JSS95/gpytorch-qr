#!/usr/bin/env bash
set -u

max_attempts=""
if [ "${1:-}" = "--attempts" ]; then
  if [ "$#" -lt 4 ] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [--attempts <positive-integer>] <step-name> <command> [arguments...]" >&2
    exit 2
  fi
  max_attempts="$2"
  shift 2
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 [--attempts <positive-integer>] <step-name> <command> [arguments...]" >&2
  exit 2
fi

step_name="$1"
shift

attempt=0
while true; do
  attempt=$((attempt + 1))
  if "$@"; then
    exit 0
  else
    exit_code=$?
  fi
  if [ -n "$max_attempts" ] && [ "$attempt" -ge "$max_attempts" ]; then
    exit "$exit_code"
  fi

  if [ -n "$max_attempts" ]; then
    echo "${step_name} failed (attempt ${attempt}/${max_attempts}); retrying in 5 seconds" >&2
  else
    echo "${step_name} failed (attempt ${attempt}); retrying in 5 seconds" >&2
  fi
  sleep 5
done
