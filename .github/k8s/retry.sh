#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 [--attempts <positive-integer>] [--sleep <non-negative-seconds>] <step-name> <command> [arguments...]" >&2
  exit 2
}

max_attempts=""
sleep_seconds="5"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --attempts)
      if [ "$#" -lt 2 ] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        usage
      fi
      max_attempts="$2"
      shift 2
      ;;
    --sleep)
      if [ "$#" -lt 2 ] || ! [[ "$2" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        usage
      fi
      sleep_seconds="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -lt 2 ]; then
  usage
fi

step_name="$1"
shift

child_pid=""
shutdown() {
  trap - TERM INT
  if [ -n "$child_pid" ]; then
    # Each command runs in its own session, so this also stops its descendants.
    kill -TERM -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}
trap shutdown TERM INT

attempt=0
while true; do
  attempt=$((attempt + 1))
  setsid "$@" &
  child_pid=$!
  if wait "$child_pid"; then
    child_pid=""
    exit 0
  else
    exit_code=$?
    child_pid=""
  fi
  if [ -n "$max_attempts" ] && [ "$attempt" -ge "$max_attempts" ]; then
    exit "$exit_code"
  fi

  retry_message="${step_name} is being retried (attempt ${attempt}"
  if [ -n "$max_attempts" ]; then
    retry_message="${retry_message}/${max_attempts}"
  fi
  retry_message="${retry_message})."
  if [ -n "${GITHUB_CHECK_RUN_ID:-}" ] && \
     ! bash -euo pipefail .github/scripts/check-run.sh \
       in_progress "" "Retrying ${step_name}" "$retry_message"; then
    echo "Could not update the check run for ${retry_message}" >&2
  fi

  if [ -n "$max_attempts" ]; then
    echo "${step_name} failed (attempt ${attempt}/${max_attempts}); retrying in ${sleep_seconds} seconds" >&2
  else
    echo "${step_name} failed (attempt ${attempt}); retrying in ${sleep_seconds} seconds" >&2
  fi
  sleep "$sleep_seconds" &
  child_pid=$!
  wait "$child_pid" || true
  child_pid=""
done
