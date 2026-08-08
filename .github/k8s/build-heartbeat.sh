heartbeat_dir="${HEARTBEAT_DIR:-/heartbeat}"
heartbeat_interval="${HEARTBEAT_INTERVAL_SECONDS:-15}"

usage() {
  echo "Usage: $0 start | finish <success|failure|cancelled> [heartbeat_pid]" >&2
  exit 2
}

wait_for_check_finalization() {
  for _ in $(seq 1 60); do
    [ -f "$heartbeat_dir/check-finalized" ] && return 0
    sleep 1
  done
  return 1
}

case "${1:-}" in
  start)
    [ "$#" -eq 1 ] || usage
    touch "$heartbeat_dir/build-started"
    while true; do
      touch "$heartbeat_dir/build-heartbeat"
      sleep "$heartbeat_interval"
    done
    ;;
  finish)
    [ "$#" -eq 2 ] || [ "$#" -eq 3 ] || usage
    case "$2" in
      success|failure|cancelled) ;;
      *) usage ;;
    esac
    if [ "$#" -eq 3 ]; then
      kill "$3" 2>/dev/null || true
    fi
    touch "$heartbeat_dir/build-$2"
    wait_for_check_finalization
    ;;
  *)
    usage
    ;;
esac
