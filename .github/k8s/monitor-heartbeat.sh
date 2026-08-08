heartbeat_dir="${HEARTBEAT_DIR:-/heartbeat}"

finalize_check() {
  bash -euo pipefail .github/k8s/check-run.sh "$@"
  touch "$heartbeat_dir/check-finalized"
}

if [ -f "$heartbeat_dir/check-finalized" ]; then
  exit 0
elif [ -f "$heartbeat_dir/build-cancelled" ]; then
  finalize_check completed cancelled "Job was cancelled"
elif [ -f "$heartbeat_dir/build-failure" ]; then
  finalize_check completed failure "Build container failed"
elif [ -f "$heartbeat_dir/build-success" ]; then
  finalize_check completed success "Build completed successfully"
elif [ -f "$heartbeat_dir/build-started" ]; then
  heartbeat_file="$heartbeat_dir/build-heartbeat"
  heartbeat_source="$heartbeat_dir/build-started"
  if [ -f "$heartbeat_file" ]; then
    heartbeat_source="$heartbeat_file"
  fi

  heartbeat_age="$(($(date +%s) - $(stat -c %Y "$heartbeat_source")))"
  if [ "$heartbeat_age" -gt 90 ]; then
    finalize_check completed failure "Build stopped"
  fi
fi
