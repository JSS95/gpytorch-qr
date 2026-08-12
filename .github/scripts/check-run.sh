if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <status> [conclusion] [title] [summary]" >&2
  exit 2
fi

check_status="$1"
check_conclusion="${2:-}"
check_title="${3:-}"
check_summary="${4:-$check_title}"

source .github/scripts/app-token.sh

if [ -z "${GITHUB_CHECK_RUN_ID:-}" ]; then
  if [ -z "${GIT_SHA:-}" ] || [ -z "${GITHUB_CHECK_RUN_NAME:-}" ]; then
    echo "GITHUB_CHECK_RUN_ID or both GIT_SHA and GITHUB_CHECK_RUN_NAME are required" >&2
    exit 1
  fi
  encoded_check_name="$(python -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$GITHUB_CHECK_RUN_NAME")"
  matching_checks="$(curl --fail --silent --show-error \
    --header "Authorization: Bearer $installation_token" \
    --header 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/commits/${GIT_SHA}/check-runs?check_name=${encoded_check_name}&filter=latest&per_page=1")"
  GITHUB_CHECK_RUN_ID="$(python -c '
import json
import sys

checks = json.load(sys.stdin).get("check_runs", [])
if not checks:
    raise SystemExit("No matching check run was found")
print(checks[0]["id"])
' <<EOF
$matching_checks
EOF
)"
fi

current_check="$(curl --fail --silent --show-error \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${GITHUB_CHECK_RUN_ID}")"
check_run_data="$(python -c '
import datetime
import json
import sys

status, conclusion, title, summary, current_check, check_name = sys.argv[1:]
current_output = json.loads(current_check).get("output") or {}
title = title or current_output.get("title") or check_name or "check"
message = summary or title
timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
state = f"status={status}"
if conclusion:
    state += f", conclusion={conclusion}"
entry = f"- {timestamp} — {message} ({state})"
previous_summary = current_output.get("summary", "").rstrip()
history = "\n".join(filter(None, (previous_summary, entry)))

payload = {"status": status}
if conclusion:
    payload["conclusion"] = conclusion
payload["output"] = {"title": title, "summary": history[-60000:]}
print(json.dumps(payload))
' "$check_status" "$check_conclusion" "$check_title" "$check_summary" "$current_check" "${GITHUB_CHECK_RUN_NAME:-}")"
curl --fail --silent --show-error --request PATCH \
  --output /dev/null \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  --header 'Content-Type: application/json' \
  --data "$check_run_data" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${GITHUB_CHECK_RUN_ID}"
