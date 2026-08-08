if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <status> [conclusion] [title] [summary]" >&2
  exit 2
fi

check_status="$1"
check_conclusion="${2:-}"
check_title="${3:-}"
check_summary="${4:-$check_title}"

source .github/k8s/app-token.sh
current_check="$(curl --fail --silent --show-error \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${GITHUB_CHECK_RUN_ID}")"
check_run_data="$(python -c '
import datetime
import json
import sys

status, conclusion, title, summary, current_check = sys.argv[1:]
current_output = json.loads(current_check).get("output") or {}
title = title or current_output.get("title") or "k8s"
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
' "$check_status" "$check_conclusion" "$check_title" "$check_summary" "$current_check")"
curl --fail --silent --show-error --request PATCH \
--header "Authorization: Bearer $installation_token" \
--header 'Accept: application/vnd.github+json' \
--header 'Content-Type: application/json' \
--data "$check_run_data" \
"https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${GITHUB_CHECK_RUN_ID}"
