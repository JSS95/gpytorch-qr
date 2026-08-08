if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <status> [conclusion] [title] [summary]" >&2
  exit 2
fi

check_status="$1"
check_conclusion="${2:-}"
check_title="${3:-}"
check_summary="${4:-$check_title}"

source .github/k8s/app-token.sh
check_run_data="$(python -c '
import json
import sys

status, conclusion, title, summary = sys.argv[1:]
payload = {"status": status}
if conclusion:
    payload["conclusion"] = conclusion
if title:
    payload["output"] = {"title": title, "summary": summary}
print(json.dumps(payload))
' "$check_status" "$check_conclusion" "$check_title" "$check_summary")"
curl --fail --silent --show-error --request PATCH \
--header "Authorization: Bearer $installation_token" \
--header 'Accept: application/vnd.github+json' \
--header 'Content-Type: application/json' \
--data "$check_run_data" \
"https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${GITHUB_CHECK_RUN_ID}"
