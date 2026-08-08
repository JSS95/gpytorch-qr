if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <status> <conclusion>" >&2
  exit 2
fi

check_status="$1"
check_conclusion="$2"

source .github/k8s/app-token.sh
check_run_data="$(python -c 'import json, sys; print(json.dumps({"status": sys.argv[1], "conclusion": sys.argv[2]}))' "$check_status" "$check_conclusion")"
curl --fail --silent --show-error --request PATCH \
--header "Authorization: Bearer $installation_token" \
--header 'Accept: application/vnd.github+json' \
--header 'Content-Type: application/json' \
--data "$check_run_data" \
"https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${GITHUB_CHECK_RUN_ID}"
