if [ -z "${GITHUB_RELEASE_TAG_NAME:-}" ]; then
  echo "Missing required environment variable: GITHUB_RELEASE_TAG_NAME" >&2
  exit 1
fi
if [ -z "${PUBLISH_DOC_CHECK_RUN_ID:-}" ]; then
  echo "Missing required environment variable: PUBLISH_DOC_CHECK_RUN_ID" >&2
  exit 1
fi

source .github/scripts/app-token.sh

dispatch_payload="$(python -c '
import json
import sys

print(json.dumps({
    "event_type": "build-release-docs",
    "client_payload": {
        "release_tag": sys.argv[1],
        "check_run_id": sys.argv[2],
    },
}))
' "$GITHUB_RELEASE_TAG_NAME" "$PUBLISH_DOC_CHECK_RUN_ID")"

curl --fail --silent --show-error --request POST \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  --header 'Content-Type: application/json' \
  --header 'X-GitHub-Api-Version: 2022-11-28' \
  --data "$dispatch_payload" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/dispatches"
