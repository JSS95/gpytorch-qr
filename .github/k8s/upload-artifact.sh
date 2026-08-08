required_variables=(
  GH_APP_ID
  GH_APP_PRIVATE_KEY
  GITHUB_REPOSITORY
  GITHUB_RELEASE_ID
  GITHUB_RELEASE_TAG_NAME
)

for variable_name in "${required_variables[@]}"; do
  if [ -z "${!variable_name:-}" ]; then
    echo "Missing required environment variable: $variable_name" >&2
    exit 1
  fi
done

asset_name="examples-${GITHUB_RELEASE_TAG_NAME}.tar.gz"
archive_file="$(mktemp --suffix=.tar.gz)"
private_key_file="$(mktemp)"
trap 'rm -f "$archive_file" "$private_key_file"' EXIT

tar -C examples -czf "$archive_file" .
printf '%s' "$GH_APP_PRIVATE_KEY" > "$private_key_file"

issued_at="$(($(date +%s) - 60))"
expires_at="$((issued_at + 540))"
jwt_header="$(printf '%s' '{"alg":"RS256","typ":"JWT"}' | openssl base64 -A | tr '+/' '-_' | tr -d '=')"
jwt_payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "$issued_at" "$expires_at" "$GH_APP_ID" | openssl base64 -A | tr '+/' '-_' | tr -d '=')"
jwt_signature="$(printf '%s' "$jwt_header.$jwt_payload" | openssl dgst -sha256 -sign "$private_key_file" | openssl base64 -A | tr '+/' '-_' | tr -d '=')"
app_jwt="$jwt_header.$jwt_payload.$jwt_signature"

installation_id="$(curl --fail --silent --show-error \
  --header "Authorization: Bearer $app_jwt" \
  --header 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/installation" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["id"])')"
installation_token="$(curl --fail --silent --show-error --request POST \
  --header "Authorization: Bearer $app_jwt" \
  --header 'Accept: application/vnd.github+json' \
  "https://api.github.com/app/installations/${installation_id}/access_tokens" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["token"])')"

existing_asset_id="$(curl --fail --silent --show-error \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/${GITHUB_RELEASE_ID}/assets?per_page=100" | \
  python -c 'import json, sys; asset_name = sys.argv[1]; print(next((asset["id"] for asset in json.load(sys.stdin) if asset["name"] == asset_name), ""))' \
  "$asset_name")"

if [ -n "$existing_asset_id" ]; then
  curl --fail --silent --show-error --request DELETE \
    --header "Authorization: Bearer $installation_token" \
    --header 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/assets/${existing_asset_id}"
fi

encoded_asset_name="$(python -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$asset_name")"
curl --fail --silent --show-error --request POST \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  --header 'Content-Type: application/gzip' \
  --data-binary "@$archive_file" \
  "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/${GITHUB_RELEASE_ID}/assets?name=${encoded_asset_name}"
