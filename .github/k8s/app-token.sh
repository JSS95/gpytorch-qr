required_variables=(
  GH_APP_ID
  GH_APP_PRIVATE_KEY
  GITHUB_REPOSITORY
)

for variable_name in "${required_variables[@]}"; do
  if [ -z "${!variable_name:-}" ]; then
    echo "Missing required environment variable: $variable_name" >&2
    return 1
  fi
done

app_private_key_file="$(mktemp)"
printf '%s' "$GH_APP_PRIVATE_KEY" > "$app_private_key_file"

issued_at="$(($(date +%s) - 60))"
expires_at="$((issued_at + 540))"
jwt_header="$(printf '%s' '{"alg":"RS256","typ":"JWT"}' | openssl base64 -A | tr '+/' '-_' | tr -d '=')"
jwt_payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "$issued_at" "$expires_at" "$GH_APP_ID" | openssl base64 -A | tr '+/' '-_' | tr -d '=')"
jwt_signature="$(printf '%s' "$jwt_header.$jwt_payload" | openssl dgst -sha256 -sign "$app_private_key_file" | openssl base64 -A | tr '+/' '-_' | tr -d '=')"
rm -f "$app_private_key_file"
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
