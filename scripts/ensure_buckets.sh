#!/usr/bin/env bash
# Idempotently create the local Supabase Storage buckets the app needs
# (patient-photos = avatars + face-enrollment selfies; incident-clips =
# event videos). Safe to re-run — an existing bucket returns 409, which
# we treat as success. Called automatically by `make db-reset` so the
# buckets always exist after a database reset.
set -euo pipefail

# Pull the local API URL + service-role key from the running Supabase.
eval "$(supabase status -o env | grep -E '^(API_URL|SERVICE_ROLE_KEY)=')"

if [[ -z "${API_URL:-}" || -z "${SERVICE_ROLE_KEY:-}" ]]; then
  echo "ensure_buckets: couldn't read API_URL / SERVICE_ROLE_KEY (is supabase running?)" >&2
  exit 0
fi

for B in patient-photos incident-clips; do
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$API_URL/storage/v1/bucket" \
    -H "Authorization: Bearer $SERVICE_ROLE_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"id\":\"$B\",\"name\":\"$B\",\"public\":false}")
  case "$code" in
    200|201) echo "  bucket $B: created" ;;
    409)     echo "  bucket $B: already exists" ;;
    *)       echo "  bucket $B: unexpected HTTP $code" >&2 ;;
  esac
done
