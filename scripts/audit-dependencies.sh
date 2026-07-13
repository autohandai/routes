#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exceptions_file="${ADVISORY_EXCEPTIONS_FILE:-$repository_root/security/advisory-exceptions.txt}"
today="${ADVISORY_TODAY:-$(date -u +%F)}"
cargo_bin="${CARGO_BIN:-cargo}"
audit_args=(audit --deny warnings)

valid_iso_date() {
  local value="$1"
  local normalized
  if [[ ! "$value" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    return 1
  fi
  if normalized="$(date -u -d "$value" +%F 2>/dev/null)"; then
    [[ "$normalized" == "$value" ]]
    return
  fi
  if normalized="$(date -j -u -f '%Y-%m-%d' "$value" '+%Y-%m-%d' 2>/dev/null)"; then
    [[ "$normalized" == "$value" ]]
    return
  fi
  return 1
}

if ! valid_iso_date "$today"; then
  echo "invalid advisory policy date: $today" >&2
  exit 2
fi

while IFS='|' read -r advisory expires owner reason; do
  if [[ -z "$advisory" || "$advisory" == \#* ]]; then
    continue
  fi
  if [[ ! "$advisory" =~ ^RUSTSEC-[0-9]{4}-[0-9]{4}$ ]]; then
    echo "invalid RustSec advisory id in $exceptions_file: $advisory" >&2
    exit 2
  fi
  if ! valid_iso_date "$expires"; then
    echo "invalid expiry for $advisory: $expires" >&2
    exit 2
  fi
  if [[ -z "$owner" || -z "$reason" ]]; then
    echo "exception $advisory requires an owner and reason" >&2
    exit 2
  fi
  if [[ "$expires" < "$today" || "$expires" == "$today" ]]; then
    echo "exception $advisory expired on $expires" >&2
    exit 1
  fi
  audit_args+=(--ignore "$advisory")
done < "$exceptions_file"

"$cargo_bin" "${audit_args[@]}"
