#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$repository_root/scripts/audit-dependencies.sh"
fixtures="$repository_root/security/test-fixtures"

CARGO_BIN=true ADVISORY_TODAY=2026-07-13 \
  ADVISORY_EXCEPTIONS_FILE="$fixtures/future.txt" "$runner"

if CARGO_BIN=true ADVISORY_TODAY=2026-07-13 \
  ADVISORY_EXCEPTIONS_FILE="$fixtures/expired.txt" "$runner"; then
  echo "expired advisory exception unexpectedly passed" >&2
  exit 1
fi

if CARGO_BIN=true ADVISORY_TODAY=2026-07-13 \
  ADVISORY_EXCEPTIONS_FILE="$fixtures/malformed.txt" "$runner"; then
  echo "malformed advisory exception unexpectedly passed" >&2
  exit 1
fi

if CARGO_BIN=true ADVISORY_TODAY=2026-07-13 \
  ADVISORY_EXCEPTIONS_FILE="$fixtures/invalid-date.txt" "$runner"; then
  echo "invalid advisory expiry unexpectedly passed" >&2
  exit 1
fi
