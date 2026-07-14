#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-artifacts/contracts}"
mkdir -p "$output_dir"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT

cargo run --quiet --locked -- --config examples/router.yaml openapi > "$output_dir/openapi.json"
cargo run --quiet --locked -- --config examples/router.yaml config-schema > "$output_dir/config-schema.json"
cargo run --quiet --locked -- --config examples/router.yaml openapi > "$temporary_dir/openapi.json"
cargo run --quiet --locked -- --config examples/router.yaml config-schema > "$temporary_dir/config-schema.json"

cmp "$output_dir/openapi.json" "$temporary_dir/openapi.json"
cmp "$output_dir/config-schema.json" "$temporary_dir/config-schema.json"

jq -e '
  .openapi == "3.1.0"
  and (.paths | length) == 19
  and (.paths["/v1/router/classify"].post != null)
  and (.paths["/v1/router/multimodel"].post != null)
  and (.paths["/v1/chat/completions"].post != null)
  and (.paths["/v1/responses"].post != null)
  and (.components.schemas.RouterError != null)
' "$output_dir/openapi.json" >/dev/null

jq -e '
  ."$schema" == "https://json-schema.org/draft/2020-12/schema"
  and .type == "object"
  and .additionalProperties == false
  and (.required | index("default_model") != null)
  and (.properties.providers != null)
  and (.properties.models != null)
  and (.properties.runtime != null)
  and (."$defs" | length) >= 30
' "$output_dir/config-schema.json" >/dev/null
