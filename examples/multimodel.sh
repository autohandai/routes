#!/usr/bin/env bash
# Example of the /v1/router/multimodel endpoint.
# Usage: ./multimodel.sh

set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8080}"

curl -s "$ROUTER_URL/v1/router/multimodel" \
  -H 'content-type: application/json' \
  -d '{
    "input": "Build a small web app from this screenshot and explain the architecture",
    "policy": "multimodal_first",
    "required_capabilities": ["vision", "web_apps"]
  }' | jq .
