#!/usr/bin/env bash
# Example of proxying a chat completion through Routes.
# Usage: ./chat_completion.sh

set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8080}"
API_KEY="${ROUTER_API_KEY:-routes-local-dev-or-bearer-token}"

curl -s "$ROUTER_URL/v1/chat/completions" \
  -H "authorization: Bearer $API_KEY" \
  -H 'content-type: application/json' \
  -d '{
    "model": "router-balanced",
    "messages": [
      {"role": "user", "content": "Explain how to design a fault-tolerant background job system"}
    ]
  }' | jq .
