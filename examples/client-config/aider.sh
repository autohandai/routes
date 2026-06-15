#!/usr/bin/env bash
# Example Aider invocation routing through Routes.

aider \
  --model openai/router-balanced \
  --openai-api-base http://127.0.0.1:8080/v1 \
  --openai-api-key routes-local-dev-or-bearer-token \
  "$@"
