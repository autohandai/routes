#!/usr/bin/env bash
# Examples of the `route` command with different policies.
# Usage: ./route.sh

set -euo pipefail

CONFIG="examples/router.yaml"
PROMPT="Design a production event sourcing system"

echo "Route with default balanced policy:"
cargo run --quiet -- --config "$CONFIG" route "$PROMPT"

echo ""
echo "Route with highest-quality policy:"
cargo run --quiet -- --config "$CONFIG" route "$PROMPT" --policy highest-quality

echo ""
echo "Route with fastest policy:"
cargo run --quiet -- --config "$CONFIG" route "$PROMPT" --policy fastest

echo ""
echo "Route with local-first policy:"
cargo run --quiet -- --config "$CONFIG" route "$PROMPT" --policy local-first

echo ""
echo "Route with privacy-first policy:"
cargo run --quiet -- --config "$CONFIG" route "$PROMPT" --policy privacy-first
