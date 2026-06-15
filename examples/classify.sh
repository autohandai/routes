#!/usr/bin/env bash
# Examples of the `classify` command.
# Usage: ./classify.sh

set -euo pipefail

CONFIG="examples/router.yaml"

echo "Classify a simple coding prompt:"
cargo run --quiet -- --config "$CONFIG" classify "Fix this typo in the README"

echo ""
echo "Classify a multimodal prompt:"
cargo run --quiet -- --config "$CONFIG" classify "Build a small web app from this screenshot"

echo ""
echo "Classify a complex architecture prompt:"
cargo run --quiet -- --config "$CONFIG" classify "Design a production event sourcing system"
