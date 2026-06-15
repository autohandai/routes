# Routes examples

This folder contains runnable examples for configuring and using Routes.

## Config examples

- [`router.yaml`](router.yaml) — full development config with local and remote providers.
- [`router.local.yaml`](router.local.yaml) — local-first config for offline or self-hosted setups.
- [`router.privacy.yaml`](router.privacy.yaml) — privacy-first config that avoids remote providers.

## CLI usage examples

- [`classify.sh`](classify.sh) — classify prompts with the router.
- [`route.sh`](route.sh) — ask the router which model it would choose under different policies.
- [`multimodel.sh`](multimodel.sh) — route a request with explicit capability requirements.
- [`chat_completion.sh`](chat_completion.sh) — proxy a chat completion through Routes.

## Coding agent config examples

The [`client-config/`](client-config/) folder contains sample configurations for pointing popular coding agents at Routes:

- [`autohand-code-cli.json`](client-config/autohand-code-cli.json)
- [`codex.env`](client-config/codex.env)
- [`pi-models.json`](client-config/pi-models.json)
- [`aider.sh`](client-config/aider.sh)
- [`cursor.md`](client-config/cursor.md)
- [`hermes.yaml`](client-config/hermes.yaml)
- [`openclaw.json`](client-config/openclaw.json)

## Eval examples

- [`eval.jsonl`](eval.jsonl) — small development eval set.
- [`eval.production.jsonl`](eval.production.jsonl) — larger production-oriented eval gate.

Run any config through validation before using it:

```bash
cargo run -- --config examples/router.local.yaml validate
```
