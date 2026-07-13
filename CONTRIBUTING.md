# Contributing to Routes

Routes was built so agents can choose the right capability by policy. Contributions that keep that promise — more providers, better policies, richer diagnostics, stronger evals — are welcome.

Before opening a PR, read this guide and [AGENTS.md](AGENTS.md). The project values small, testable changes that improve one boundary over broad refactors.

## Extension boundaries

Routes is composable. Decide which boundary owns your change:

- **Classifier behavior** → `src/classifier.rs`
- **Route selection / scoring / policies** → `src/router.rs`
- **Provider integration** → `src/provider.rs`
- **Public API / contracts / OpenAPI** → `src/server.rs`, `src/types.rs`, `src/openapi.rs`
- **Production proof** → eval data, conformance checks, docs, or metrics

## Development setup

You need a recent Rust toolchain.

```bash
cargo fmt
cargo test
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml openapi
./scripts/check-contracts.sh
./scripts/test-advisory-exceptions.sh
./scripts/audit-dependencies.sh
```

The reusable release-blocking quality workflow also runs strict Clippy, validates both example and production configs, enforces the 50-example production eval gate, and exercises the controlled HTTP runtime gate. Dependency exceptions follow the owner/reason/expiry contract in [docs/security.md](docs/security.md).

## Adding a provider

1. Add a provider kind to `src/provider.rs` or reuse the OpenAI-compatible adapter.
2. Register the kind in config validation.
3. Add a config example to `examples/router.yaml` or `docs/examples/router.production.yaml`.
4. Run conformance:

```bash
cargo run -- --config examples/router.yaml provider-conformance-matrix
```

## Adding a routing policy

1. Implement the policy in `src/router.rs`.
2. Add the public policy name and alias mapping in config.
3. Add eval examples that exercise the new policy.
4. Run the eval gate:

```bash
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl
```

## Documentation expectations

If your change affects public behavior, update:

- `README.md` for high-level narrative and examples.
- `docs/usage.md` for commands, API behavior, or configuration.
- OpenAPI output (`cargo run -- --config examples/router.yaml openapi`).
- Config schema (`cargo run -- --config examples/router.yaml config-schema`).

## Production proof

Production readiness is tracked in [docs/production-readiness.md](docs/production-readiness.md). New features that affect routing correctness, provider health, auth, budgets, or observability should include evidence in that doc or in CI artifacts.

## Good first contributions

- Add an eval slice for a real prompt family: small web apps, coding agents, multimodal prompts, safety-sensitive requests, long-context work, or local-model routing.
- Add or harden a provider adapter for an OpenAI-compatible or native inference server.
- Improve policy diagnostics so route decisions are easier to explain.
- Add deployment examples for real model stacks.
- Improve docs for local Ollama, llama.cpp, vLLM, OpenRouter, or Cloudflare AI Gateway setups.

## Pull request checklist

- Keep the public API OpenAI-compatible where possible.
- Keep providers and models data-driven from config.
- Include route decision metadata when behavior changes.
- Add or update focused tests for router, provider, config, or server behavior.
- Update `README.md`, `docs/usage.md`, OpenAPI output, or config schema docs when public behavior changes.
- Do not hard-code a single provider as the only path.
- Preserve fail-closed fallback behavior for automatic routing.

## Issue reports

Useful routing issues include:

- Prompt shape or request payload.
- Expected model/provider or policy.
- Actual selected model/provider.
- Relevant config snippet.
- `decision_trace` or score component fields.
- Provider health and latency observations, if relevant.

Useful provider issues include:

- Provider kind and endpoint paths.
- Request route used by the client.
- HTTP status, content type, and response shape.
- Whether `provider-conformance` or `provider-conformance-matrix` passes.

## Design principle

Routes should stay composable. Prefer small, testable changes that improve one boundary over broad changes that couple routing logic directly to HTTP handlers.
