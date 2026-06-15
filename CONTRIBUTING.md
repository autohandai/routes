# Contributing to Routes

Thanks for helping make Routes easier to inspect, evaluate, and improve.

Routes is built around a few stable extension points. Before adding a feature, decide which boundary owns it:

- Classifier behavior belongs behind `PromptClassifier` in `src/classifier.rs`.
- Route selection belongs in scoring policy or candidate filtering code in `src/router.rs`.
- Provider integration belongs in `src/provider.rs`.
- Public API behavior belongs in `src/server.rs`, `src/types.rs`, and `src/openapi.rs`.
- Production proof belongs in eval data, conformance checks, docs, or metrics.

## Good First Contributions

- Add an eval slice for a real prompt family: small web apps, coding agents, multimodal prompts, safety-sensitive requests, long-context work, or local-model routing.
- Add or harden a provider adapter for an OpenAI-compatible or native inference server.
- Improve policy diagnostics so route decisions are easier to explain.
- Add deployment examples for real model stacks.
- Improve docs for local Ollama, llama.cpp, vLLM, OpenRouter, or Cloudflare AI Gateway setups.

## Local Workflow

```bash
cargo fmt
cargo test
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml openapi
```

For routing changes, also run:

```bash
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl
```

For provider changes, run the relevant conformance check:

```bash
cargo run -- --config examples/router.yaml provider-conformance-matrix
```

## Pull Request Checklist

- Keep the public API OpenAI-compatible where possible.
- Keep providers and models data-driven from config.
- Include route decision metadata when behavior changes.
- Add or update focused tests for router, provider, config, or server behavior.
- Update `README.md`, `docs/usage.md`, OpenAPI output, or config schema docs when public behavior changes.
- Do not hard-code a single provider as the only path.
- Preserve fail-closed fallback behavior for automatic routing.

## Issue Reports

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

## Design Notes

Routes should stay composable. Prefer small, testable changes that improve one boundary over broad changes that couple routing logic directly to HTTP handlers.
