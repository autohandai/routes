# AGENTS.md

This repository is a Rust LLM router intended to sit in front of OpenAI-compatible providers, local inference servers, Ollama, llama.cpp, OpenRouter, Cloudflare AI Gateway, and any service that accepts `/v1/chat/completions`.

## Engineering Contract

- Keep the public API OpenAI-compatible wherever possible.
- Keep Morph-compatible router endpoints:
  - `POST /v1/router/classify`
  - `POST /v1/router/multimodel`
- Never hard-code a single provider as the only path. Providers and models must be data driven from config.
- Router failures must fail closed to a configured fallback model, not crash the proxy path.
- Route decisions should include enough metadata to debug why a model was chosen.
- Local deterministic routing is required even when no external classifier, GEPA optimizer, or judge model is configured.
- Provider calls must use configured timeouts/retries and expose enough health/metric data for production debugging.
- Provider calls must respect configured concurrency limits before dispatching upstream.
- Automatic route requests may fail over to the next scored candidate on transient upstream failures; explicit model requests must stay strict.
- Auth must be optional for localhost development but available through bearer tokens before exposing provider keys on a network.
- Routing must respect model context windows using conservative token estimates and expose context eligibility in route diagnostics.
- Scoring policy weights are configuration, not hard-coded constants; use eval/optimize before changing routing tradeoffs.
- Metrics should expose selected model/provider counts and parsed token/cost usage without breaking streaming passthrough.
- Optional process-local budgets should reject over-limit chat requests before upstream dispatch.
- Optional decision trace JSONL should default to redacted inputs and be suitable for future eval/optimization workflows.
- Keep `/openapi.json` and the `openapi` CLI command in sync with public routes and response headers.

## Commands

- `cargo fmt`
- `cargo test`
- `cargo run -- --config examples/router.yaml validate`
- `cargo run -- --config examples/router.yaml openapi`
- `cargo run -- --config examples/router.yaml serve`
- `cargo run -- init-config router.yaml`
- `cargo run -- --config examples/router.yaml classify "Fix this typo"`
- `cargo run -- --config examples/router.yaml route "Design an event sourcing system"`
- `cargo run -- --config examples/router.yaml eval examples/eval.jsonl`
- `cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl --output router.eval-gate.json`
- `cargo run -- --config examples/router.yaml calibrate examples/eval.jsonl`
- `cargo run -- --config examples/router.yaml optimize examples/eval.jsonl --write-config router.optimized.yaml --artifact router.optimization.json`
- `cargo run -- load-test --url http://127.0.0.1:8080 --requests 1000 --concurrency 32 --output router.load.json`
- `cargo run -- load-suite --url http://127.0.0.1:8080 --requests-per-scenario 1000 --concurrency 32 --output router.load-suite.json`
- `cargo run -- --config router.with-judge.yaml judge-smoke --output router.judge-smoke.json`
- `cargo run -- --config examples/router.yaml provider-conformance local-fast --output router.provider-conformance.json`
- `cargo run -- --config examples/router.yaml provider-conformance-matrix --output router.provider-matrix.json`
- `curl -s http://127.0.0.1:8080/metrics`
- `curl -s http://127.0.0.1:8080/metrics/prometheus`
- `curl -s http://127.0.0.1:8080/v1/router/providers`

## Architecture

- `src/config.rs`: YAML/env backed runtime config.
- `src/types.rs`: API contracts, labels, policies, provider/model definitions.
- `src/classifier.rs`: deterministic prompt classifier with Morph-like labels.
- `src/router.rs`: candidate filtering, policies, fallback handling, and route explanations.
- `src/provider.rs`: provider adapter registry and OpenAI-compatible upstream forwarding.
- `src/server.rs`: axum HTTP API.

When adding advanced routers, implement them behind the `PromptClassifier` or route-scoring boundary instead of coupling them directly to HTTP handlers.
