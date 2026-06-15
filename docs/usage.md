# Routes Usage Guide

This guide keeps the detailed examples out of the main README while preserving the commands needed to run, inspect, evaluate, and operate Routes.

## Run

```bash
cargo run -- init-config router.yaml
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml openapi
cargo run -- --config examples/router.yaml config-schema
cargo run -- --config examples/router.yaml serve
```

Production-style checks:

```bash
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl \
  --min-examples 50 \
  --min-accuracy 0.90 \
  --min-domain-accuracy 0.90 \
  --min-model-accuracy 0.95 \
  --min-provider-accuracy 0.95 \
  --output router.eval-gate.json

cargo run -- load-test \
  --url http://127.0.0.1:8080 \
  --requests 1000 \
  --concurrency 32 \
  --output router.load.json

cargo run -- load-suite \
  --url http://127.0.0.1:8080 \
  --requests-per-scenario 1000 \
  --concurrency 32 \
  --output router.load-suite.json
```

Then point OpenAI-compatible clients at:

```text
http://127.0.0.1:8080/v1/chat/completions
http://127.0.0.1:8080/v1/responses
http://127.0.0.1:8080/v1/embeddings
http://127.0.0.1:8080/v1/images/generations
http://127.0.0.1:8080/v1/audio/speech
http://127.0.0.1:8080/v1/audio/transcriptions
http://127.0.0.1:8080/v1/audio/translations
```

Use model `auto`, `router-balanced`, `router-lowest-cost`, `router-fastest`, `router-highest-quality`, `router-local`, `router-privacy`, `router-multimodal`, `router-floor`, `router-nitro`, `router-quality`, `router-cost`, `router-capability`, or `router-domain` to let the router select the upstream model. Passing a configured model id or alias forwards to that model.

Proxied responses include:

```text
x-autohand-router-model: <selected model>
x-autohand-router-provider: <selected provider>
x-autohand-router-failovers: <number of skipped transient failures>
x-autohand-router-input-tokens: <estimated prompt tokens>
x-autohand-router-output-tokens: <requested output tokens>
x-autohand-router-cache: hit|miss
```

## Router API

Classify a prompt:

```bash
curl -s http://127.0.0.1:8080/v1/router/classify \
  -H 'content-type: application/json' \
  -d '{"input":"Add error handling to this function","classes":["difficulty","ambiguity","domain","modality","safety","cacheability","latency_sensitivity","reasoning_depth"]}'
```

Ask Routes to select a model:

```bash
curl -s http://127.0.0.1:8080/v1/router/multimodel \
  -H 'content-type: application/json' \
  -d '{
    "input": "Design a production event sourcing architecture",
    "allowed_providers": ["ollama", "openrouter"],
    "policy": "highest_quality",
    "default_model": "llama3.1:8b"
  }'
```

The response includes `model`, `provider`, `difficulty`, `ambiguity`, `domain`, `modality`, `safety`, `cacheability`, `latency_sensitivity`, `reasoning_depth`, confidence fields, policy, reason, fallback status, token estimates, context eligibility, capability eligibility, and a decision trace.

Clients that only need a difficulty label can call `/v1/router/raw`:

```bash
curl -s http://127.0.0.1:8080/v1/router/raw \
  -H 'content-type: application/json' \
  -d '{"input":"Add error handling to this function","mode":"balanced"}'
```

Provider-specific compatibility routes internally route through the multimodel engine constrained to that provider:

```bash
curl -s http://127.0.0.1:8080/v1/router/ollama \
  -H 'content-type: application/json' \
  -d '{"input":"Add error handling to this function","mode":"aggressive"}'
```

## Operations

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/openapi.json
curl -s http://127.0.0.1:8080/metrics
curl -s http://127.0.0.1:8080/metrics/prometheus
curl -s http://127.0.0.1:8080/v1/router/providers
```

Provider config supports `kind`, `base_url`, `timeout_ms`, `retries`, `health_path`, endpoint paths, `max_concurrency`, and `queue_timeout_ms`. Supported provider kinds are `open_ai_compatible`, `ollama`, `ollama_native`, `llama_cpp`, `llama_cpp_native`, `vllm`, `openrouter`, and `cloudflare_ai_gateway`.

To let routing react to provider health over time:

```yaml
runtime:
  provider_health_sampler:
    enabled: true
    interval_ms: 30000
    initial_delay_ms: 500
```

When enabled during `serve`, the router periodically checks provider health endpoints, records observed latency/status, and applies sampled latency/error penalties during automatic routing.

## Cache

Enable semantic response caching for automatic non-stream chat and Responses requests:

```yaml
cache:
  semantic:
    enabled: true
    embedding_model: local-hash
    similarity_threshold: 0.92
    ttl_seconds: 3600
    max_entries: 1024
    backend: file
    file_path: router.semantic-cache.json
    lock_timeout_ms: 1000
```

`embedding_model: local-hash` is deterministic and does not require a provider call. Setting `embedding_model` to a configured model id or alias with embeddings support uses provider-backed vectors.

## Shadow Evaluation

Collect pairwise routing data without changing foreground responses:

```yaml
shadow_eval:
  enabled: true
  sample_rate: 0.01
  output_path: router.shadow-eval.jsonl
  include_bodies: false
  judge:
    enabled: true
    model: qwen-classifier
    timeout_ms: 5000
```

The router returns the selected model response normally, then sends the same prompt to the next scored candidate in the background. The JSONL artifact records selected/shadow model IDs, providers, HTTP status, latency, body sizes, optional truncated bodies, and a winner judgement.

## Safety Routing

```yaml
safety:
  enabled: true
  unsafe_action: reject
  sensitive_action: redact
  force_model: safer-local-model
  redaction_replacement: "[redacted]"
```

Actions are `allow`, `reject`, `redact`, or `force_route`. `force_route` requires `force_model` to reference a configured model id or alias.

## Sticky Routing

```yaml
sticky_routing:
  enabled: true
  ttl_seconds: 1800
  prefer_model: true
  backend: file
  file_path: router.sticky-routes.json
  lock_timeout_ms: 1000
```

Sticky routing applies only to `auto` and `router-*` chat and Responses requests. It keys affinity from `user`, `metadata.session_id`, `metadata.conversation_id`, or `metadata.thread_id`, then prefers the previous model/provider before dispatching.

## Budgets

Budgets are optional. When a limit would be exceeded, the router returns `429` before upstream dispatch.

```yaml
budget:
  max_chat_requests: 10000
  max_total_tokens: 50000000
  max_estimated_cost_micros: 25000000
  accounting:
    backend: process
    file_path:
    lock_timeout_ms: 1000
```

`max_estimated_cost_micros` is measured in micro-dollars using each model's configured per-million token prices. The `file` backend uses a lock-protected JSON ledger so multiple local router processes can share reservations.

## Decision Traces

Decision traces are optional JSONL records for building evaluation datasets from real routing traffic:

```yaml
telemetry:
  decision_log_path: ./data/router-decisions.jsonl
  include_inputs: false
```

When enabled, `/v1/router/multimodel` and automatic chat routing write selected model/provider, classifications, token estimates, policy, candidates, and fallback status. Inputs are redacted unless `include_inputs` is set to `true`.

## Auth

Auth is disabled when no tokens are configured. To protect the router, set `auth.bearer_token_env` or `auth.bearer_tokens`:

```yaml
auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]
  bearer_tokens: []
```

Then call protected endpoints with:

```bash
curl -H "Authorization: Bearer $AUTOHAND_ROUTER_TOKEN" http://127.0.0.1:8080/metrics
```

`/health` and CORS preflight requests stay public. Protected responses include `x-autohand-router-request-id` for tracing.

## CLI

```bash
cargo run -- --config examples/router.yaml classify "Fix this typo"
cargo run -- --config examples/router.yaml route "Design an event sourcing system" --policy highest-quality
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml calibrate examples/eval.jsonl
cargo run -- --config examples/router.yaml optimize examples/eval.jsonl --write-config router.optimized.yaml --artifact router.optimization.json
```

## Configuration

Configuration is YAML-driven. Providers define upstreams. Models define capability, cost, domain strengths, aliases, context windows, capability tags, and whether they are local.

Model capability tags let the router reason about modality and feature requirements:

```yaml
models:
  - id: gemma4:12b-mlx
    provider: ollama
    capability: 0.62
    domains: [general, summary, coding]
    context_window: 128000
    capabilities:
      supports_vision: true
      supports_tools: true
      supports_json: true
      supports_code: true
      supports_web_apps: true
      supports_long_context: true
```

`/v1/router/multimodel` accepts `required_capabilities` such as `vision`, `audio`, `tools`, `json`, `code`, `web_apps`, and `long_context`. Automatic chat and Responses routing infer `vision`, `tools`, and `json` from OpenAI request payloads; automatic audio routes require `audio`.

Startup validation rejects ambiguous configs: provider names, model IDs, and aliases must be unique; provider URLs and paths must be valid HTTP-style values; timeouts and concurrency limits must be positive.

## Provider Conformance

Provider conformance records a live adapter artifact for a configured model:

```bash
cargo run -- --config examples/router.yaml provider-conformance local-fast \
  "Verify native adapter conformance" \
  --output router.provider-conformance.json
```

For release gates, validate every configured model/provider pair:

```bash
cargo run -- --config examples/router.yaml provider-conformance-matrix \
  "Verify every configured model adapter" \
  --output router.provider-matrix.json
```

The matrix also exercises each configured optional endpoint path for Responses, embeddings, images, speech, transcriptions, and translations. Paths set to `null` are recorded as skipped.

## Evaluation and Calibration

Evaluation datasets are JSONL files with prompt, expected tier, optional domain, optional exact model/provider expectations, policy, filters, and required capabilities:

```json
{"input":"Build a small web app","expected_tier":"balanced","expected_domain":"coding","expected_model":"gemma","expected_provider":"ollama","policy":"balanced","required_capabilities":["web_apps"]}
```

`eval` reports tier, domain, model, and provider accuracy, average selected cost, average capability, and miss details. `eval-gate` fails non-zero unless the dataset is large enough and meets configured thresholds. `calibrate` grid-searches heuristic difficulty cutoffs. `optimize` searches scoring-policy weights and uses lower average cost as a tiebreaker when accuracy is equal.

Run the production eval gate before promoting routing changes:

```bash
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl \
  --min-examples 50 \
  --min-accuracy 0.90 \
  --min-domain-accuracy 0.90 \
  --min-model-accuracy 0.95 \
  --min-provider-accuracy 0.95 \
  --output router.eval-gate.json
```

For production tuning:

```bash
cargo run -- --config examples/router.yaml calibrate examples/eval.jsonl --write-config router.calibrated.yaml
cargo run -- --config examples/router.yaml optimize examples/eval.jsonl --write-config router.optimized.yaml --artifact router.optimization.json
```

The optimization artifact includes the dataset fingerprint, deterministic train/holdout split metadata, baseline report, optimized train report, holdout validation report, selected config patch, replay command, and rollback guidance.
