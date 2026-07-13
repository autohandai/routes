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

`allowed_models` and `allowed_providers` are independent allowlists. If both are present, a candidate must satisfy both. Models rejected for an allowlist, required capability, or context-window constraint remain visible in diagnostics where applicable, but are never used for fallback or upstream failover.

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

`/metrics` includes bounded histogram summaries, and `/metrics/prometheus` exports cumulative bucket, sum, and count series for request-to-headers, route decisions, provider queue wait, upstream headers, upstream body, retry delay, stream first chunk, and stream duration. Labels are limited to fixed endpoint/outcome values plus configured provider/model identifiers; request paths and prompt content never become labels. Streaming time-to-first-chunk and full stream duration are separate from the non-streaming upstream-body distribution.

Provider config supports `kind`, `base_url`, `connect_timeout_ms`, `timeout_ms` (response headers), `stream_idle_timeout_ms`, `retries`, `retry_max_delay_ms`, `health_path`, endpoint paths, `max_concurrency`, and `queue_timeout_ms`. Supported provider kinds are `open_ai_compatible`, `ollama`, `ollama_native`, `llama_cpp`, `llama_cpp_native`, `vllm`, `openrouter`, and `cloudflare_ai_gateway`.

Transient retries honor numeric and HTTP-date `Retry-After` values up to `retry_max_delay_ms`. When the provider gives no delay, each attempt uses full jitter within a capped exponential window so concurrent router instances do not synchronize. `connect_timeout_ms` is enforced while establishing the socket, `timeout_ms` bounds receipt of response headers, and `stream_idle_timeout_ms` only bounds gaps between upstream response chunks. Explicit model requests retry the same provider according to its policy and never fail over; automatic requests may proceed to the next scored model only after that provider policy is exhausted.

### Ingress Resource Controls

Request admission is configured independently from provider concurrency. JSON and multipart limits plus the request-body idle timeout are enabled by default. Global admission and per-credential rate limits are opt-in so localhost development remains frictionless:

```yaml
runtime:
  ingress:
    max_json_body_bytes: 2097152
    max_multipart_body_bytes: 33554432
    body_idle_timeout_ms: 30000
    max_in_flight_requests: 256
    admission_queue_timeout_ms: 100
    per_credential_requests_per_minute: 600
```

`max_in_flight_requests` bounds work admitted into the router; requests that cannot enter within `admission_queue_timeout_ms` receive an OpenAI-shaped `503 router_overloaded`. Rate limits are isolated by configured bearer credential and return `429 rate_limit_exceeded`. Oversized bodies return `413 request_too_large`, and stalled uploads return `408 request_timeout`. Every rejection includes `x-autohand-router-request-id`.

The idle deadline only watches request-body progress. It is not a total request deadline and does not terminate long-lived streaming responses. Provider `max_concurrency` and `queue_timeout_ms` still apply immediately before each upstream dispatch.

### Native Chat Adapter Contracts

Native adapters are intentionally narrower than OpenAI-compatible adapters. The router validates their request contract before provider admission, so unsupported fields never reach the upstream and are never silently discarded. Explicit requests receive `unsupported_adapter_feature`; automatic requests exclude incompatible native candidates and report the adapter exclusions when no eligible model remains.

| Adapter | Preserved Chat controls | Structured output | Streaming and extended messages |
| --- | --- | --- | --- |
| `ollama_native` | `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`, `seed`, `stop`, and native `options` | `json_object` maps to Ollama JSON mode; `json_schema` maps to the schema object | Rejected before dispatch; message content must be a string and message extensions, tools, vision, audio, and other unmapped OpenAI fields are rejected |
| `llama_cpp_native` | `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`, `seed`, and `stop` | Rejected before dispatch | Rejected before dispatch; message content must be a string and message extensions, tools, vision, audio, and other unmapped OpenAI fields are rejected |

Use `kind: ollama` or `kind: llama_cpp` with the provider's OpenAI-compatible endpoints when the application needs a broader OpenAI request surface or streaming passthrough. Config validation also rejects model capability declarations that the selected native adapter cannot preserve.

To let routing react to provider health over time:

```yaml
runtime:
  provider_health_sampler:
    enabled: true
    interval_ms: 30000
    initial_delay_ms: 500
    check_timeout_ms: 5000
    max_concurrent_checks: 8
    observation_ttl_ms: 90000
    circuit_failure_threshold: 3
    circuit_open_ms: 30000
```

`/health` and `/health/live` are dependency-free liveness probes. `/health/ready` returns 200 while at least one configured model has a viable provider and 503 only when sampled provider failures leave no safe route.

When enabled during `serve`, the router checks providers concurrently up to `max_concurrent_checks`, bounds each probe by `check_timeout_ms`, and records latency/status for automatic routing. Observations stop affecting scores after `observation_ttl_ms`. Repeated failures open a provider circuit; after `circuit_open_ms`, exactly one half-open probe is admitted, and a successful probe closes the circuit. `/v1/router/providers` reports freshness and circuit state for each provider.

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

For safety, semantic caching is disabled when bearer authentication is enabled (to avoid cross-tenant response reuse) and for requests with behavior-changing options such as tools, structured output, sampling, or a `user` field. Use a tenant-aware cache namespace before enabling shared authenticated caching.

## Shadow Evaluation

Collect pairwise routing data without changing foreground responses:

```yaml
shadow_eval:
  enabled: true
  sample_rate: 0.01
  output_path: router.shadow-eval.jsonl
  include_bodies: false
  writer_queue_capacity: 1024
  max_file_bytes: 67108864
  retained_files: 5
  max_pending_tasks: 64
  max_concurrent_tasks: 4
  judge:
    enabled: true
    model: qwen-classifier
    timeout_ms: 5000
```

The router returns the selected model response normally, then sends the same prompt to the next scored candidate through a bounded background pool. `max_pending_tasks` bounds running plus queued evaluations and `max_concurrent_tasks` bounds simultaneous provider work. The JSONL artifact records selected/shadow model IDs, providers, HTTP status, latency, body sizes, optional truncated bodies, and a winner judgement.

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

Safety routing applies to automatic `auto` and `router-*` Chat Completions and Responses requests; explicit model requests retain their strict model contract. The local safety preflight classifies a normalized view of every forwarded text field before an optional external classifier is called. With `redact`, it covers user/system/assistant/tool content, nested content arrays and objects, tool/function arguments, request metadata, user identifiers, and URLs. Structural values such as roles, item types, tool names and IDs, and JSON-schema control fields are preserved.

Tool/function `arguments` strings are parsed as JSON, redacted recursively, and serialized back to valid JSON. If a sensitive request contains an ambiguous arguments shape that cannot be safely parsed, the router rejects it locally with `unsafe_redaction_shape` and does not dispatch upstream. Decision-trace and shadow-eval inputs use the redacted view. Sticky routing is skipped for redacted or force-routed sensitive requests so sensitive session identifiers are not persisted or collapsed onto the replacement value.

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
    semantics: logical_request
    scope: credential
```

`max_estimated_cost_micros` is measured in micro-dollars using each model's configured per-million token prices. The `file` backend uses a lock-protected JSON ledger so multiple local router processes can share reservations. `scope: credential` maintains an independent allowance for each configured bearer token without storing token values; unauthenticated localhost traffic uses the `anonymous` scope. `scope: global` preserves one process/file-wide allowance.

Budget accounting deliberately uses `semantics: logical_request`, not authoritative provider spend. One foreground proxy request reserves a conservative input-plus-requested-output estimate exactly once before any upstream dispatch. Cache hits retain that one logical charge. Retries, transient failover to another model, and failures after reservation do not add or refund charges; shadow evaluation and classifier/judge control-plane calls are explicitly uncharged. This avoids ambiguous double charges and makes the guard deterministic across streaming, missing usage data, and restarts. Provider-reported actual usage remains available in usage metrics, but operators needing invoice-grade spend enforcement must reconcile those metrics with provider billing rather than treating this logical guard as a spend ledger.

## Decision Traces

Decision traces are optional JSONL records for building evaluation datasets from real routing traffic:

```yaml
telemetry:
  decision_log_path: ./data/router-decisions.jsonl
  include_inputs: false
  queue_capacity: 1024
  max_file_bytes: 67108864
  retained_files: 5
```

When enabled, `/v1/router/multimodel` and automatic chat routing enqueue selected model/provider, classifications, token estimates, policy, candidates, and fallback status without awaiting disk I/O. Inputs are redacted unless `include_inputs` is set to `true`. Dedicated FIFO writers rotate at `max_file_bytes`, retain the configured number of files, drop new records when their bounded queue is full, and expose written/dropped/error/rotation counters. Graceful shutdown drains admitted background evaluations and flushes both JSONL queues within the configured shutdown window.

## Auth

Auth is optional for loopback development. Set `auth.bearer_token_env` or `auth.bearer_tokens` before binding to a network interface:

```yaml
auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]
  bearer_tokens: []
  allow_unauthenticated_network: false
```

Configured environment variables are resolved once when `serve` starts. A missing or empty token fails startup instead of disabling auth. Non-loopback binds also require at least one token source unless `allow_unauthenticated_network: true` is set explicitly for a deployment where a trusted API gateway enforces authentication and direct origin access is blocked.

Then call protected endpoints with:

```bash
curl -H "Authorization: Bearer $AUTOHAND_ROUTER_TOKEN" http://127.0.0.1:8080/metrics
```

`/health`, `/openapi.json`, and CORS preflight requests stay public. Authentication failures include `WWW-Authenticate: Bearer` and `x-autohand-router-request-id`.

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
      supported_endpoints: [chat, responses]
      supports_vision: true
      supports_tools: true
      supports_json: true
      supports_code: true
      supports_web_apps: true
      supports_long_context: true
```

Endpoint support is explicit at both boundaries. Optional provider paths default to absent, and a model with no `supported_endpoints` entry is Chat-only. List only APIs proven for that exact provider/model pair:

```yaml
    capabilities:
      supported_endpoints: [chat, responses, embeddings]
```

Supported endpoint names are `chat`, `responses`, `embeddings`, `images`,
`speech`, `audio_transcriptions`, and `audio_translations`. Automatic routes
and explicit model requests both enforce this allowlist before dispatch.

`/v1/router/multimodel` accepts `required_capabilities` such as `vision`, `audio`, `tools`, `json`, `code`, `web_apps`, and `long_context`. Automatic chat and Responses routing infer `vision`, `tools`, and `json` from OpenAI request payloads; automatic audio routes require `audio`.

Startup validation rejects ambiguous configs: provider names, model IDs, and aliases must be unique; provider URLs and paths must be valid HTTP-style values; timeouts and concurrency limits must be positive.
It also rejects duplicate model endpoints, endpoints without a compatible provider path, and non-Chat paths on native Ollama/llama.cpp adapters.

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

The matrix exercises only the intersection of explicit provider paths and each model's declared endpoints. A provider path alone never causes an endpoint probe or makes a model eligible. Paths or model endpoints that are absent are recorded as skipped.

To require live conformance evidence at startup, point runtime config at a previously generated matrix. Relative paths resolve from the router config directory:

```yaml
runtime:
  provider_conformance_artifact: router.provider-matrix.json
```

Every declared endpoint for every configured provider/model pair must have a passing matrix entry. Missing, failed, duplicate, unknown-version, or mismatched reports fail config loading before the server starts.

## Evaluation and Calibration

Evaluation datasets are JSONL files with prompt, expected tier, optional domain, optional exact model/provider expectations, policy, filters, and required capabilities:

```json
{"input":"Build a small web app","expected_tier":"balanced","expected_domain":"coding","expected_model":"gemma","expected_provider":"ollama","policy":"balanced","required_capabilities":["web_apps"]}
```

`eval` reports tier, domain, model, and provider accuracy, average selected cost, average capability, and miss details. `eval-gate` is the deterministic heuristic gate. `configured-eval-gate` independently exercises the configured classifier on a reproducible seeded holdout and fails when its observed fallback rate exceeds the configured maximum. Both artifacts identify the classifier/runtime and make domain/model/provider sample minimums explicit. `runtime-gate` runs auth, capability, context, failover, outcome-metric, and streaming scenarios through the real Axum stack against controlled mock providers. `calibrate` grid-searches heuristic difficulty cutoffs. `optimize` searches scoring-policy weights and uses lower average cost as a tiebreaker when accuracy is equal.

Run the production eval gate before promoting routing changes:

```bash
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl \
  --min-examples 50 \
  --min-accuracy 0.90 \
  --min-domain-accuracy 0.90 \
  --min-model-accuracy 0.95 \
  --min-provider-accuracy 0.95 \
  --min-domain-examples 50 \
  --min-model-examples 11 \
  --min-provider-examples 22 \
  --output router.eval-gate.json
```

For a configuration that enables `llm_judge` or `route_llm`, run the independent credentialed gate and the deterministic HTTP-runtime suite:

```bash
cargo run -- --config router.production.yaml configured-eval-gate examples/eval.production.jsonl \
  --holdout-ratio 0.20 --holdout-seed 2709397542 \
  --min-examples 10 --max-fallback-rate 0.05 \
  --output router.configured-eval-gate.json
cargo run -- runtime-gate --output router.runtime-gate.json
```

For production tuning:

```bash
cargo run -- --config examples/router.yaml calibrate examples/eval.jsonl --write-config router.calibrated.yaml
cargo run -- --config examples/router.yaml optimize examples/eval.jsonl --write-config router.optimized.yaml --artifact router.optimization.json
```

The optimization artifact includes the dataset fingerprint, deterministic train/holdout split metadata, baseline report, optimized train report, holdout validation report, selected config patch, replay command, and rollback guidance.
