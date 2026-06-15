# Autohand Router

Rust LLM router and OpenAI-compatible proxy for hosted and local inference. It exposes Morph-style routing endpoints and can sit in front of Ollama, llama.cpp, OpenRouter, Cloudflare AI Gateway, or any OpenAI-compatible chat, Responses, or embeddings service.

See [PRODUCTION.md](PRODUCTION.md) for the explicit 100M-user readiness bar and open-weight provider roadmap.
See [docs/](docs/README.md) for container packaging and deployment examples for AWS, Google Cloud, Azure, and Cloudflare.
Use `cargo run -- config-schema` to print a JSON Schema for `router.yaml` that can be wired into YAML editors or CI checks.

## Run

```bash
cargo run -- init-config router.yaml
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml openapi
cargo run -- --config examples/router.yaml config-schema
cargo run -- --config examples/router.yaml serve
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl --output router.eval-gate.json
cargo run -- load-test --url http://127.0.0.1:8080 --requests 1000 --concurrency 32 --output router.load.json
cargo run -- load-suite --url http://127.0.0.1:8080 --requests-per-scenario 1000 --concurrency 32 --output router.load-suite.json
cargo run -- --config router.with-judge.yaml judge-smoke --output router.judge-smoke.json
cargo run -- --config examples/router.yaml provider-conformance local-fast --output router.provider-conformance.json
cargo run -- --config examples/router.yaml provider-conformance-matrix --output router.provider-matrix.json
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

Use model `auto`, `router-balanced`, `router-floor`, `router-nitro`, `router-quality`, `router-cost`, `router-capability`, or `router-domain` to let the router select the upstream model. Passing a configured model id or alias forwards to that model.

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

```bash
curl -s http://127.0.0.1:8080/v1/router/classify \
  -H 'content-type: application/json' \
  -d '{"input":"Add error handling to this Rust function","classes":["difficulty","ambiguity","domain","modality","safety","cacheability","latency_sensitivity","reasoning_depth"]}'
```

```bash
curl -s http://127.0.0.1:8080/v1/router/multimodel \
  -H 'content-type: application/json' \
  -d '{
    "input": "Design a production event sourcing architecture",
    "allowed_providers": ["ollama", "openrouter"],
    "policy": "capability_heavy",
    "default_model": "llama3.1:8b"
  }'
```

The response includes `model`, `provider`, `difficulty`, `ambiguity`, `domain`, `modality`, `safety`, `cacheability`, `latency_sensitivity`, `reasoning_depth`, confidence fields, policy, reason, and whether fallback was used.
It also includes estimated input tokens, requested output tokens, per-candidate context eligibility, and per-candidate capability eligibility so oversized or modality-specific prompts do not route to models that cannot fit or satisfy them.
`decision_trace` includes the classifier labels, selected policy weights, required capabilities, selected score, rejected candidates, and per-candidate score components for debugging routing decisions.
Reasoning depth raises or lowers the capability target, and latency sensitivity changes how strongly observed/static provider latency affects candidate scores.
Policy presets are config-driven: `balanced` is the default tradeoff, `floor` selects the cheapest acceptable candidate, `nitro` emphasizes fast healthy providers, and `quality` favors the strongest candidate. Legacy policy names `cost_efficient`, `capability_heavy`, and `domain_skills` remain supported.
Optional learned scoring is configured under `scoring.learned`; it applies a bounded linear score boost from trainable feature/model weights such as `difficulty.hard`, `domain.coding`, `modality.vision`, `supports_web_apps`, `capability`, `cost`, `provider.<name>`, and `model.<id-or-alias>`. Learned contributions are exposed as `score_components.learned_score_boost`.
When `default_model` is provided with `allowed_models` or `allowed_providers`, the default must satisfy those filters. The router will not silently return an out-of-filter fallback.

Legacy Morph clients can call `/v1/router/raw` for a difficulty-only response:

```bash
curl -s http://127.0.0.1:8080/v1/router/raw \
  -H 'content-type: application/json' \
  -d '{"input":"Add error handling to this Rust function","mode":"balanced"}'
```

Provider-specific legacy routes are also supported and internally route through the multimodel engine constrained to that provider:

```bash
curl -s http://127.0.0.1:8080/v1/router/ollama \
  -H 'content-type: application/json' \
  -d '{"input":"Add error handling to this Rust function","mode":"aggressive"}'
```

## Operations

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/openapi.json
curl -s http://127.0.0.1:8080/metrics
curl -s http://127.0.0.1:8080/metrics/prometheus
curl -s http://127.0.0.1:8080/v1/router/providers
```

Provider config supports `kind`, `timeout_ms`, `retries`, and `health_path`. Supported provider kinds are `open_ai_compatible`, `ollama`, `ollama_native`, `llama_cpp`, `llama_cpp_native`, `vllm`, `openrouter`, and `cloudflare_ai_gateway`. Upstream chat, responses, embeddings, image-generation, speech, transcription, and translation calls retry transient `408`, `429`, and `5xx` responses, plus timeout/connect failures, before returning a router error.
Provider config also supports `chat_path`, optional `responses_path`, optional `embeddings_path`, optional `images_path`, optional `speech_path`, optional `audio_transcriptions_path`, optional `audio_translations_path`, optional `max_concurrency`, and `queue_timeout_ms` to apply backpressure before dispatching to local or hosted inference. The OpenRouter adapter injects safe default attribution headers unless config overrides them through `extra_headers`.

Set `responses_path`, `embeddings_path`, `images_path`, `speech_path`, `audio_transcriptions_path`, or `audio_translations_path` to `null` for providers that do not support those OpenAI-compatible endpoints.

To let routing react to provider health over time, enable the runtime health sampler:

```yaml
runtime:
  provider_health_sampler:
    enabled: true
    interval_ms: 30000
    initial_delay_ms: 500
```

When enabled during `serve`, the router periodically checks each provider health endpoint, records observed latency and status, and applies those sampled latency/error penalties during automatic routing. `/v1/router/providers` returns both the live check result and the latest sampled observations.

To enable the semantic response cache for automatic non-stream chat and Responses requests:

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

The cache uses cosine similarity. `embedding_model: local-hash` is deterministic and does not require a provider call; setting `embedding_model` to any configured model id or alias with embeddings support uses real provider-backed vectors. `backend: memory` is process-local; `backend: file` stores JSON cache entries behind a lock file so multiple local router replicas can share hits. It only stores successful buffered responses for prompts classified as medium/high cacheability, and it skips explicit model requests and streaming passthrough. Cache behavior is visible through `x-autohand-router-cache`, `x-autohand-router-cache-similarity`, JSON metrics, and Prometheus event counters.

To collect pairwise routing data without changing foreground responses, enable shadow evaluation:

```yaml
shadow_eval:
  enabled: true
  sample_rate: 0.01
  output_path: router.shadow-eval.jsonl
  include_bodies: false
  judge:
    enabled: true
    # Optional. Omit for deterministic local judging.
    model: qwen-classifier
    timeout_ms: 5000
```

For sampled automatic non-stream chat and Responses requests, the router returns the selected model response normally, then sends the same prompt to the next scored candidate in the background. The JSONL artifact records selected/shadow model IDs, providers, HTTP status, latency, body sizes, optional truncated bodies, and a `winner` judgement. Without `judge.model`, the judgement is deterministic and local, using status/content/latency scores. With `judge.model`, the router asks the configured model for JSON answer-quality scores and falls back to local judging if the model times out, errors, or returns invalid output. Bodies are redacted by default.

To enforce safety routing on automatic chat and Responses requests:

```yaml
safety:
  enabled: true
  unsafe_action: reject
  sensitive_action: redact
  force_model: safer-local-model
  redaction_replacement: "[redacted]"
```

Actions are `allow`, `reject`, `redact`, or `force_route`. `force_route` requires `force_model` to reference a configured model id or alias. Unsafe prompts include jailbreak/prompt-injection and abuse markers; sensitive prompts include PII, secrets, credentials, and regulated-workflow markers. Safety actions are exposed through JSON and Prometheus counters.

To keep multi-turn automatic requests on a stable backend, enable sticky routing:

```yaml
sticky_routing:
  enabled: true
  ttl_seconds: 1800
  prefer_model: true
  backend: file
  file_path: router.sticky-routes.json
  lock_timeout_ms: 1000
```

Sticky routing applies only to `auto`/`router-*` chat and Responses requests. It keys affinity from `user`, `metadata.session_id`, `metadata.conversation_id`, or `metadata.thread_id`, then prefers the previously selected model/provider before dispatching. `backend: memory` is process-local; `backend: file` stores JSON affinity records behind a lock file so multiple local router replicas can share them. Explicit model requests remain strict. Affinity activity is visible through `sticky_routing_hits`, `sticky_routing_writes`, and Prometheus event counters.

Use `kind: ollama` for Ollama's OpenAI-compatible surface. Use `kind: ollama_native` with `chat_path: /api/chat` to transform native Ollama chat responses into OpenAI-compatible `/v1/chat/completions` responses for local open-weight models. Use `kind: llama_cpp` for llama.cpp's OpenAI-compatible server and `kind: llama_cpp_native` with `chat_path: /completion` for the native completion server. Use `kind: vllm` for vLLM's OpenAI-compatible server; vLLM currently belongs on the OpenAI-compatible adapter path, with explicit provider identity for health, metrics, and conformance artifacts.

`serve` handles Ctrl-C by stopping new accepts and giving in-flight work `runtime.graceful_shutdown_timeout_ms` to finish before the server future is forced to stop.

For `auto` and `router-*` chat, responses, embeddings, image-generation, speech, transcription, or translation requests, the router also fails over across the scored candidate list. Explicit model requests stay strict and do not silently switch models.

`/metrics` includes request counters, selected model/provider counters, semantic cache hit/miss counters, shadow-eval sample/success/error counters, safety reject/redact/force-route counters, sticky-routing hit/write counters, classifier adapter success/fallback counters, parsed upstream token usage for non-stream responses, and estimated cost from configured per-million token prices. `/metrics/prometheus` exposes the same snapshot in Prometheus text exposition format for fleet scraping. Streaming responses are passed through without buffering, so token usage is only counted when the upstream sends it in a buffered JSON response.

## Load Testing

The built-in load tester exercises the live HTTP API and fails non-zero if SLO thresholds are missed. It defaults to `/v1/router/multimodel`, which verifies routing latency without requiring an upstream model provider:

```bash
cargo run -- load-test \
  --url http://127.0.0.1:8080 \
  --requests 1000 \
  --concurrency 32 \
  --slo-p95-ms 250 \
  --slo-error-rate 0.001 \
  --output router.load.json
```

The JSON report includes request counts, success/failure counts, error rate, throughput, p50/p90/p95/p99/max latency, and the evaluated SLO result.

For production release gates, run the multi-endpoint suite against a live router and configured providers:

```bash
cargo run -- load-suite \
  --url http://127.0.0.1:8080 \
  --requests-per-scenario 1000 \
  --concurrency 32 \
  --slo-p95-ms 250 \
  --slo-error-rate 0.001 \
  --output router.load-suite.json
```

The suite emits one report per scenario and fails non-zero unless every scenario meets SLOs. The default scenarios cover `/v1/router/multimodel`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/images/generations`, and `/v1/audio/speech`.

## Budgets

Budgets are optional. When a limit would be exceeded, the router returns `429` before dispatching to the upstream provider. `max_chat_requests` applies to all model front-door calls: chat completions, Responses, embeddings, image generations, speech, transcription, and translation.

```yaml
budget:
  max_chat_requests: 10000
  max_total_tokens: 50000000
  max_estimated_cost_micros: 25000000
  accounting:
    # process is the default. Use file for shared local multi-process enforcement.
    backend: process
    file_path:
    lock_timeout_ms: 1000
```

`max_estimated_cost_micros` is measured in micro-dollars using each model's configured per-million token prices. `/metrics` reports the accounting backend, configured limits, used budget, remaining budget, and `budget_rejections`. The `file` backend uses a lock-protected JSON ledger so multiple local router processes can share request/token/cost reservations before upstream dispatch.

## Decision Traces

Decision traces are optional JSONL records for building evaluation datasets from real routing traffic:

```yaml
telemetry:
  decision_log_path: ./data/router-decisions.jsonl
  include_inputs: false
```

When enabled, `/v1/router/multimodel` and automatic chat routing write the selected model/provider, classifications, token estimates, policy, candidates, and fallback status. Inputs are redacted unless `include_inputs` is set to `true`.

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
cargo run -- --config examples/router.yaml route "Design an event sourcing system" --policy capability-heavy
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml calibrate examples/eval.jsonl
cargo run -- --config examples/router.yaml optimize examples/eval.jsonl --write-config router.optimized.yaml --artifact router.optimization.json
```

## Configuration

Configuration is YAML-driven. Providers define OpenAI-compatible upstreams. Models define capability, cost, domain strengths, aliases, context windows, capability tags, and whether they are local.

Model capability tags let the router reason about modality and feature requirements without collapsing everything into one capability score:

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

`/v1/router/multimodel` accepts `required_capabilities` such as `vision`, `audio`, `tools`, `json`, `code`, `web_apps`, and `long_context`. Automatic chat and Responses routing infer `vision`, `tools`, and `json` from OpenAI request payloads; automatic audio speech/transcription/translation routes require `audio`. Candidate diagnostics include `capability_eligible` and `missing_capabilities`.

Startup validation rejects ambiguous configs: provider names, model IDs, and aliases must be unique; provider URLs and paths must be valid HTTP-style values; timeouts and concurrency limits must be positive. This prevents silent routing drift when many local and hosted providers are configured together.

The deterministic local classifier is always available, so routing works without external services. For learned routing, set `classifier.backend` to `llm_judge` or `route_llm` and configure `classifier.adapters.<backend>.model` with any model id or alias, for example a small Qwen classifier model. Adapter requests go through the same provider boundary as chat, so native adapters such as `ollama_native` and `llama_cpp_native` can be used for local open-weight classifiers. `classifier.llm_judge_model` remains a compatibility shortcut for `classifier.backend: llm_judge`.

Classifier adapter output must include difficulty, ambiguity, domain, and their confidence fields with finite values from `0.0` to `1.0`; advanced heads such as modality, safety, cacheability, latency sensitivity, and reasoning depth are accepted when present and otherwise receive conservative defaults. Invalid output, timeout, or provider errors increment classifier fallback counters and automatically fall back to the heuristic classifier. The architecture keeps classifier and scoring boundaries separate so GEPA-style optimizers or remote Morph calls can be added without changing HTTP handlers.

Before enabling an LLM judge in production, run a live smoke against the configured judge model:

```bash
cargo run -- --config router.with-judge.yaml judge-smoke \
  "Design a production Rust LLM router with provider failover" \
  --output router.judge-smoke.json
```

`judge-smoke` fails non-zero unless exactly one judge request succeeds without fallback or heuristic routing. The JSON report includes the configured judge model, selected classifications, and before/after judge metrics.

## Provider Conformance

Provider conformance records a live adapter artifact for a configured model. It sends an OpenAI chat request through the selected provider adapter and fails non-zero unless the response is an OpenAI-compatible chat completion with the selected model and assistant content.

```bash
cargo run -- --config examples/router.yaml provider-conformance local-fast \
  "Verify native adapter conformance" \
  --output router.provider-conformance.json
```

The JSON report includes provider name/kind, health probe result, HTTP status, content type, OpenAI chat-shape checks, usage fields, and a top-level `pass` boolean. Use this before promoting native adapters such as `ollama_native` into production routing.

For release gates, validate every configured model/provider pair in one artifact:

```bash
cargo run -- --config examples/router.yaml provider-conformance-matrix \
  "Verify every configured model adapter" \
  --output router.provider-matrix.json
```

The matrix report includes every per-model conformance report, `passed`/`failed` counts, and a top-level `pass` boolean. It also exercises each configured optional endpoint path for Responses, embeddings, images, speech, transcriptions, and translations; paths set to `null` are recorded as skipped. It continues after individual provider failures so CI keeps the full failure set for debugging.

## Evaluation and Calibration

Evaluation datasets are JSONL files with prompt, expected tier, optional domain, optional exact model/provider expectations, policy, filters, and required capabilities:

```json
{"input":"Build a small web app","expected_tier":"balanced","expected_domain":"coding","expected_model":"gemma","expected_provider":"ollama","policy":"balanced","required_capabilities":["web_apps"]}
```

`eval` reports tier, domain, model, and provider accuracy, average selected cost, average capability, and miss details for each checked dimension. `eval-gate` fails non-zero unless the dataset is large enough and meets configured tier/domain/model/provider accuracy thresholds. `calibrate` grid-searches the heuristic difficulty cutoffs and can write a calibrated config. `optimize` also searches scoring-policy weights and uses lower average cost as a tiebreaker when accuracy is equal.

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

For production tuning, write both the optimized config and a replayable JSON artifact:

```bash
cargo run -- --config examples/router.yaml calibrate examples/eval.jsonl --write-config router.calibrated.yaml
cargo run -- --config examples/router.yaml optimize examples/eval.jsonl --write-config router.optimized.yaml --artifact router.optimization.json
```

The optimization artifact includes the dataset fingerprint, deterministic train/holdout split metadata, baseline report, optimized train report, holdout validation report, selected config patch, replay command, and rollback guidance. For datasets with at least five examples, optimization trains on a stable 80% split and reports whether holdout tier/domain accuracy stays at least as good as baseline. It does not serialize provider API keys from the runtime config.
