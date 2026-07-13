# Production readiness

This router is intended to be the Autohand Code model gateway for hosted and open-weight inference. It should be safe to put in front of Ollama, llama.cpp, vLLM, OpenRouter, Cloudflare AI Gateway, and future provider adapters without changing application callers.

## Desired properties for opening to the world

The router is not considered production-complete until these requirements are true and verified, for many more use cases and providers that we have yet to support.

- Correctness: Current implementation is compatible routing must never return a model outside request constraints. Fallback behavior must be explicit, observable, and safe.
- Provider abstraction: provider integrations must be adapter-backed, not only OpenAI-compatible URL forwarding. OpenAI-compatible chat is the first adapter; future adapters include Responses API, embeddings, native provider APIs, and local open-weight runtimes.
- Open-weight first: local and self-hosted models must be first-class targets with capability, context, cost, domain, health, concurrency, and queue controls.
- Routing intelligence: heuristic routing, LLM-judge routing, and offline optimization must be measurable against held-out eval datasets. Any GEPA-style optimizer must produce replayable configs, reports, and rollback-safe artifacts.
- Operations: auth, budgets, metrics, request IDs, decision traces, health checks, graceful shutdown, and config validation must be enabled before internet exposure.
- Reliability: transient failover must be tested across candidate providers, not only unit-tested. Explicit model calls must remain strict.
- Scale: the HTTP service must support horizontal replicas with externalized counters/traces when global budgets or fleet-wide accounting are required.
- Security: provider keys must stay server-side, auth comparisons must remain constant-time, and trace logging must default to redacted prompts.

## Current Evidence

- CI and release call the same release-blocking quality workflow. It enforces formatting, strict Clippy, all tests, example and production config validation, deterministic structural OpenAPI/config-schema checks, the 50-example eval gate with uploaded failure artifacts, the controlled HTTP runtime gate, and a pinned RustSec scan. Release build and publish jobs depend on that gate.
- Release validation generates and schema-validates a hash-linked evidence bundle at the exact Git revision. It includes repeated controlled six-scenario load runs with p95/p99 and confidence intervals, environment/resource metadata, explicit thresholds, and the schema-v2 support matrix. Release notes and assets identify this as loopback-mock router overhead and explicitly exclude external-provider variability.
- A protected self-hosted staging workflow runs the live matrix and a fail-closed provider promotion gate. Freshness, redacted config identity, reported provider/model versions, every advertised endpoint, and every advertised streaming/tools/JSON/vision/audio probe must pass; raw failure artifacts are retained and unadvertised skips remain explicitly justified.
- A separate staging job requires a live `llm_judge`/`route_llm` adapter to meet success, fallback, p95, and seeded-holdout accuracy thresholds. Its redacted aggregate artifact also proves timeout, invalid-JSON, and 429 responses each produce one deterministic heuristic fallback with the expected counters.
- A third staging job gates every advertised provider/model stream profile on SSE framing, first-chunk/completion bounds, terminal usage, byte-for-byte passthrough, cancellation accounting, released concurrency capacity, and post-cancel readiness. Credential-free injections separately prove capped `Retry-After`, incomplete-body error propagation/counters, zero leaked streams, and bounded active-stream shutdown.
- A fourth staging job requires an already deployed candidate to report the exact Git revision and redacted config fingerprint, then enforces sustained mixed unary/stream/multipart p95, p99, error, provider-queue, and Linux peak-RSS thresholds. It also proves multipart size boundaries, exact multi-process file-budget admission under contention, stale-lock reuse, corrupt-ledger fail-closed behavior, restart persistence, and bounded two-replica rolling replacement in controlled OS processes.
- Current endpoints are implemented: `/v1/router/classify`, `/v1/router/multimodel`, `/v1/router/raw`, and `/v1/router/{provider}`.
- OpenAI-compatible chat proxying works for local Ollama through a configured alias.
- OpenAI-compatible chat forwarding is isolated behind a provider adapter registry with explicit provider kinds for generic OpenAI-compatible services, Ollama, llama.cpp, vLLM, OpenRouter, and Cloudflare AI Gateway.
- Native Ollama `/api/chat` can be configured with `kind: ollama_native`; the adapter rewrites OpenAI chat requests into Ollama chat payloads and transforms native Ollama responses back into OpenAI-compatible chat completions with usage accounting.
- Native llama.cpp `/completion` can be configured with `kind: llama_cpp_native`; the adapter rewrites OpenAI chat requests into llama.cpp completion prompts and transforms native completion responses back into OpenAI-compatible chat completions with usage accounting.
- Native adapters publish strict feature contracts. Supported sampling/token controls round-trip into captured native bodies; unsupported fields, message shapes, capabilities, and streaming are rejected locally before concurrency admission or upstream dispatch. Automatic routing intersects model capabilities with the adapter contract.
- vLLM can be configured with `kind: vllm`; it uses the OpenAI-compatible adapter path while retaining distinct provider identity for health checks, metrics, and conformance reports.
- OpenAI-compatible `/v1/responses` forwarding uses the same routing, budget, failover, usage, and adapter boundary as chat.
- OpenAI-compatible `/v1/embeddings` forwarding uses the same configured provider path, timeout, retry, concurrency, failover, budget, usage, and adapter boundary as chat.
- OpenAI-compatible `/v1/images/generations` forwarding uses the same configured provider path, timeout, retry, concurrency, failover, budget, and adapter boundary as chat.
- OpenAI-compatible `/v1/audio/speech` forwarding uses the same configured provider path, timeout, retry, concurrency, failover, budget, and adapter boundary as chat.
- OpenAI-compatible multipart `/v1/audio/transcriptions` and `/v1/audio/translations` forwarding use configured provider paths, model rewriting, timeout, retry, concurrency, failover, budget, metrics, and the adapter boundary.
- Automatic chat routing has an end-to-end transient failover test across two candidate providers, including router headers and failover metrics.
- Request-scoped model/provider filters are intersected and enforced for default-model fallback. Capability- or context-ineligible diagnostic candidates are excluded from upstream failover.
- Bearer-token sources are resolved once at startup; missing configured secrets fail closed, and non-loopback binds require auth unless an explicit trusted-gateway override is configured.
- The deterministic `eval-gate`, seeded `configured-eval-gate`, and controlled `runtime-gate` produce distinct artifacts. They identify the classifier/runtime, enforce explicit subgroup sample counts, measure classifier adapter fallbacks, and exercise auth, capability/context rejection, transient failover, outcome metrics, and SSE passthrough through Axum.
- Optimizer runs can emit replayable, secret-safe JSON artifacts with dataset fingerprints, deterministic train/holdout split metadata, baseline/optimized reports, holdout validation reports, selected config patches, replay commands, and rollback guidance.
- LLM-judge classification can be enabled with any configured model, with fallback to the heuristic classifier on judge failure.
- LLM-judge requests use the provider adapter boundary, so configured headers, timeouts, retries, concurrency controls, and native response transforms apply to judge models as well as normal chat models.
- LLM-judge output is schema-validated and exposed through success/fallback/invalid-output counters.
- LLM-judge success, invalid-output fallback, timeout fallback, and native Ollama response transforms are covered with mock upstream tests.
- A `judge-smoke` command validates a live configured judge model and fails CI unless the judge succeeds without fallback or heuristic routing.
- `provider-conformance` and `provider-conformance-matrix` emit schema-v2, config-fingerprinted artifacts for one model or every configured pair. Joint provider/model endpoint declarations receive strict positive and malformed-fixture schema checks; undeclared paths carry explicit skip reasons. Chat requires valid usage and has streaming/tools/JSON/vision probes, while speech/transcription/translation use bounded audio fixtures. Router and reported provider/model versions are recorded.
- Budget accounting can use a lock-protected file ledger so multiple local router processes share request/token/cost reservations before upstream dispatch; an end-to-end two-router test verifies shared request-limit rejection.
- `/metrics/prometheus` exposes request, selection, token, cost, budget, and judge counters in Prometheus text exposition format for fleet scraping.
- Config validation rejects ambiguous provider/model handles and unsafe provider settings.
- Built-in `load-test` and `load-suite` commands can exercise one live HTTP endpoint or the default production endpoint set, emit latency/throughput/error-rate JSON reports, and fail CI on p95 latency or error-rate SLO misses.

## Remaining Work

- Expand native provider adapters beyond Ollama chat and llama.cpp completion where upstreams require request/response transforms, and publish matrix conformance artifacts from real providers.
- Capture and publish live judge-smoke artifacts from actual configured local and hosted models.
- Grow the production eval gate with real traffic and provider-specific traces beyond the current curated 24-example suite.
- Extend the replayable optimizer beyond deterministic search toward learned GEPA-style prompt/program optimization when enough held-out traffic exists.
- Retain successful deployment-gate artifacts per release and extend the staging workload when new endpoint/provider classes are promoted.
- Add recorded upstream conformance fixtures for multipart audio endpoints across real providers.
