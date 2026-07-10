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

- Current endpoints are implemented: `/v1/router/classify`, `/v1/router/multimodel`, `/v1/router/raw`, and `/v1/router/{provider}`.
- OpenAI-compatible chat proxying works for local Ollama through a configured alias.
- OpenAI-compatible chat forwarding is isolated behind a provider adapter registry with explicit provider kinds for generic OpenAI-compatible services, Ollama, llama.cpp, vLLM, OpenRouter, and Cloudflare AI Gateway.
- Native Ollama `/api/chat` can be configured with `kind: ollama_native`; the adapter rewrites OpenAI chat requests into Ollama chat payloads and transforms native Ollama responses back into OpenAI-compatible chat completions with usage accounting.
- Native llama.cpp `/completion` can be configured with `kind: llama_cpp_native`; the adapter rewrites OpenAI chat requests into llama.cpp completion prompts and transforms native completion responses back into OpenAI-compatible chat completions with usage accounting.
- vLLM can be configured with `kind: vllm`; it uses the OpenAI-compatible adapter path while retaining distinct provider identity for health checks, metrics, and conformance reports.
- OpenAI-compatible `/v1/responses` forwarding uses the same routing, budget, failover, usage, and adapter boundary as chat.
- OpenAI-compatible `/v1/embeddings` forwarding uses the same configured provider path, timeout, retry, concurrency, failover, budget, usage, and adapter boundary as chat.
- OpenAI-compatible `/v1/images/generations` forwarding uses the same configured provider path, timeout, retry, concurrency, failover, budget, and adapter boundary as chat.
- OpenAI-compatible `/v1/audio/speech` forwarding uses the same configured provider path, timeout, retry, concurrency, failover, budget, and adapter boundary as chat.
- OpenAI-compatible multipart `/v1/audio/transcriptions` and `/v1/audio/translations` forwarding use configured provider paths, model rewriting, timeout, retry, concurrency, failover, budget, metrics, and the adapter boundary.
- Automatic chat routing has an end-to-end transient failover test across two candidate providers, including router headers and failover metrics.
- Request-scoped model/provider filters are intersected and enforced for default-model fallback. Capability- or context-ineligible diagnostic candidates are excluded from upstream failover.
- Eval, eval-gate, calibration, and scoring optimization commands exist for deterministic router tuning; `examples/eval.production.jsonl` provides a 24-example production gate with minimum tier/domain accuracy thresholds.
- Optimizer runs can emit replayable, secret-safe JSON artifacts with dataset fingerprints, deterministic train/holdout split metadata, baseline/optimized reports, holdout validation reports, selected config patches, replay commands, and rollback guidance.
- LLM-judge classification can be enabled with any configured model, with fallback to the heuristic classifier on judge failure.
- LLM-judge requests use the provider adapter boundary, so configured headers, timeouts, retries, concurrency controls, and native response transforms apply to judge models as well as normal chat models.
- LLM-judge output is schema-validated and exposed through success/fallback/invalid-output counters.
- LLM-judge success, invalid-output fallback, timeout fallback, and native Ollama response transforms are covered with mock upstream tests.
- A `judge-smoke` command validates a live configured judge model and fails CI unless the judge succeeds without fallback or heuristic routing.
- `provider-conformance` and `provider-conformance-matrix` commands record adapter artifacts for one model or every configured model/provider pair and fail CI unless chat output satisfies the OpenAI-compatible chat-completion contract. Matrix conformance also exercises every configured Responses, embeddings, image, speech, transcription, and translation provider path.
- Budget accounting can use a lock-protected file ledger so multiple local router processes share request/token/cost reservations before upstream dispatch; an end-to-end two-router test verifies shared request-limit rejection.
- `/metrics/prometheus` exposes request, selection, token, cost, budget, and judge counters in Prometheus text exposition format for fleet scraping.
- Config validation rejects ambiguous provider/model handles and unsafe provider settings.
- Built-in `load-test` and `load-suite` commands can exercise one live HTTP endpoint or the default production endpoint set, emit latency/throughput/error-rate JSON reports, and fail CI on p95 latency or error-rate SLO misses.

## Remaining Work

- Expand native provider adapters beyond Ollama chat and llama.cpp completion where upstreams require request/response transforms, and publish matrix conformance artifacts from real providers.
- Capture and publish live judge-smoke artifacts from actual configured local and hosted models.
- Grow the production eval gate with real traffic and provider-specific traces beyond the current curated 24-example suite.
- Extend the replayable optimizer beyond deterministic search toward learned GEPA-style prompt/program optimization when enough held-out traffic exists.
- Add published sustained load-suite artifacts from real deployment targets, including provider queue behavior and production external-counter tests.
- Add recorded upstream conformance fixtures for multipart audio endpoints across real providers.
