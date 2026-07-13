# Router Improvements Audit

Audit date: 2026-07-13 (Pacific/Auckland)

Audited revision: `f9ecd4b` (`main`)

Scope: read-only audit of routing quality, reliability, performance, security, OpenAI compatibility, observability, provider behavior, evaluation, and API-gateway readiness.

## Executive summary

The router has a credible deterministic core and unusually broad offline coverage for a young gateway: routing remains available without an external classifier, automatic and explicit requests have intentionally different failover semantics, context and capability filtering are represented in diagnostics, upstream concurrency permits live through streamed bodies, process budgets reserve atomically, and the public contract is generated as OpenAPI 3.1. All required offline gates pass at the audited revision, including 158 tests and a 50-example eval gate at 94% exact-tier accuracy.

No P0 issue was found. The 23 findings comprise 12 P1, 9 P2, and 2 P3 items. The most immediate implementation themes are:

1. Prevent semantic-cache collisions between materially different conversations.
2. Apply capability and context eligibility to explicit and safety-forced models.
3. Make failover, error, and streaming-usage metrics describe actual outcomes.
4. Make runtime YAML parsing reject fields the generated schema rejects.
5. Normalize local validation and error responses to the OpenAI contract.
6. Harden file-backed budget, cache, and sticky stores against blocking and crash residue.
7. Ensure configured safety redaction covers all forwarded request fields.
8. Stop endpoint/capability defaults from advertising unsupported model-provider combinations.
9. Add configurable ingress resource controls before treating the process as an exposed API gateway.

The recommended sequence is D1-D4 and D6-D8 first because they are correctness or security boundaries; A1 and A2 next for gateway reliability; D5 and D9 for compatibility/provider integrity; then the observability, retry, evaluation, and release-gate work. Live provider claims should remain explicitly unverified until the evidence plan in the last section is run against controlled infrastructure.

## Method and verification

The audit traced all HTTP handlers through request decoding, automatic and explicit selection, classifier/scoring, safety policy, endpoint/capability/context filtering, cache and sticky behavior, budget reservation, provider admission, retries/failover, response forwarding, health recording, metrics/traces, and generated contracts. Findings were checked against tests, local runtime probes, recent commits, and the current source to remove duplicates and already-fixed issues.

### Required command results

| Command | Result | Evidence |
| --- | --- | --- |
| `cargo fmt --check` | PASS | Exit 0; no diff. |
| `cargo clippy --all-targets --all-features -- -D warnings` | PASS | Exit 0; dev profile finished without warnings. |
| `cargo test --locked` | PASS | 158 passed, 0 failed, 0 ignored. |
| `cargo run --quiet --locked -- --config examples/router.yaml validate` | PASS | 7 providers and 7 models. |
| `cargo run --quiet --locked -- --config docs/examples/router.production.yaml validate` | PASS | 2 providers and 2 models. |
| `cargo run --quiet --locked -- --config examples/router.yaml openapi` | PASS | Generated OpenAPI 3.1; structural check found 17 paths. |
| `cargo run --quiet --locked -- --config examples/router.yaml config-schema` | PASS | Generated JSON Schema draft 2020-12 with a closed root object. |
| `cargo run --quiet --locked -- --config examples/router.yaml eval-gate examples/eval.production.jsonl --min-examples 50 --min-accuracy 0.90 --min-domain-accuracy 0.90 --min-model-accuracy 0.95 --min-provider-accuracy 0.95` | PASS | 50 examples; tier 47/50 (0.94), domain 50/50, model 11/11, provider 22/22. |

Additional checks:

- A local elevated-sandbox runtime probe used `examples/router.yaml` on `127.0.0.1:18080`, then verified that no listener remained after shutdown.
- Malformed chat JSON returned HTTP 400 as `text/plain`, not an OpenAI JSON error object.
- `{"model":"auto"}` was accepted with an empty default `messages` array and reached the configured local upstream, which rejected it.
- `/v1/models` returned all seven configured models, but model objects had no `created` field.
- A temporary copy of `examples/router.minimal.yaml` containing an unknown root key still passed `validate`, despite the generated schema setting `additionalProperties: false`.
- `cargo audit --version` could not run because `cargo-audit` is not installed. Dependency advisory status is therefore not established by this audit.

### Coverage map

| Public route | Reviewed path and result |
| --- | --- |
| `GET /health` | `server::health`; static liveness response reviewed. See A2. |
| `GET /openapi.json` | `server::openapi_json` and `openapi`; generated document inspected. See D5 and E3. |
| `POST /v1/router/raw` | Compatibility classifier path, auth middleware, telemetry, and schema reviewed. |
| `POST /v1/router/classify` | Deterministic/adapter classifier boundary and response labels reviewed. |
| `POST /v1/router/multimodel` | Candidate filtering, fallback, score explanation, and telemetry reviewed. |
| `GET /v1/router/providers` | Live checks plus sampled state reviewed. See A2. |
| `POST /v1/router/:provider` | Provider-constrained selection, fallback, and error contract reviewed. |
| `GET /v1/models` | Config projection and OpenAPI schema reviewed. See D5. |
| `POST /v1/chat/completions` | Automatic/explicit routes, capabilities, context, safety, cache, sticky, budgets, retries, streaming, usage, and failover reviewed. See D1-D8. |
| `POST /v1/responses` | Automatic/explicit routes, input inference, safety, cache, budgets, streaming, and failover reviewed. See D1-D5 and D7. |
| `POST /v1/embeddings` | Endpoint selection, budget, adapter forwarding, and response metrics reviewed. See D2, D8, and E3. |
| `POST /v1/images/generations` | Capability selection, explicit path, forwarding, and conformance reviewed. See D2, D8, and E3. |
| `POST /v1/audio/speech` | Automatic audio requirement, explicit path, binary forwarding, and conformance reviewed. See D2, D8, and E3. |
| `POST /v1/audio/transcriptions` | Multipart buffering, automatic/explicit selection, retries, and conformance reviewed. See A1 and E3. |
| `POST /v1/audio/translations` | Multipart buffering, automatic/explicit selection, retries, and conformance reviewed. See A1 and E3. |
| `GET /metrics` | Counter snapshot and budget/cache/sticky/classifier fields reviewed. See D3 and A4. |
| `GET /metrics/prometheus` | Prometheus rendering, labels, token/cost limitations, and escaping reviewed. See D3 and A4. |

Provider registry coverage:

| Provider kind | Adapter path reviewed |
| --- | --- |
| `open_ai_compatible` | Shared OpenAI-compatible request, multipart, retry, health, timeout, header, and permit path. |
| `ollama` | Shared adapter with Ollama profile. |
| `ollama_native` | Native chat transform and response normalization. See D9. |
| `llama_cpp` | Shared adapter with llama.cpp profile. |
| `llama_cpp_native` | Native completion transform and response normalization. See D9. |
| `vllm` | Shared adapter with vLLM profile. |
| `openrouter` | Shared adapter with OpenRouter profile and attribution headers. |
| `cloudflare_ai_gateway` | Shared adapter with Cloudflare AI Gateway profile. |

Other boundaries reviewed: heuristic, LLM-judge, and route-LLM classifiers; learned and configured score components; fallback; provider health penalties; automatic transient failover; strict explicit requests; context windows; endpoint allowlists; semantic cache; sticky routing; process/file budgets; redacted decision JSONL; shadow evaluation; graceful shutdown; config validation/schema; OpenAPI; eval/calibrate/optimize; load tools; provider conformance; judge smoke; CI; and release packaging.

## Verified strengths

- **Deterministic fail-closed routing.** `classifier.rs` and `router.rs` preserve a local heuristic path and configured fallback even without external classifier services.
- **Useful route explanations.** Candidate scores expose capability, domain, cost, latency, health, learned, context, and exclusion components, plus rejected-candidate reasons.
- **Correct automatic-versus-explicit failover split.** Dispatch only advances to the next eligible candidate for automatic requests on transient failures; explicit model requests remain strict.
- **Provider admission is stream-safe.** The configured semaphore permit is retained in `ProviderResponse` until the buffered body or stream is consumed.
- **Context/capability-aware automatic routing.** Automatic chat and Responses estimate the serialized request context and pass inferred requirements into the routing engine.
- **Atomic process budgets.** Validation and reservation occur under one process-ledger mutex; recent regression coverage exercises shared file accounting too.
- **Safer cache/sticky behavior than prior revisions.** Authenticated or behavior-changing top-level requests fail closed for semantic reuse, and sticky routes are written only after a completed selection.
- **Operational contracts exist.** Request IDs, route headers, JSONL traces, JSON and Prometheus metrics, provider health sampling, conformance commands, load commands, OpenAPI, and config schema are present and tested.
- **Clean offline baseline.** Formatting, strict Clippy, all tests, both config examples, contract generation, and the production eval gate pass at this revision.

## Confirmed defects

### D1 — P1: semantic cache can reuse a response across different conversation histories

- **Affected path:** `src/types.rs::OpenAiChatRequest::prompt_text`; `src/server.rs::chat_completions`, `semantic_cache_request_for_route`, and `semantic_cache_safe_for_request`; `src/semantic_cache.rs::best_hit`.
- **Current behavior:** The cache embedding/key is derived from concatenated user/system text. Assistant turns, tool results, roles beyond user/system, and `ChatMessage.extra` are omitted. The safety guard examines only top-level flattened request keys, not message extensions.
- **Evidence:** `context_text` correctly serializes full history for context eligibility, but cache construction receives `route_input`, which starts from `prompt_text`. Cache lookup then compares endpoint, candidate model, and prompt embedding. Existing cache tests cover auth and top-level variants but not distinct assistant/tool histories with the same user/system text.
- **Impact:** With semantic cache enabled, materially different conversations can receive the same cached answer. This is a correctness and potential cross-context disclosure problem, even though authenticated traffic is currently excluded.
- **Recommendation:** Build a canonical cache input from the complete ordered conversation plus behavior-affecting options, or conservatively disable caching whenever assistant/tool turns, message extensions, or unrecognized fields are present.
- **Acceptance criteria:** Distinct histories never collide; canonicalization is deterministic; tool content and roles affect the key; auth and unknown-field fail-closed behavior remains; cache diagnostics explain why a request was ineligible.
- **Expected verification:** Unit tests for same prompt/different assistant history, tool result, role, and message extension; property test for canonical determinism; end-to-end two-request cache probe; `cargo test --locked`.

### D2 — P1: capability and context eligibility has explicit, forced, and Responses-format gaps

- **Affected path:** Explicit branches in `src/server.rs` for chat, Responses, embeddings, images, speech, and multipart audio; `enforce_safety_route`; `eligible_route_models`; model eligibility helpers in `src/router.rs` and `src/types.rs`.
- **Current behavior:** Automatic chat/Responses infer some required capabilities and use full serialized context estimates. Explicit branches validate only model existence and endpoint support, then estimate selected requests from prompt text. `safety.force_model` rewrites the chosen model/provider without rechecking the forced model's endpoint, capabilities, or context window. Automatic Responses looks for Chat Completions' top-level `response_format`, not Responses' `text.format`, when inferring JSON capability.
- **Evidence:** `chat_completions` passes `required_capabilities` and `estimate_tokens(request.context_text())` only in the automatic branch; the explicit branch returns `vec![model]` with `estimate_tokens(prompt)`. Responses follows the same branch shape. `OpenAiResponsesRequest::required_capabilities` calls `response_format_requires_json(self.extra.get("response_format"))`, while the official OpenAI SDK's generated Responses request type places structured-output configuration under `text`: <https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_create_params.py>. `enforce_safety_route` directly assigns `route.model` and `route.provider`.
- **Impact:** Strict explicit requests can forward tools, vision, JSON, audio, or oversized contexts to an incompatible model, and a safety policy can force the same invalid route. The eventual upstream error is less deterministic and can be expensive or lossy.
- **Recommendation:** Keep explicit selection strict but validate that exact model locally against endpoint, inferred capabilities, and conservative full-context tokens. Revalidate a safety-forced model before dispatch and fail closed with a structured error.
- **Acceptance criteria:** Incompatible explicit or forced requests produce a local OpenAI-shaped 4xx/5xx as appropriate, never call upstream, and report the precise eligibility reason; Responses `text.format` requires JSON capability; compatible explicit requests remain unchanged.
- **Expected verification:** Per-endpoint tests for capability and context rejection; Responses `text.format` versus Chat `response_format` inference tests; forced-model regression tests; a counting mock asserting zero upstream calls; full test suite.

### D3 — P1: failover, upstream-error, and token metrics misdescribe final outcomes

- **Affected path:** All `dispatch_*` loops and `upstream_response` in `src/server.rs`; JSON and Prometheus metric rendering.
- **Current behavior:** After one transient candidate, any final HTTP response increments `failover_successes`, including a 4xx. Final upstream HTTP failures are not consistently counted as `upstream_errors`; transport/body failures are. Streaming bodies are passed through without parsing usage, while metric help explicitly describes token counters as buffered usage.
- **Evidence:** Dispatch increments `failover_successes` solely when `failovers > 0`, before status success is checked. `upstream_response` parses usage only in the non-streaming buffered branch. Stream mapping records provider health on a body error but does not increment the main upstream-error counter or parse terminal usage events.
- **Impact:** Operators can see successful failover during a failed request, undercount upstream failures, and miss streaming tokens/cost. SLO, cost, provider, and routing-policy decisions can therefore be based on misleading data.
- **Recommendation:** Define metrics by final logical request outcome and individual upstream attempt outcome. Count failover success only for successful final responses, count terminal HTTP failures by class/provider/model, and add streaming usage accounting without breaking byte passthrough.
- **Acceptance criteria:** Counters distinguish attempts, transient retries, failovers, final success, final 4xx/5xx, transport error, and stream-body error; streaming usage is recorded when the provider emits it; labels remain bounded.
- **Expected verification:** Mock sequences `503→200`, `503→400`, final `503`, transport error, malformed stream, and SSE usage; exact JSON/Prometheus assertions; load smoke ensuring passthrough remains incremental.

### D4 — P1: runtime YAML accepts fields that the generated schema forbids

- **Affected path:** Configuration structs and `RouterConfig::from_path` in `src/config.rs`; closed objects emitted by `src/config_schema.rs`; `validate` CLI.
- **Current behavior:** Config structs do not use `serde(deny_unknown_fields)`, so unknown YAML keys are ignored. The generated schema declares `additionalProperties: false` for the root and most nested objects.
- **Evidence:** A temporary minimal config with `unknown_router_field: silently_accepted` passed validation as one provider/one model. `from_path` calls `serde_yaml::from_str` and semantic validation; no unknown-key pass exists.
- **Impact:** Misspelled security, timeout, retry, budget, capability, or routing settings can silently fall back to defaults while schema-aware tooling says the same config is invalid.
- **Recommendation:** Make deserialization and schema strictness share one source of truth. Prefer unknown-field rejection with intentional aliases for compatibility, and surface a path-aware CLI error.
- **Acceptance criteria:** Unknown root and nested keys fail `validate`; documented aliases still work; free-form maps such as headers and score maps remain open; generated schema matches runtime behavior.
- **Expected verification:** Table-driven tests for every config object level and allowed open maps; example/prod validation; schema generation; a runtime CLI regression probe.

### D5 — P1: local request rejection and model listing are not consistently OpenAI-compatible

- **Affected path:** Axum `Json` extractors on public JSON routes, request types in `src/types.rs`, `server::models`, and matching schemas/responses in `src/openapi.rs`.
- **Current behavior:** Extractor failures return Axum plain text. Chat `messages` and Responses/embeddings `input` default when absent, allowing incomplete requests to reach upstream. `/v1/models` omits the standard `created` field and adds router extensions. Some rejection statuses/content types are not modeled, and the OpenAPI `ModelCapabilities` schema omits runtime `supported_endpoints`.
- **Evidence:** Runtime probes returned plain-text HTTP 400 for malformed JSON and forwarded `{"model":"auto"}` with empty messages. `/v1/models` objects contained `id`, `object`, `owned_by`, and extensions but no `created`. The OpenAI model object documents `created` as a field: <https://platform.openai.com/docs/api-reference/models/object>. `src/types.rs::ModelCapabilities` serializes `supported_endpoints`, while its hand-written schema in `src/openapi.rs` lists only the older capability fields.
- **Impact:** OpenAI SDKs and gateway clients cannot rely on one error envelope, malformed requests consume upstream capacity, and strict model-object consumers may reject the list response.
- **Recommendation:** Introduce a shared OpenAI-shaped extractor rejection and validation layer, require endpoint-mandatory fields locally, add a stable `created` value or explicitly version a documented extension strategy, and model all local failures.
- **Acceptance criteria:** Malformed/missing/invalid requests return JSON `{error:{message,type,param,code}}` with correct status/content type and request ID; zero upstream calls occur; model objects satisfy the documented base shape; extensions remain additive; generated schemas include every serialized public field.
- **Expected verification:** Black-box probes with official OpenAI SDKs plus raw HTTP cases for malformed JSON, missing fields, wrong content type, oversized body, and model list deserialization; serialization-to-schema parity tests and OpenAPI response assertions.

### D6 — P1: file-backed state blocks async workers and is vulnerable to stale locks and torn writes

- **Affected path:** `src/accounting.rs::FileBudgetLedger`, `src/semantic_cache.rs::FileSemanticCacheStore`, and `src/sticky.rs::FileStickyRoutingStore`, called from async request handlers.
- **Current behavior:** File operations and lock acquisition are synchronous; contention loops use `std::thread::sleep`. Locks are empty create-new files removed only by `Drop`, with no owner, lease, or stale recovery. State is overwritten directly with `fs::write`.
- **Evidence:** Each store performs read-modify-write under its own `FileLock`; all three implementations use 10 ms thread sleeps and direct writes. Only decision-log appends are moved through `spawn_blocking`.
- **Impact:** File mode can block Tokio workers under contention, a crashed process can wedge later requests until every lock timeout, and a process/filesystem failure can leave corrupt state. Budget failure semantics are especially sensitive.
- **Recommendation:** Move file stores behind a bounded async interface or dedicated blocking worker, add lease/owner-aware stale-lock recovery, and persist with same-directory temp file plus flush/sync/rename. Define fail-closed behavior per store.
- **Acceptance criteria:** No blocking filesystem call runs on request workers; crash residue is recoverable without unsafe concurrent writers; writes are atomic; corrupted budget state rejects conservatively; cache/sticky corruption degrades safely with diagnostics.
- **Expected verification:** Multi-thread contention test, killed-writer/stale-lock test, injected partial-write test, Tokio worker starvation test, and cross-process budget limit test.

### D7 — P1: safety redaction does not cover all data forwarded upstream

- **Affected path:** `enforce_safety_for_chat`, `enforce_safety_for_responses`, `redact_chat_request`, classifier prompt extraction in `src/types.rs`, and flattened request/message fields.
- **Current behavior:** Chat redaction walks only `message.content`; it does not redact `ChatMessage.extra` such as tool calls/function arguments or top-level extras such as metadata. Responses redaction walks only `input`. Classification text similarly omits many extension fields.
- **Evidence:** The redaction helper iterates messages and calls `redact_value_strings` only on `content`. The existing end-to-end safety test covers a secret in simple user content, not tool arguments, metadata, or other forwarded strings.
- **Impact:** When operators enable redact safety, sensitive values can still leave the process through valid OpenAI extension fields, creating a false security guarantee.
- **Recommendation:** Define a field-aware redaction contract for all forwardable strings, including tool arguments and metadata, while preserving control/schema fields. Classify over the same normalized security-relevant view.
- **Acceptance criteria:** Documented sensitive fields are redacted or rejected before dispatch; non-sensitive schemas remain valid; logs/traces and route inputs use the redacted view; unsupported ambiguous shapes fail closed.
- **Expected verification:** Capturing-upstream tests for tool calls, function arguments, assistant/tool messages, metadata, URLs, and nested arrays/objects; snapshot of forwarded payload and decision trace.

### D8 — P1: endpoint and capability defaults can advertise unsupported combinations

- **Affected path:** `ProviderConfig` endpoint defaults and `ModelCapabilities::supports_endpoint` in `src/types.rs`; `supported_model_ids` in `src/server.rs`; example and production configs.
- **Current behavior:** Optional provider endpoint paths default to configured values, while `model.capabilities.supported_endpoints: null` means all provider-configured endpoints. A model/provider pair can therefore become eligible for Responses, embeddings, images, speech, or audio without an explicit model allowlist or proof.
- **Evidence:** `examples/router.minimal.yaml` omits optional paths/capabilities but implicitly exposes them. Models in the larger examples frequently omit endpoint allowlists. Routing eligibility trusts these defaults; conformance evidence is not required at startup.
- **Impact:** Automatic routing can select an endpoint a model does not implement, and `/v1/*` availability can look broader than actual provider/model support.
- **Recommendation:** Make endpoint support explicit and provider-kind-aware. Treat absent model allowlists conservatively for non-chat endpoints, or require a validated provider capability catalog/conformance artifact.
- **Acceptance criteria:** A provider path alone never makes an unverified model eligible; config validation reports ambiguous combinations; examples enumerate supported endpoints; diagnostics explain endpoint exclusions.
- **Expected verification:** Config matrix across provider kinds and all endpoints; automatic and explicit eligibility tests; conformance artifact import test; example config validation.

### D9 — P2: native adapters silently discard requested OpenAI features

- **Affected path:** `ollama_chat_body`, `ollama_chat_response`, `llama_cpp_completion_body`, and native adapter dispatch in `src/provider.rs`; capability eligibility in `src/router.rs`.
- **Current behavior:** Ollama native converts each message to text, forwards only `options`, and forces `stream: false`. llama.cpp native flattens messages into one prompt, preserves only token/temperature controls, and also forces non-streaming. Router model metadata is not constrained by adapter transformation support.
- **Evidence:** Images and message extensions disappear during text conversion; tools, structured output, stream, and most request controls are not forwarded. Tests exercise simple text transformation, not rejection or preservation of advanced features.
- **Impact:** If a native-backed model is tagged with tools, vision, JSON, or streaming support, the router can select it and silently alter the request instead of rejecting an unsupported contract.
- **Recommendation:** Declare adapter-level feature support and intersect it with provider/model capabilities. Reject unsupported explicit requests, exclude them from automatic routing, and only transform features with defined native mappings.
- **Acceptance criteria:** No accepted feature is silently dropped; diagnostics name adapter exclusions; supported controls round-trip; streaming is either implemented or rejected before upstream.
- **Expected verification:** Adapter contract tests for every OpenAI field/capability, capture tests for transformed bodies, and native-provider conformance probes.

## Architectural improvements

### A1 — P1: add configurable ingress resource controls before network exposure

- **Affected path:** `server::app`, request middleware/config, multipart parsing, auth identity, and all public proxy endpoints.
- **Current behavior:** The app layers request context/auth, tracing, and permissive CORS. It has no router-configured in-flight request limit, per-credential rate limit, explicit JSON/multipart size policy, slow-client policy, load shedding, or endpoint-level timeout. Multipart fields are buffered into memory and cloned for retryable upstream dispatch.
- **Evidence:** The layer stack contains only request-context middleware, `TraceLayer`, and `CorsLayer::permissive`. No request-limit middleware/config appears. Multipart parsing calls `field.bytes()` and stores every part in memory.
- **Impact:** A network-exposed process can exhaust memory, worker time, provider queues, or budget capacity before provider concurrency limits apply. Static bearer tokens do not provide tenant-level quotas.
- **Recommendation:** Add opt-in/configurable total and endpoint-specific body limits, global and credential-scoped admission/rate limits, bounded queue/load shedding, and streaming-aware header/body idle deadlines. Keep localhost defaults ergonomic.
- **Acceptance criteria:** Limits are configurable and documented; rejected requests return OpenAI-shaped 413/429/503 errors with request IDs; streaming responses are not killed by a total request deadline; provider permits remain the upstream boundary.
- **Expected verification:** Oversized JSON/multipart, slow upload, connection flood, per-token fairness, queue saturation, and long-stream tests; load-suite comparison before/after.

### A2 — P1: separate liveness/readiness and make sampled health fresh and concurrent

- **Affected path:** `/health`, `/v1/router/providers`, `start_provider_health_sampler` in `src/server.rs`; `ProviderHealthStore`; health penalties in `src/router.rs`.
- **Current behavior:** `/health` always reports OK. Provider status and the sampler check providers serially. Observations include a timestamp, but scoring applies stored penalties without an age/decay rule; a transient failure can continue to demote a provider if sampling is disabled or delayed.
- **Evidence:** Both loops await each provider one after another. `ProviderHealthObservation` stores `observed_unix_seconds`, while the score helper reads `health_penalty` without freshness logic.
- **Impact:** Readiness cannot signal that no provider/fallback is usable, checks take the sum of provider latencies, and stale degradation can distort routing indefinitely.
- **Recommendation:** Preserve cheap liveness, add readiness based on configured routing viability, run bounded provider checks concurrently, and introduce TTL/decay plus circuit-breaker half-open probes.
- **Acceptance criteria:** Liveness is dependency-free; readiness fails only when no safe route exists; status latency is bounded near the slowest check rather than the sum; stale observations stop affecting scores predictably.
- **Expected verification:** Fake-clock health tests, multi-provider timing test, readiness matrix, circuit open/half-open/recovery tests, and provider outage integration test.

### A3 — P2: retry policy needs `Retry-After`, jitter, and separate timeout phases

- **Affected path:** Retry loops, `is_transient_status`, `backoff`, and `reqwest::Client` setup in `src/provider.rs`; timeout/retry config.
- **Current behavior:** 408, 429, and 5xx responses use a fixed 100 ms exponential delay. `Retry-After` is ignored and no jitter is applied. One configured request timeout does not distinguish connect, response-header, or streaming idle time.
- **Evidence:** Every adapter retry arm calls `backoff(attempt)`; `backoff` is exactly `100 * 2^attempt` milliseconds. Response headers are not consulted before sleeping.
- **Impact:** Concurrent routers can synchronize retry storms, providers' requested cooldowns are violated, and a timeout suitable for unary calls can terminate legitimate streams or wait too long on connection setup.
- **Recommendation:** Honor bounded `Retry-After`, add decorrelated/full jitter, separate connect/header and stream-idle policies, and expose retry delay/reason metrics.
- **Acceptance criteria:** Retry sleeps obey provider guidance within configured caps; clients desynchronize; strict explicit calls still retry only within their configured provider policy and never fail over; stream idle timeout is distinct.
- **Expected verification:** Paused-time unit tests for seconds/date `Retry-After`, cap/jitter bounds, connect/header/idle failures, and retry/failover interaction.

### A4 — P2: expose latency, queue, and retry distributions

- **Affected path:** `Metrics`, `record_provider_dispatch_*`, provider admission, route handlers, and Prometheus rendering in `src/server.rs`/`src/provider.rs`.
- **Current behavior:** Metrics are primarily counters plus token/cost totals. There are no request, route-decision, upstream, provider-queue, or retry-delay histograms; last sampled latency exists only in provider status state.
- **Evidence:** Metric rendering enumerates counts and totals, while request `Instant` values are used for health observations rather than exported distributions.
- **Impact:** Operators cannot establish latency SLOs, distinguish router overhead from provider latency, size concurrency, or identify queue saturation and retry amplification.
- **Recommendation:** Add bounded histograms for end-to-end, routing, queue wait, upstream headers/body, and retry delay, labeled only by endpoint/provider/model/outcome where cardinality is controlled.
- **Acceptance criteria:** Streaming metrics distinguish time-to-first-byte and stream duration; queue time is visible; histogram labels are bounded; existing counters remain backward-compatible.
- **Expected verification:** Deterministic mock-latency tests, Prometheus exposition assertions, cardinality audit, and load-suite dashboard check.

### A5 — P2: decision traces and shadow evaluation need bounded lifecycle management

- **Affected path:** `src/telemetry.rs::record_route`, shadow-eval task spawning in `src/server.rs`, and graceful shutdown.
- **Current behavior:** Each traced automatic request awaits a `spawn_blocking` open/append operation. Shadow evaluation creates detached tasks. There is no bounded writer queue, rotation/retention, dropped-record metric, or shutdown drain for either path.
- **Evidence:** `record_route` awaits `spawn_blocking(append_jsonl_blocking)` per request. Shadow dispatch uses `tokio::spawn`; shutdown only drains the HTTP server within its timeout.
- **Impact:** Slow filesystems add request latency, unbounded background work can amplify overload, traces can consume disk indefinitely, and shutdown can lose pending evaluation records.
- **Recommendation:** Use bounded asynchronous queues with dedicated writers, rotation/retention, explicit overflow policy and metrics, and graceful flush/drain within the shutdown budget.
- **Acceptance criteria:** Request latency is decoupled from normal disk append latency; memory is bounded; overflow is observable; files rotate safely; shutdown reports flushed/dropped counts.
- **Expected verification:** Slow-disk simulation, queue saturation, rotation, disk-full, and timed shutdown-drain tests.

### A6 — P2: define budget semantics for actual upstream spend and tenancy

- **Affected path:** `src/accounting.rs`, `reserve_budget` and dispatch/failover/shadow/classifier calls in `src/server.rs`, auth context, and usage metrics.
- **Current behavior:** A request reserves one conservative estimate against the initially selected/cache-hit model. It is not reconciled with actual usage, failures, retries, failover model changes, shadow calls, or classifier/judge calls. Limits are process/file global rather than credential-scoped.
- **Evidence:** `BudgetReservation::new` charges estimated input plus requested output once; the ledger supports only `reserve` and `snapshot`, with no settlement/refund/upstream-attempt record.
- **Impact:** The feature is useful as a conservative logical-request guard but cannot serve as authoritative spend control, especially with failover or multi-tenant network exposure.
- **Recommendation:** Explicitly choose and document logical-request reservation versus actual-upstream spend accounting. If spend control is intended, add reservation IDs, settlement/reconciliation, attempt classes, and optional credential/tenant scopes.
- **Acceptance criteria:** Every billable upstream path has a defined accounting treatment; no ambiguous double charge/refund; failure policy is documented; per-tenant limits cannot affect unrelated tokens.
- **Expected verification:** Scenarios for cache hit, retry, failover to different price, failure before/after headers, streaming usage, shadow evaluation, judge call, restart, and concurrent settlement.

## Evaluation and model-coverage gaps

### E1 — P1: the production eval gate does not exercise the configured classifier or HTTP runtime

- **Affected path:** `src/eval.rs::eval_gate`/`evaluate_with_heuristic`, `examples/eval.production.jsonl`, CI, and all server/provider boundaries.
- **Current behavior:** The gate always constructs `HeuristicClassifier`, regardless of configured LLM-judge or route-LLM backend, and calls `RoutingEngine` directly. It does not cover request inference, HTTP validation, safety, auth, budgets, cache/sticky, provider adapters, retry/failover, streaming, or response compatibility.
- **Evidence:** `eval_gate` calls `evaluate_with_heuristic`; that helper constructs the heuristic and invokes the in-memory engine. The 50-example corpus has 50 domain labels but only 11 model and 22 provider expectations.
- **Impact:** A passing 94% gate is meaningful for deterministic tier/domain routing but can coexist with regressions in the configured production classifier or gateway data path.
- **Recommendation:** Retain the fast deterministic gate and add separate configured-classifier and HTTP scenario gates. Build holdouts from redacted decision traces and require enough examples per supported domain/model/provider before enforcing subgroup thresholds.
- **Acceptance criteria:** Reports identify which classifier/runtime was tested; configured adapter failure/fallback is measured; HTTP scenarios exercise capabilities/context and upstream outcomes; subgroup minimum sample sizes are explicit.
- **Expected verification:** Heuristic and configured-backend artifacts, seeded holdout split, mock-provider HTTP suite, and a credentialed judge smoke where configured.

### E2 — P1: CI and release builds do not enforce the repository's own quality gates

- **Affected path:** `.github/workflows/ci.yml`, `.github/workflows/release.yml`, example/production configs, OpenAPI/schema output, and eval gate.
- **Current behavior:** CI runs formatting, build, one example validation, and tests on three operating systems. It does not run strict Clippy, production-config validation, contract generation assertions, the 50-example eval gate, or a dependency advisory scan. Release jobs build/package without depending on a comprehensive validation job.
- **Evidence:** The audited local commands all pass, but none of the missing gates appears in CI/release YAML. `cargo-audit` is also absent from the local toolchain.
- **Impact:** A merge or tagged binary can regress lint cleanliness, production config, public contract generation, routing accuracy, or known dependency vulnerabilities despite a green workflow.
- **Recommendation:** Add one Linux quality job for strict Clippy, both configs, generated-contract structural/diff checks, eval gate, and a pinned advisory scanner; make release depend on it while retaining cross-platform build/test coverage.
- **Acceptance criteria:** The exact audited commands are enforced; generated artifacts are deterministic or structurally asserted; eval JSON is uploaded on failure; release cannot publish when a gate fails; advisory exceptions are explicit and expiring.
- **Expected verification:** CI run with intentional failures for each gate, successful main/tag run, artifact inspection, and release dependency check.

### E3 — P2: provider conformance is not model-capability-aware and validates weak response shapes

- **Affected path:** `src/conformance.rs::run_endpoint_conformance`, endpoint validators, model endpoint allowlists, and `provider-conformance-matrix`.
- **Current behavior:** The matrix probes an endpoint whenever the provider path exists, regardless of `model.capabilities.supported_endpoints`. Responses accepts any JSON; speech/audio accept any non-empty body; chat can pass without usage. Advanced streaming, tool, structured-output, and error-shape behavior is not probed.
- **Evidence:** Each endpoint's `configured` flag checks only `provider.*_path.is_some()`. Validators range from schema fragments to `validate_non_empty_response`.
- **Impact:** Supported models can be reported failed for intentionally excluded endpoints, while incompatible or malformed responses can pass. The artifact is insufficient as a capability source of truth.
- **Recommendation:** Generate probes from the intersection of provider paths, model allowlists, adapter features, and declared capabilities; validate OpenAI shapes/content types and add feature-specific cases.
- **Acceptance criteria:** Skips are explicit and justified; each declared endpoint has positive and negative schema checks; streaming/tools/JSON/vision/audio fixtures are represented; artifacts include provider/model/version/config hash.
- **Expected verification:** Deterministic mock matrix for every adapter, schema-negative fixtures, and live matrix runs from the evidence plan below.

### E4 — P3: performance and support claims need versioned artifacts

- **Affected path:** README/production-readiness claims, `load-test`, `load-suite`, and release evidence.
- **Current behavior:** Load and conformance commands exist, but no checked, revision-linked baseline artifact establishes throughput, latency, resource use, or supported provider/model/endpoint combinations for this revision.
- **Evidence:** The repository contains load tooling and documentation commands, but this audit found no release-bound result artifact or threshold gate. Live load was outside the available infrastructure.
- **Impact:** Operators cannot compare regressions or know which environment substantiates readiness/performance claims.
- **Recommendation:** Publish sanitized, versioned benchmark and conformance artifacts with hardware, config hash, provider versions, concurrency, payload mix, and pass thresholds.
- **Acceptance criteria:** Every release links to reproducible artifacts; claims name their environment and percentile; regressions have thresholds; external provider variability is separated from router overhead.
- **Expected verification:** Controlled local mock-provider baseline, sustained deployed load suite, repeated runs with confidence intervals, and artifact-schema validation.

## Live-provider evidence still required

These are evidence gaps, not claims that a provider is broken. Credentials, external network access, deployed infrastructure, and all configured local inference servers were not available as a controlled test matrix during this audit.

### L1 — P2: current provider/model endpoint support is unverified live

- **Affected path:** All seven configured model/provider pairs in `examples/router.yaml`; `provider-conformance-matrix`; endpoint/capability configuration.
- **Current behavior:** Offline validation proves configuration consistency, not that each advertised provider/model/endpoint combination responds correctly now.
- **Evidence:** The live conformance matrix was not run because it can contact OpenRouter/Cloudflare endpoints and assumes local Ollama/llama.cpp/vLLM services and credentials that were not established for this audit.
- **Impact:** Endpoint defaults and provider drift can remain undetected until traffic arrives.
- **Recommendation:** Run conformance in a controlled credentialed environment using explicit test models, budgets, and redacted artifacts.
- **Acceptance criteria:** Every advertised combination has a timestamped pass/fail/skip reason and provider/model version; failures block promotion or remove eligibility.
- **Expected verification:** `provider-conformance-matrix` plus adapter-specific feature probes and sanitized artifact review.

### L2 — P2: judge/route-LLM production behavior is unverified live

- **Affected path:** LLM-judge and route-LLM classifier adapters, configured judge model, fallback counters, and `judge-smoke`.
- **Current behavior:** Schema validation and fallback logic are covered offline, but no live configured classifier was available; the audited example uses the heuristic backend.
- **Evidence:** Running `judge-smoke` would require a configured accessible judge model and credentials. Neither was established.
- **Impact:** Latency, output validity, rate limits, fallback frequency, and classification accuracy for the advanced backend remain unknown.
- **Recommendation:** Gate each production classifier configuration with smoke plus a labeled holdout and failure-injection run.
- **Acceptance criteria:** Judge succeeds without fallback at the required rate/latency; invalid/timeout responses fail closed; credentials and payloads remain redacted.
- **Expected verification:** `judge-smoke` artifact, configured-backend eval, timeout/invalid JSON/429 probes, and fallback metric assertions.

### L3 — P2: streaming, cancellation, usage, and retry behavior is unverified across real providers

- **Affected path:** Streaming chat/Responses, provider semaphore lifetime, retry policy, disconnect handling, usage metrics, and graceful shutdown.
- **Current behavior:** Unit/integration tests cover local mock paths, but provider-specific SSE framing, terminal usage, disconnect semantics, and `Retry-After` behavior were not observed live.
- **Evidence:** No controlled real-provider stream fixtures or credentials were available, and the current conformance command does not probe these behaviors.
- **Impact:** Production differences may cause leaked permits, missing usage, delayed cancellation, or incorrect error accounting despite mock success.
- **Recommendation:** Add a credentialed stream conformance suite with short outputs, deliberate client cancellation, forced throttling, and shutdown during active streams.
- **Acceptance criteria:** Permit release, cancellation propagation, byte-for-byte passthrough, terminal usage, error counters, and shutdown bounds are proven per provider profile.
- **Expected verification:** Timed stream/cancel probes, 429 with `Retry-After`, mid-body close, usage reconciliation, and post-test concurrency/health inspection.

### L4 — P3: sustained load, shared-file behavior, and large multipart handling lack deployment proof

- **Affected path:** `load-suite`, Tokio runtime, provider concurrency queues, file budget/cache/sticky stores, multipart audio, and metrics.
- **Current behavior:** Offline tests are short-lived and process-local. No sustained multi-process or large-upload workload was run against a deployed target.
- **Evidence:** The audit had no safe target, workload budget, audio fixtures, or multi-instance shared filesystem environment.
- **Impact:** Tail latency, memory high-water marks, queue fairness, lock contention, and restart recovery remain unknown.
- **Recommendation:** Establish a repeatable staging workload covering mixed unary/stream/multipart traffic and multiple router processes sharing supported state backends.
- **Acceptance criteria:** Defined p95/p99, RSS, error, queue, and recovery thresholds pass for a sustained window; budget limits remain atomic; uploads remain bounded; artifacts are revision-linked.
- **Expected verification:** `load-suite` plus resource telemetry, multi-process contention, rolling restart, corrupted/stale-lock recovery, and large multipart cases.

## False positives and already-fixed items removed from the backlog

The following were explicitly checked and are not findings at `f9ecd4b`:

- Provider concurrency permits are retained through response-body or stream consumption.
- Process budget validation/reservation is atomic; shared file accounting has regression coverage.
- Automatic chat and Responses use full serialized request context for context-window eligibility.
- Sticky routes are recorded only after successful selection/response handling, not before dispatch.
- Semantic cache is disabled for authenticated requests and unknown behavior-changing top-level variants.
- Unknown `router-*` automatic policies are rejected instead of silently mapping to a default.
- SIGTERM participates in graceful shutdown in addition to Ctrl-C.
- Explicit model requests intentionally do not fail over; that strictness matches the engineering contract.
- Safety routing intentionally applies to automatic router requests; explicit safety bypass was not treated as a defect.
- Permissive CORS alone is not classified as an exploitable credential issue because bearer tokens are not ambient browser credentials. It is included only as part of configurable ingress posture in A1.
- Provider base URLs are operator configuration rather than request-controlled SSRF input.
- Transparent failover after streamed response bytes have been committed was not proposed; the actionable gap is correct observability/cancellation semantics.

## Prioritized execution plan

1. **Correctness/security boundary:** D1, D2, D4, D7, D8. Add rejection/eligibility tests before changing behavior.
2. **State and gateway resilience:** D6, A1, A2. Decide file-backend support and ingress defaults before implementation.
3. **Outcome integrity and compatibility:** D3, D5, D9. Define metric and OpenAI error contracts first.
4. **Operational depth:** A3, A4, A5, A6. Keep labels/queues bounded and semantics explicit.
5. **Evidence gates:** E1, E2, E3, E4. Preserve the fast heuristic gate while adding independent runtime/provider gates.
6. **Controlled live proof:** L1-L4. Use test credentials, budgets, sanitized artifacts, and a revision-linked staging environment.

No source, configuration, test, or deployment-document changes are part of this audit.
