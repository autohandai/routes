# Credentialed staging evidence

Controlled mock evidence proves router behavior, but it cannot prove current external or self-hosted provider behavior. The `Staging live evidence` workflow runs only on a self-hosted runner labeled `router-staging` in the protected `router-staging` GitHub environment. That runner must have network access to every provider in the selected config and narrowly scoped test credentials with provider-side spend limits.

## Provider/model promotion gate

Run the workflow against the exact candidate revision and staging config. It creates a schema-v2 `provider-conformance-matrix` artifact and immediately evaluates it with `provider-promotion-gate`.

Promotion fails when any of these conditions is true:

- the artifact is stale, from a different redacted config fingerprint, malformed, or duplicated;
- a configured provider/model pair is absent or an unconfigured pair appears;
- provider or model version evidence is not reported;
- an advertised endpoint is skipped, fails, or lacks positive and negative schema proof;
- an advertised streaming, tools, JSON, vision, or audio feature is skipped or fails.

Unadvertised endpoints/features remain explicit `skip` checks with reasons and do not block promotion. `--allow-unreported-versions` exists only for local diagnosis; the staging workflow intentionally does not use it. Both the raw matrix and promotion report are uploaded on every run, including failures. Provider failures therefore block promotion until the config removes that eligibility or a fresh run passes.

Artifacts must contain no credentials. The conformance config fingerprint is calculated after bearer tokens, inline provider keys, and private header values are removed.

## Classifier promotion gate

The staging config must explicitly enable `llm_judge` or `route_llm`; a heuristic config fails because it is not live advanced-classifier evidence. `classifier-live-gate` runs five fixed benign smoke classifications through the credentialed adapter, then evaluates a reproducible 20% holdout from the redacted labeled dataset. It enforces adapter success/fallback rates, smoke p95, holdout size, and tier/domain accuracy.

The same command replaces the configured classifier provider with a credential-free loopback fixture for three mandatory failure injections: timeout, invalid JSON, and HTTP 429. Each must produce exactly one adapter request, zero success, one fallback, and one heuristic route; invalid JSON must additionally increment the invalid-output counter. These tests prove fail-closed behavior without asking a real provider to malfunction.

The uploaded artifact contains only aggregate counts, latencies, accuracies, the dataset filename/fingerprint, config fingerprint, and revision. Raw smoke prompts, holdout inputs/misses, bearer tokens, provider keys, and private headers are not serialized. A failed live adapter still writes and uploads this redacted artifact before the job exits non-zero.

## Stream lifecycle promotion gate

The independent `stream-promotion` job runs `stream-live-gate` through a temporary local router backed by every configured provider/model pair whose endpoint and adapter contracts advertise chat streaming. Each live profile must return OpenAI SSE with `[DONE]` and terminal usage within the configured first-chunk and completion bounds. The command compares the client body byte count and FNV-1a digest with the router's observed passthrough, deliberately cancels a second stream, waits for the active-stream gauge and cancellation counter to converge, then proves the provider concurrency permit was released with a completed follow-up request. Readiness must still report that model viable after cancellation.

The same artifact includes credential-free controlled transport injections for a capped `Retry-After` retry, an incomplete chunked response after a forwarded prefix, and graceful shutdown during an active stream. Those probes require the expected retry count and delay, client/proxy body-error accounting, zero leaked active streams, and bounded shutdown. This separates failure behavior that cannot be safely forced on an external provider from the provider-specific framing, usage, timing, cancellation, capacity, and health observations that are run live.

Native adapters that explicitly reject streaming remain a justified skip. Every jointly advertised streaming profile must pass; a timeout, missing terminal usage, altered body, leaked permit, unhealthy post-cancel state, or failed controlled injection blocks the workflow. The redacted artifact records only provider/model identifiers, reported version headers, aggregate timings/counters/digests, config fingerprint, and candidate revision.
