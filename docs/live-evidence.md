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
