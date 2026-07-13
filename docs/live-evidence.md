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
