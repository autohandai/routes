# Checked release evidence

Each directory under `baselines/` is named for the exact source revision exercised by the controlled evidence command. The manifest binds the benchmark and conformance JSON with content fingerprints and repeats the source revision and redacted config fingerprint.

The checked baseline is regression evidence for the router plus a loopback mock provider. It is not evidence of external-provider latency, availability, or capacity. Release workflows regenerate the same bundle at the release SHA and attach it to the GitHub release; credentialed provider and deployed-load artifacts remain separate evidence classes.

Validate a baseline with:

```bash
cargo run --locked -- evidence-validate evidence/baselines/<revision> \
  --expected-revision <revision>
```
