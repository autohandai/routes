# Dependency security policy

The release-blocking quality workflow installs `cargo-audit` at an exact version and audits the committed `Cargo.lock`. Warnings and vulnerabilities fail the gate.

Fix or upgrade an affected dependency first. If an advisory demonstrably cannot affect this binary and no fix is available, a temporary exception may be added to `security/advisory-exceptions.txt` using this pipe-delimited contract:

```text
RUSTSEC-YYYY-NNNN|YYYY-MM-DD|owner|reason
```

The owner is accountable for removing the exception. The expiry must be later than the current UTC date; malformed and expired entries fail before the audit runs. Permanent or unowned exceptions are not accepted.
