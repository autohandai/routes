# Routes

Routes was built from the ground up inside Autohand so agents can choose the capability they need, at the moment they need it.

After powering millions of coding sessions, we learned that model choice is not a one-time decision. It is a continuous judgment about capability, cost, latency, privacy, context, and provider health. Routes makes that judgment explicit, configurable, measurable, and testable.

It is an OpenAI-compatible routing layer that sits in front of local models, hosted providers, Ollama, llama.cpp, vLLM, OpenRouter, Cloudflare AI Gateway, and any service that accepts OpenAI-compatible requests. Instead of hard-wiring every fallback, budget, cache, classifier, retry, and model-specific rule into your application, you describe your providers, define your policies, and let Routes decide.

The benchmark suite has exercised the router across 100M routing requests. The result is a system designed around inspectable decisions, eval gates, provider health, multimodal capability checks, and fail-closed fallbacks — so model choice becomes something you can evaluate and improve, not a pile of provider-specific conditionals.

**Use Routes when you want agents to pick the right capability by policy, not by accident.**

## The problem with model choice today

Most teams start with one provider and one model. Then they add a local model for cost. Then a bigger model for hard prompts. Then a vision model. Then a fallback for outages. Before long, model selection is buried in `if/else` branches, hidden provider timeouts, and undocumented budget rules.

Routes replaces those branches with configuration:

- "Use a local model for simple prompts, but fall back to a hosted model when vision or long context is needed."
- "Prefer the fastest healthy provider, unless quality is critical."
- "Reject this request before dispatch if it would exceed the team budget."
- "Explain exactly why this model was chosen, with scores and rejected candidates."

## What Routes does

Routes is an OpenAI-compatible routing layer. You point your client at one URL and use model names like `auto`, `router-balanced`, or `router-highest-quality`. Routes inspects the prompt, checks capability requirements, scores candidates, respects health and budgets, and forwards to the best upstream.

It handles the messy parts of multi-model operation:

| Concern                    | How Routes handles it                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| **Capability matching**    | Vision, audio, JSON mode, tool calls, code strength, web-app generation, long context         |
| **Policy routing**         | Balanced, lowest-cost, fastest, highest-quality, local-first, privacy-first, multimodal-first |
| **Provider health**        | Retries, timeouts, concurrency limits, health sampling, transient failover                    |
| **Operational safety**     | Budgets, safety routing, fail-closed fallbacks, sticky routing                                |
| **Observability**          | Prometheus metrics, decision traces, route diagnostics                                        |
| **Continuous improvement** | Eval gates, calibration, optimization, provider conformance                                   |

## Why Routes is different

1. **Agent-native routing.** Routes is not a load balancer with model names. It classifies prompts, understands capabilities, and lets agents request a capability profile rather than a specific model.

2. **Deterministic by default.** The local classifier works without external services, so routing keeps working when providers are flaky.

3. **Fail-closed, not fail-open.** When no candidate is eligible, Routes falls back to a configured fallback model instead of silently sending requests to the wrong provider.

4. **Configuration as the source of truth.** Scoring weights, policies, provider metadata, and context windows live in config, not code. Change routing behavior without redeploying your application.

5. **Built for evaluation.** Every routing change can be tested against an eval corpus before it reaches production.

## Battle-tested at scale

Routes powers [Autohand Code Enterprise](https://www.autohand.ai/code/enterprise/) across millions of coding sessions. It acts as the model gateway between Autohand Code clients, local inference nodes, hosted providers, and private model pools so teams can route coding work by policy instead of hard-coding one provider into every developer workflow.

Our open source Coding harness agent [Autohand Code CLI](https://github.com/autohandai/code-cli/tree/main/docs) can point at Routes through its OpenAI-compatible provider settings. A typical enterprise setup runs Routes near the available model capacity, such as one router per GPU node, region, or private network segment, then lets Autohand Code choose `auto` or a `router-*` policy model.

```json
{
  "provider": "autohandai",
  "autohandai": {
    "authMode": "api-key",
    "apiKey": "routes-local-dev-or-bearer-token",
    "baseUrl": "http://router.internal:8080/v1",
    "model": "router-balanced",
    "contextWindow": 128000
  }
}
```

Use `router-local` for local-first coding work, `router-privacy` for sensitive repositories, `router-fastest` for low-latency edits, or `router-highest-quality` for complex architecture and review tasks. Routes can sit in front of Ollama, llama.cpp, vLLM, OpenRouter, Cloudflare AI Gateway, and OpenAI-compatible providers across local nodes and internet-reachable gateways.

## Install Routes

The executable is named `routes`. It can be installed from crates.io, downloaded from a GitHub release archive, or built with Cargo from source.

### Prebuilt executables

Release tags publish archives for Linux, macOS, and Windows:

| Platform            | Release asset                            |
| ------------------- | ---------------------------------------- |
| Linux x86_64        | `routes-x86_64-unknown-linux-gnu.tar.gz` |
| macOS Apple Silicon | `routes-aarch64-apple-darwin.tar.gz`     |
| macOS Intel         | `routes-x86_64-apple-darwin.tar.gz`      |
| Windows x86_64      | `routes-x86_64-pc-windows-msvc.zip`      |

Install on Linux:

```bash
curl -L https://github.com/autohandai/routes/releases/latest/download/routes-x86_64-unknown-linux-gnu.tar.gz -o routes.tar.gz
tar -xzf routes.tar.gz
sudo install -m 755 routes-x86_64-unknown-linux-gnu/routes /usr/local/bin/routes
routes --help
```

Install on macOS Apple Silicon:

```bash
curl -L https://github.com/autohandai/routes/releases/latest/download/routes-aarch64-apple-darwin.tar.gz -o routes.tar.gz
tar -xzf routes.tar.gz
sudo install -m 755 routes-aarch64-apple-darwin/routes /usr/local/bin/routes
routes --help
```

Install on macOS Intel by replacing `aarch64-apple-darwin` with `x86_64-apple-darwin`.

Install on Windows PowerShell:

```powershell
Invoke-WebRequest https://github.com/autohandai/routes/releases/latest/download/routes-x86_64-pc-windows-msvc.zip -OutFile routes.zip
Expand-Archive .\routes.zip -DestinationPath .
.\routes-x86_64-pc-windows-msvc\routes.exe --help
```

Move `routes.exe` into a directory on your `PATH` when you want it available from any terminal.

### Cargo install

Cargo users can install the latest stable release from crates.io:

```bash
cargo install autohand-router --locked
routes --help
```

Install a specific canary when testing an upcoming release:

```bash
cargo install autohand-router --version 0.1.0-canary.1 --locked
```

The exact `.crate` source package published to crates.io is also attached to the matching GitHub Release. To build directly from the repository instead:

```bash
cargo install --git https://github.com/autohandai/routes --bin routes --locked
routes init-config router.yaml
routes --config router.yaml validate
```

From a local checkout:

```bash
cargo install --path . --bin routes --locked
routes init-config router.yaml
routes --config router.yaml serve
```

Once `routes serve` is running, any coding agent that supports an OpenAI-compatible endpoint can use it with base URL `http://127.0.0.1:8080/v1` and a policy model such as `router-balanced`, `router-fastest`, `router-local`, or `router-highest-quality`.

### Release channels

The version in `Cargo.toml` and the release tag must match exactly. Stable tags such as `v0.2.0` publish a normal crates.io version and become the latest GitHub Release. Canary tags such as `v0.2.0-canary.1` publish a crates.io prerelease and a GitHub prerelease without replacing the latest stable release.

Every release publishes the `autohand-router` crate to crates.io, generates GitHub release notes, and attaches the matching `.crate` source package, `routes-<tag>-CHANGELOG.md`, `routes-<tag>-RELEASE_NOTES.md`, and platform archives for Linux, macOS, and Windows.

## Get started in 10 minutes

This path does not require provider API keys. It checks the config, exercises the deterministic classifier, shows a route decision, and confirms the project is healthy before you wire it into an application.

1. Confirm the example config is valid.

   ```bash
   cargo run -- --config examples/router.yaml validate
   ```

   This catches YAML mistakes, invalid provider/model references, missing fallbacks, and policy errors before the router accepts traffic.

2. See how Routes reads a prompt.

   ```bash
   cargo run -- --config examples/router.yaml classify "Build a small multimodal web app from a screenshot"
   ```

   The output shows the deterministic labels Routes uses for local routing, such as task type, complexity, modality needs, and safety signals.

3. Ask for a full routing decision.

   ```bash
   cargo run -- --config examples/router.yaml route "Design a production event sourcing system" --policy highest-quality
   ```

   Look for the selected model/provider, the policy that influenced the score, and any rejected candidates. That is the same decision metadata exposed through the router diagnostics.

4. Start the HTTP router.

   ```bash
   cargo run -- --config examples/router.yaml serve
   ```

   In another terminal, verify the OpenAI-compatible surface is live:

   ```bash
   curl -s http://127.0.0.1:8080/v1/router/providers
   ```

5. Run the test suite before changing behavior.

   ```bash
   cargo test
   ```

After these steps, you have validated the default config, inspected a classification, inspected a policy-driven route choice, and started the local API. To route real chat completions, add your providers and keys to a config file, then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`.

## Quickstart

```bash
cargo run -- init-config router.yaml
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml serve
```

If you installed the executable, the equivalent commands are:

```bash
routes init-config router.yaml
routes --config router.yaml validate
routes --config router.yaml serve
```

Then point any OpenAI-compatible client at:

```text
http://127.0.0.1:8080/v1/chat/completions
http://127.0.0.1:8080/v1/responses
http://127.0.0.1:8080/v1/embeddings
http://127.0.0.1:8080/v1/images/generations
http://127.0.0.1:8080/v1/audio/speech
http://127.0.0.1:8080/v1/audio/transcriptions
http://127.0.0.1:8080/v1/audio/translations
```

Use `auto` or a router model name when you want Routes to choose the upstream model:

```text
auto
router-balanced
router-lowest-cost
router-fastest
router-highest-quality
router-local
router-privacy
router-multimodal
```

Passing a configured model id or alias keeps the request strict and forwards to that model.

## Use with your favorite coding agent

Any agent that speaks OpenAI-compatible APIs can route through Routes. Point it at the router URL, pick a policy model, and keep provider keys server-side.

For a complete integration guide — including codebase setup, capability best practices, policy selection, and per-agent configuration — see [SKILL.md](SKILL.md).

### Autohand Code CLI

```json
{
  "provider": "autohandai",
  "autohandai": {
    "authMode": "api-key",
    "apiKey": "routes-local-dev-or-bearer-token",
    "baseUrl": "http://router.internal:8080/v1",
    "model": "router-balanced",
    "contextWindow": 128000
  }
}
```

Use `router-local` for local-first coding work, `router-privacy` for sensitive repositories, `router-fastest` for low-latency edits, or `router-highest-quality` for complex architecture and review tasks.

### OpenAI Codex

Set environment variables before running `codex`:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="routes-local-dev-or-bearer-token"
export OPENAI_MODEL="router-highest-quality"
codex
```

### Pi

Add Routes as a custom provider in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "routes": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "routes-local-dev-or-bearer-token",
      "models": [
        {
          "id": "router-balanced",
          "name": "Routes Balanced",
          "contextWindow": 128000
        }
      ]
    }
  }
}
```

If your Routes build does not understand the `developer` role, add `"compat": { "supportsDeveloperRole": false }` to the provider.

### Aider

```bash
aider --model openai/router-balanced \
      --openai-api-base http://127.0.0.1:8080/v1 \
      --openai-api-key routes-local-dev-or-bearer-token
```

### Cursor

In Cursor settings, add a custom OpenAI-compatible provider:

- Base URL: `http://router.internal:8080/v1`
- API Key: `routes-local-dev-or-bearer-token`
- Model: `router-balanced`

### Hermes Agent

Add Routes as a custom OpenAI-compatible provider in `~/.hermes/config.yaml`:

```yaml
env:
  ROUTES_API_KEY: routes-local-dev-or-bearer-token

inference:
  provider: routes

agents:
  defaults:
    model:
      primary: router-balanced

models:
  providers:
    routes:
      baseUrl: http://127.0.0.1:8080/v1
      apiKey: ${ROUTES_API_KEY}
      api: openai-completions
      models:
        - id: router-balanced
          name: Routes Balanced
          contextWindow: 128000
```

### OpenClaw

Add Routes as a custom provider in `~/.openclaw/openclaw.json`:

```json
{
  "env": {
    "ROUTES_API_KEY": "routes-local-dev-or-bearer-token"
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "routes/router-balanced"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "routes": {
        "baseUrl": "http://127.0.0.1:8080/v1",
        "apiKey": "${ROUTES_API_KEY}",
        "api": "openai-completions",
        "models": [
          {
            "id": "router-balanced",
            "name": "Routes Balanced",
            "contextWindow": 128000
          }
        ]
      }
    }
  }
}
```

### Other agents

If your agent supports an OpenAI-compatible endpoint, configure:

- **Base URL**: `http://<router-host>:8080/v1`
- **API Key**: your Routes bearer token
- **Model**: `auto`, `router-balanced`, `router-highest-quality`, `router-fastest`, `router-local`, `router-privacy`, or any configured model alias

## Example: route by capability

Ask for a multimodal-capable route:

```bash
curl -s http://127.0.0.1:8080/v1/router/multimodel \
  -H 'content-type: application/json' \
  -d '{
    "input": "Build a small web app from this screenshot and explain the architecture",
    "policy": "multimodal_first",
    "required_capabilities": ["vision", "web_apps"]
  }'
```

The response includes the selected model/provider, prompt labels, policy, confidence values, token estimates, context eligibility, capability eligibility, rejected candidates, and a decision trace explaining the score.

For detailed commands, API examples, and operations setup, see [docs/usage.md](docs/usage.md).

## Routing policies

Routes ships with policy presets that are controlled by configuration, not hidden constants:

- `balanced`: general-purpose capability, cost, latency, and domain tradeoff.
- `lowest_cost_acceptable`: chooses the cheapest candidate that clears context and capability gates.
- `fastest_healthy`: emphasizes low-latency healthy providers.
- `highest_quality`: favors the strongest available candidate.
- `local_first`: prefers local models when they can satisfy the request.
- `privacy_first`: strongly penalizes remote candidates.
- `multimodal_first`: favors models with vision, audio, tool, JSON, code, web-app, or long-context capabilities.

Legacy policy names remain supported for compatibility: `floor`, `nitro`, `quality`, `cost_efficient`, `capability_heavy`, and `domain_skills`.

## Who it's for

- **AI engineers and developers** building agents that need model choice without provider lock-in.
- **Platform/MLOps teams** running mixed local and hosted inference who need health checks, retries, budgets, and metrics.
- **Research teams** comparing routing policies and measuring tradeoffs before production deployment.

## Documentation and contribution

- [Usage guide](docs/usage.md): commands, router APIs, headers, operations, auth, configuration, evaluation, calibration, and provider conformance.
- [Deployment docs](docs/README.md): container packaging and hosting examples.
- [Container runtime](docs/deployment/container.md): production image and config guidance.
- [Monitoring](docs/monitoring/README.md): Prometheus metrics, dashboards, and alerts.
- [Examples](examples/): runnable CLI commands, curl requests, agent configs, and minimal config variants.
- [Production example config](docs/examples/router.production.yaml): fuller production-oriented config.
- [Production readiness checklist](docs/production-readiness.md): current evidence and remaining work.

Useful local commands:

```bash
cargo run -- --config examples/router.yaml openapi
cargo run -- --config examples/router.yaml config-schema
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl
cargo run -- --config router.production.yaml configured-eval-gate examples/eval.production.jsonl
cargo run -- runtime-gate
cargo run -- controlled-evidence --revision "$(git rev-parse HEAD)"
cargo run -- evidence-validate artifacts/release-evidence
cargo run -- --config examples/router.yaml provider-conformance-matrix
cargo run -- --config router.with-judge.yaml classifier-live-gate examples/eval.production.jsonl
cargo run -- --config router.production.yaml stream-live-gate --revision "$(git rev-parse HEAD)"
cargo run -- --config router.production.yaml deployment-live-gate --url https://router-staging.example.com --revision "$(git rev-parse HEAD)"
```

Routes is designed to be composable. New routing ideas should usually land behind the classifier boundary, scoring-policy boundary, provider adapter boundary, or eval tooling rather than inside HTTP handlers. Start with [CONTRIBUTING.md](CONTRIBUTING.md), then choose a small change with a clear validation path.

Run the core checks before opening a PR:

```bash
cargo fmt
cargo test
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml openapi
```

## Support

Routes is maintained by Autohand. Open a GitHub issue for bugs, design questions, provider reports, or proposed routing policies. See [CONTRIBUTING.md](CONTRIBUTING.md) for issue templates and what to include.
