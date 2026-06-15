# Routes integration skill

Use this skill when you need to route LLM requests through an Autohand Routes gateway instead of calling a single provider directly.

## What Routes is

Routes is an OpenAI-compatible routing layer. It sits in front of multiple models and providers — local inference, hosted APIs, Ollama, vLLM, OpenRouter, Cloudflare AI Gateway, and any OpenAI-compatible endpoint — and picks the best upstream for each request based on capability, cost, latency, privacy, health, and policy.

## When to use this skill

- The user wants a coding agent to use a Routes-managed endpoint.
- The user says "use the router", "point at Routes", "route through Autohand", or gives you a `router-*` model name.
- You need to switch from a direct provider to a policy-driven model selection endpoint.
- You are setting up Routes for a codebase, team, or project.

## Generic integration pattern

Routes exposes an OpenAI-compatible API. Configure the agent exactly as you would configure an OpenAI-compatible provider, but use the Routes base URL and a Routes model name.

Required fields:

- **Base URL**: `http://<router-host>:8080/v1`
- **API key**: the Routes bearer token (optional for localhost, required on a network)
- **Model**: one of the policy model names below, or any configured model alias

## Policy model names

These model names tell Routes to choose the upstream model automatically:

- `auto` — let Routes decide based on the prompt.
- `router-balanced` — general capability/cost/latency tradeoff.
- `router-highest-quality` — strongest available candidate.
- `router-fastest` — lowest-latency healthy candidate.
- `router-lowest-cost` — cheapest candidate that clears capability gates.
- `router-local` — prefer local/self-hosted models.
- `router-privacy` — strongly prefer local/private candidates.
- `router-multimodal` — favor vision, audio, tool, JSON, code, or long-context models.

To bypass routing and hit a specific model, use any configured model id or alias directly.

## Verify Routes is reachable

Before configuring the agent, confirm the router is healthy:

```bash
curl -s http://<router-host>:8080/health
```

And confirm it can classify and route a prompt:

```bash
cargo run -- --config examples/router.yaml classify "Build a small web app from a screenshot"
cargo run -- --config examples/router.yaml route "Design an event sourcing system" --policy highest-quality
```

## Integrating Routes with a codebase

### Where to run Routes

Place Routes close to the model capacity the codebase uses:

- **Local development**: one router on the developer machine in front of Ollama or a local vLLM server.
- **Team/shared workspace**: one router per team, private network segment, or region.
- **GPU nodes**: one router per GPU node so local models are first-class candidates.
- **Enterprise gateway**: one or more routers behind a load balancer with externalized accounting.

Keep the same base URL and policy names across environments. Swap the provider list in config, not the client configuration.

### Config organization

Store routing config in version control and environment-specific overrides outside of it:

- `router.yaml` — base team config with providers, models, policies, and aliases.
- `router.production.yaml` — production overrides (stricter budgets, more providers, external counters).
- Environment variables or secrets manager for provider API keys.

Never commit provider API keys. Routes keeps them server-side; clients only need the Routes bearer token.

### Model aliases and capability mapping

Define aliases that describe capability, not vendor:

```yaml
models:
  - id: gpt-4o
    alias: strong-general
    capabilities: [vision, tools, json, long_context]
  - id: qwen2.5-coder:14b
    alias: local-code
    capabilities: [code]
    local: true
```

Then agents request `router-balanced` or `router-highest-quality` and Routes resolves to the right model based on prompt and policy.

### Per-codebase conventions

- Choose a default policy for the project and document it in `CONTRIBUTING.md` or a `ROUTING.md` file.
- Use `router-local` or `router-privacy` for sensitive repositories.
- Use `router-highest-quality` for architecture reviews, security audits, and complex refactors.
- Use `router-fastest` for quick edits, completions, and low-latency tasks.
- Reserve direct model aliases for cases where the team explicitly wants strict routing.

## Capability and policy best practices

### Match the prompt to a capability

Routes classifies prompts automatically, but you can be explicit. Use `/v1/router/multimodel` or `required_capabilities` when the agent knows what it needs:

- `vision` — screenshots, diagrams, images in the prompt.
- `audio` — speech input/output.
- `tools` / `tool_calls` — function calling.
- `json` — structured output.
- `code` — code generation, review, refactoring.
- `web_apps` — frontend/full-stack generation.
- `long_context` — large files or many files in context.

Example:

```bash
curl -s http://127.0.0.1:8080/v1/router/multimodel \
  -H 'content-type: application/json' \
  -d '{
    "input": "Build a small web app from this screenshot",
    "policy": "multimodal_first",
    "required_capabilities": ["vision", "web_apps"]
  }'
```

### Picking a policy

| Goal | Policy |
|---|---|
| General coding, let Routes decide | `router-balanced` |
| Architecture, security, complex reasoning | `router-highest-quality` |
| Fast completions and small edits | `router-fastest` |
| Cost-sensitive batch work | `router-lowest-cost` |
| Sensitive/private repositories | `router-privacy` |
| Prefer self-hosted models | `router-local` |
| Multimodal tasks | `router-multimodal` |

### Privacy and data residency

- Use `router-privacy` or `router-local` when code, prompts, or outputs must not leave the network.
- Configure `local: true` on self-hosted models so privacy policies can identify them.
- Enable redacted decision traces for production logging.

### Budgets

Set project or process-local budgets in `router.yaml` to reject over-limit requests before upstream dispatch. This prevents runaway costs from agents that loop or generate large outputs.

### Fail-closed fallbacks

Always configure a fallback model. If no candidate is eligible, Routes fails to the fallback rather than sending the request to an unsuitable provider.

### Evaluate before shipping

Before changing routing config, run the eval gate:

```bash
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl
```

This verifies that policy changes still route the codebase's typical prompts correctly.

### Debug with decision traces

When a route looks wrong, inspect the decision trace:

```bash
cargo run -- --config examples/router.yaml route "Your prompt here" --policy balanced
```

Or read the `decision_trace` field from a `/v1/router/multimodel` response. It shows labels, scores, rejected candidates, and why the winner was chosen.

## Per-agent configuration

### Autohand Code CLI

Update `~/.autohand/config.json`:

```json
{
  "provider": "openai",
  "openai": {
    "authMode": "api-key",
    "apiKey": "routes-local-dev-or-bearer-token",
    "baseUrl": "http://router.internal:8080/v1",
    "model": "router-balanced",
    "contextWindow": 128000
  }
}
```

### OpenAI Codex CLI

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

If the Routes build does not accept the `developer` role, add:

```json
"compat": { "supportsDeveloperRole": false }
```

to the provider block.

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

Some Hermes versions read provider base URLs from `~/.hermes/.env`. If the config above does not take effect, set:

```bash
ROUTES_BASE_URL=http://127.0.0.1:8080/v1
ROUTES_API_KEY=routes-local-dev-or-bearer-token
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

OpenClaw requires both the provider definition and the allowlist entry (`agents.defaults.model.primary`). The fully-qualified model ref is `routes/router-balanced`; Routes only sees the model `id`, which is `router-balanced`.

### Other agents

If the agent supports an OpenAI-compatible endpoint, use:

- **Base URL**: `http://<router-host>:8080/v1`
- **API Key**: Routes bearer token
- **Model**: any policy model name or configured alias

## Capability-aware routing

Routes can inspect a prompt and choose a model that supports required capabilities. Use the `/v1/router/multimodel` endpoint when you need to request specific capabilities:

```bash
curl -s http://127.0.0.1:8080/v1/router/multimodel \
  -H 'content-type: application/json' \
  -d '{
    "input": "Build a small web app from this screenshot",
    "policy": "multimodal_first",
    "required_capabilities": ["vision", "web_apps"]
  }'
```

## Troubleshooting

- **Connection refused**: Routes is not running or the host/port is wrong. Check `cargo run -- --config examples/router.yaml serve`.
- **401 Unauthorized**: The Routes bearer token is missing or incorrect. Check the `auth` section of the Routes config.
- **Model not found**: The model name is not a known policy name or configured alias. Use `cargo run -- --config examples/router.yaml validate` to check configured models.
- **Unexpected provider selected**: Include `decision_trace` from the response or run the `classify`/`route` CLI commands to see how Routes scored candidates.
- **Budget exceeded**: Review `accounting` config and token/cost limits. Decision traces include budget eligibility.
- **Capability mismatch**: Use `/v1/router/multimodel` with `required_capabilities` or pick a policy like `router-multimodal` that favors the needed capability.

## See also

- [README.md](README.md) — project overview and quickstart.
- [docs/usage.md](docs/usage.md) — full command and API reference.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to extend Routes.
- [docs/production-readiness.md](docs/production-readiness.md) — production evidence checklist.
