# Routes

Routes is an OpenAI-compatible routing layer for teams running more than one model, provider, or inference backend.

It gives you one API in front of local models, hosted providers, Ollama, llama.cpp, vLLM, OpenRouter, Cloudflare AI Gateway, and any service that accepts OpenAI-compatible requests. Instead of hand-wiring every fallback, budget, cache, classifier, provider retry, and model-specific rule into your application, Routes makes those decisions explicit, configurable, measurable, and testable.

We built Routes inside Autohand to battle-test routing across high-volume agent workloads without manually configuring every point of failure. The benchmark suite has exercised the router across 100M routing requests, and the result is a routing system designed around inspectable decisions, eval gates, provider health, multimodal capability checks, and fail-closed fallbacks.

**Use Routes when you want model choice to be a policy you can inspect, evaluate, and improve instead of a pile of provider-specific conditionals.**

## Who It Is For

- AI engineers and developers exploring next-generation routing with local models, hosted models, multimodal inputs, tool use, long context, and OpenAI-compatible clients.
- MLOps teams that need learning-based routing optimization, provider conformance checks, production metrics, budget controls, and custom model-selection strategies.
- Research teams comparing routing policies before production deployment, including quality-first, cost-aware, privacy-first, local-first, and multimodal-first strategies.

## What Routes Solves

Running multiple LLMs is not just picking the biggest model. Real routing needs to answer questions like:

- Is this prompt simple enough for a small local model?
- Does it need vision, audio, JSON mode, tool calls, code strength, web-app generation, or long context?
- Which providers are healthy right now?
- Should this request optimize for latency, cost, privacy, capability, or quality?
- Can we reject over-budget requests before provider dispatch?
- Can we explain why a model was chosen after something goes wrong?
- Can we evaluate routing changes before shipping them?

Routes turns those questions into data-driven routing decisions with diagnostics.

## Highlights

- OpenAI-compatible front door for chat completions, Responses, embeddings, image generation, speech, transcription, and translation.
- Automatic routing through `auto` and `router-*` model names.
- Config-driven provider and model registry with aliases, capabilities, context windows, local/remote metadata, costs, and domain strengths.
- Deterministic local classifier that works without external services.
- Optional LLM classifier or judge model for learned routing experiments.
- Routing policies for balanced, lowest-cost acceptable, fastest healthy, highest-quality, local-first, privacy-first, multimodal-first, and legacy presets.
- Candidate diagnostics that expose labels, score components, required capabilities, context eligibility, rejected candidates, and fallback behavior.
- Provider retries, timeouts, concurrency limits, health sampling, and transient failover for automatic routes.
- Optional semantic cache, sticky routing, safety routing, budgets, shadow evaluation, decision traces, Prometheus metrics, load tests, and eval gates.
- JSON Schema and OpenAPI output for editor integration, CI checks, and client generation.

## Autohand Code Enterprise

Routes powers [Autohand Code Enterprise](https://www.autohand.ai/code/enterprise/) across millions of coding sessions. It acts as the model gateway between Autohand Code clients, local inference nodes, hosted providers, and private model pools so teams can route coding work by policy instead of hard-coding one provider into every developer workflow.

The flagship [Autohand Code CLI](https://github.com/autohandai/code-cli/tree/main/docs) can point at Routes through its OpenAI-compatible provider settings. A typical enterprise setup runs Routes near the available model capacity, such as one router per GPU node, region, or private network segment, then lets Autohand Code choose `auto` or a `router-*` policy model.

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

Use `router-local` for local-first coding work, `router-privacy` for sensitive repositories, `router-fastest` for low-latency edits, or `router-highest-quality` for complex architecture and review tasks. Routes can sit in front of Ollama, llama.cpp, vLLM, OpenRouter, Cloudflare AI Gateway, and OpenAI-compatible providers across local nodes and internet-reachable gateways.

## First 10 Minutes

1. Validate the example config.

   ```bash
   cargo run -- --config examples/router.yaml validate
   ```

2. Ask Routes how it would classify a prompt.

   ```bash
   cargo run -- --config examples/router.yaml classify "Build a small multimodal web app from a screenshot"
   ```

3. Ask Routes which model it would choose.

   ```bash
   cargo run -- --config examples/router.yaml route "Design a production event sourcing system" --policy highest-quality
   ```

4. Run the test suite before changing routing behavior.

   ```bash
   cargo test
   ```

## Quickstart

```bash
cargo run -- init-config router.yaml
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml serve
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

## Example

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

## Routing Policies

Routes ships with policy presets that are controlled by configuration, not hidden constants:

- `balanced`: general-purpose capability, cost, latency, and domain tradeoff.
- `lowest_cost_acceptable`: chooses the cheapest candidate that clears context and capability gates.
- `fastest_healthy`: emphasizes low-latency healthy providers.
- `highest_quality`: favors the strongest available candidate.
- `local_first`: prefers local models when they can satisfy the request.
- `privacy_first`: strongly penalizes remote candidates.
- `multimodal_first`: favors models with vision, audio, tool, JSON, code, web-app, or long-context capabilities.

Legacy policy names remain supported for compatibility: `floor`, `nitro`, `quality`, `cost_efficient`, `capability_heavy`, and `domain_skills`.

## Repository Map

- [src/classifier.rs](src/classifier.rs): deterministic prompt classification and optional model-backed classifier adapters.
- [src/router.rs](src/router.rs): candidate filtering, scoring policies, fallback handling, and route explanations.
- [src/provider.rs](src/provider.rs): provider adapters for OpenAI-compatible, Ollama, llama.cpp, vLLM, OpenRouter, and Cloudflare AI Gateway paths.
- [src/server.rs](src/server.rs): HTTP API, OpenAI-compatible proxy routes, metrics, budgets, cache, safety, and sticky routing.
- [examples/router.yaml](examples/router.yaml): local example config for development and tests.
- [docs/examples/router.production.yaml](docs/examples/router.production.yaml): fuller production-oriented config.

## Documentation

- [Usage guide](docs/usage.md): commands, router APIs, headers, operations, auth, configuration, evaluation, calibration, and provider conformance.
- [Deployment docs](docs/README.md): container packaging and hosting examples.
- [Container runtime](docs/deployment/container.md): production image and config guidance.
- [Monitoring](docs/monitoring/README.md): Prometheus metrics, dashboards, and alerts.
- [Production example config](docs/examples/router.production.yaml): fuller production-oriented config.

Useful local commands:

```bash
cargo run -- --config examples/router.yaml openapi
cargo run -- --config examples/router.yaml config-schema
cargo run -- --config examples/router.yaml eval examples/eval.jsonl
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl
cargo run -- --config examples/router.yaml provider-conformance-matrix
```

## Contributing

Routes is designed to be composable. New routing ideas should usually land behind the classifier boundary, scoring-policy boundary, provider adapter boundary, or eval tooling rather than inside HTTP handlers.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), then choose a small change with a clear validation path. Good first contributions usually add one provider, one eval slice, one routing policy, one docs example, or one focused diagnostic improvement.

Good areas for contributors:

- Provider adapters for more OpenAI-compatible and native inference servers.
- Eval corpora for small web apps, coding agents, multimodal prompts, safety-sensitive prompts, long-context work, and local-model routing.
- Routing policies with clear behavior and measurable tradeoffs.
- Learned scoring features, optimizer experiments, and judge-model evaluation workflows.
- Dashboard panels, deployment recipes, and examples for real local-model setups.

Run the core checks before opening a PR:

```bash
cargo fmt
cargo test
cargo run -- --config examples/router.yaml validate
cargo run -- --config examples/router.yaml openapi
```

## Help and Maintainers

Routes is maintained by Autohand. Use GitHub issues for bugs, design questions, provider compatibility reports, and proposed routing policies. When reporting routing behavior, include the prompt shape, expected policy, selected model/provider, config snippet, and any `decision_trace` fields that explain the choice.

## Project Status

Routes is private while we prepare the first collaborator cohort, but it is already structured for outside contribution: public API contracts, focused docs, local deterministic routing, reproducible eval gates, and a small set of extension boundaries.
