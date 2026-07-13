# Container Packaging

Use the same container image on AWS, Google Cloud, Azure, Cloudflare, or Kubernetes. The router should listen on all interfaces inside the container and the platform should expose port `8080`.

## Minimal Dockerfile

Place this at the repository root as `Dockerfile` when building a deployable image:

```dockerfile
FROM rust:1.87-bookworm AS build
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/target/release/routes /usr/local/bin/routes
COPY docs/examples/router.production.yaml /app/router.yaml

ENV AUTOHAND_ROUTER_CONFIG=/app/router.yaml
EXPOSE 8080
CMD ["routes", "--config", "/app/router.yaml", "serve"]
```

## Production Config Shape

Cloud platforms route traffic to the container port. Set `bind` to `0.0.0.0:8080` in the config file used by the image:

```yaml
bind: 0.0.0.0:8080
default_model: router-balanced
policy: balanced

auth:
  bearer_token_env: [AUTOHAND_ROUTER_TOKEN]

telemetry:
  decision_log_path:
  include_inputs: false

budget:
  max_chat_requests:
  max_total_tokens:
  max_estimated_cost_micros:
  accounting:
    backend: process
    file_path:
    lock_timeout_ms: 1000

providers:
  - name: vllm
    kind: vllm
    base_url: http://vllm.internal:8000
    chat_path: /v1/chat/completions
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    images_path:
    speech_path:
    audio_transcriptions_path: /v1/audio/transcriptions
    audio_translations_path: /v1/audio/translations
    health_path: /health
    timeout_ms: 120000
    retries: 1
    max_concurrency: 8
    queue_timeout_ms: 10000

  - name: openrouter
    kind: openrouter
    base_url: https://openrouter.ai/api
    chat_path: /v1/chat/completions
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    images_path: /v1/images/generations
    speech_path: /v1/audio/speech
    audio_transcriptions_path: /v1/audio/transcriptions
    audio_translations_path: /v1/audio/translations
    health_path: /v1/models
    api_key_env: OPENROUTER_API_KEY
    timeout_ms: 120000
    retries: 2
    max_concurrency: 32
    queue_timeout_ms: 10000

models:
  - id: Qwen/Qwen2.5-Coder-32B-Instruct
    provider: vllm
    aliases: [open-weight-coder, vllm-coder]
    capability: 0.74
    cost_per_million_input: 0.04
    cost_per_million_output: 0.04
    domains: [coding, data, general]
    context_window: 32768
    local: true

  - id: openrouter/anthropic/claude-sonnet-4.5
    provider: openrouter
    aliases: [sonnet, default-strong]
    capability: 0.88
    cost_per_million_input: 3.0
    cost_per_million_output: 15.0
    domains: [coding, design, general]
    context_window: 200000
```

See [../examples/router.production.yaml](../examples/router.production.yaml) for a complete container-ready example.

## Build And Smoke Locally

```bash
docker build -t autohand-router:local .
docker run --rm -p 8080:8080 \
  -e AUTOHAND_ROUTER_TOKEN=dev-token \
  -e AUTOHAND_ROUTER_REVISION="$(git rev-parse HEAD)" \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  autohand-router:local
```

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/router/classify \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer dev-token' \
  -d '{"input":"Add streaming retry logic to this Rust service"}'
```

## Release Gates

Run these before cutting traffic over:

```bash
cargo run -- --config docs/examples/router.production.yaml validate
cargo run -- --config examples/router.yaml eval-gate examples/eval.production.jsonl --min-examples 50 --min-accuracy 0.90 --min-domain-accuracy 0.90 --min-model-accuracy 0.95 --min-provider-accuracy 0.95 --output router.eval-gate.json
cargo run -- --config docs/examples/router.production.yaml provider-conformance-matrix --output router.provider-matrix.json
cargo run -- load-suite --url https://router.example.com --output router.load-suite.json
cargo run -- --config docs/examples/router.production.yaml deployment-live-gate --url https://router.example.com --revision "$(git rev-parse HEAD)" --output router.deployment-gate.json
```
